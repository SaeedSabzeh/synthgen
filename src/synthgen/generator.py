"""Batched, validated, deduplicated generation."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from synthgen.prompts import build_system_prompt, build_user_prompt, diversity_hint, extract_records
from synthgen.schemas import dedup_keys_for

logger = logging.getLogger(__name__)


class ChatClient(Protocol):
    chat: Any


@dataclass
class GenerationStats:
    requested: int = 0
    returned: int = 0
    invalid: int = 0
    duplicates: int = 0
    api_calls: int = 0
    retries: int = 0
    elapsed_seconds: float = 0.0

    @property
    def yield_rate(self) -> float:
        raw = self.returned + self.invalid + self.duplicates
        return self.returned / raw if raw else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "yield_rate": round(self.yield_rate, 3)}


@dataclass
class GenerationResult:
    records: list[BaseModel]
    stats: GenerationStats
    errors: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in self.records]

    def to_dataframe(self):  # pragma: no cover - thin pandas wrapper
        import pandas as pd

        return pd.DataFrame(self.to_dicts())


class SyntheticDataGenerator:
    """Generate `n` schema-valid, de-duplicated records.

    Splits the request into concurrent batches, validates each row against the
    Pydantic schema, drops duplicates on the schema's dedup keys, and tops up
    with extra batches until the target is met or the budget runs out.
    """

    def __init__(
        self,
        client: ChatClient,
        schema: type[BaseModel],
        model: str = "gpt-4o-mini",
        *,
        batch_size: int = 25,
        concurrency: int = 4,
        max_retries: int = 3,
        temperature: float = 1.0,
        seed: int | None = None,
        max_topup_rounds: int = 3,
        sleep: Any = time.sleep,
    ) -> None:
        self.client = client
        self.schema = schema
        self.model = model
        self.batch_size = max(1, batch_size)
        self.concurrency = max(1, concurrency)
        self.max_retries = max(1, max_retries)
        self.temperature = temperature
        self.max_topup_rounds = max_topup_rounds
        self._rng = random.Random(seed)
        self._sleep = sleep
        self._system_prompt = build_system_prompt(schema)
        self._extra_instructions = ""
        self._errors: list[str] = []
        self._dedup_keys = dedup_keys_for(schema)

    # --- internals --------------------------------------------------------
    def _dedup_signature(self, record: BaseModel) -> tuple:
        data = record.model_dump(mode="json")
        return tuple(str(data.get(key, "")).strip().lower() for key in self._dedup_keys)

    def _call_api(self, n: int, avoid: list[str], stats: GenerationStats) -> list[dict]:
        hint = diversity_hint(self._rng, self._extra_instructions)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": build_user_prompt(n, hint, avoid)},
        ]
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                stats.api_calls += 1
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                return extract_records(json.loads(content))
            except Exception as exc:  # noqa: BLE001 - SDK + JSON errors both retry
                last_error = exc
                stats.retries += 1
                if attempt == self.max_retries - 1:
                    break
                delay = 2**attempt + self._rng.random()
                logger.warning("Batch failed (%s); retrying in %.1fs", exc, delay)
                self._sleep(delay)
        logger.error("Batch permanently failed: %s", last_error)
        self._errors.append(f"{type(last_error).__name__}: {last_error}")
        return []

    def _validate(self, rows: Iterable[dict], stats: GenerationStats) -> list[BaseModel]:
        valid: list[BaseModel] = []
        for row in rows:
            try:
                valid.append(self.schema.model_validate(row))
            except ValidationError as exc:
                stats.invalid += 1
                logger.debug("Dropped invalid row: %s", exc.errors()[:1])
        return valid

    # --- public API -------------------------------------------------------
    def generate(self, n: int, extra_instructions: str = "") -> GenerationResult:
        if n <= 0:
            raise ValueError("n must be positive")

        self._extra_instructions = extra_instructions
        self._errors: list[str] = []
        stats = GenerationStats(requested=n)
        started = time.perf_counter()

        kept: list[BaseModel] = []
        seen: set[tuple] = set()

        for round_index in range(self.max_topup_rounds + 1):
            missing = n - len(kept)
            if missing <= 0:
                break
            if round_index and not self._errors:
                logger.info("Top-up round %d for %d more records", round_index, missing)

            batches = [self.batch_size] * (missing // self.batch_size)
            if missing % self.batch_size:
                batches.append(missing % self.batch_size)
            avoid = [" ".join(sig) for sig in list(seen)[-25:]]

            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = [pool.submit(self._call_api, size, avoid, stats) for size in batches]
                for future in as_completed(futures):
                    for record in self._validate(future.result(), stats):
                        signature = self._dedup_signature(record)
                        if signature in seen:
                            stats.duplicates += 1
                            continue
                        seen.add(signature)
                        kept.append(record)

        stats.returned = len(kept[:n])
        stats.elapsed_seconds = round(time.perf_counter() - started, 3)
        if len(kept) < n:
            logger.warning("Returned %d of %d requested records", len(kept), n)
        return GenerationResult(records=kept[:n], stats=stats, errors=self._errors)
