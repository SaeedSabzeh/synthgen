"""Prompt construction, including the anti-mode-collapse hints.

Asking one model for "20 random people" repeatedly returns roughly the same
twenty people: the same handful of names, the same three cities, birthdays
clustered in the 1980s-90s. Each batch here is nudged along rotating axes
(region, decade, life stage) plus a nonce, which is what actually buys diversity
across a few hundred rows.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from synthgen.schemas import field_summary

REGIONS = (
    "Western Europe", "Eastern Europe", "the Middle East", "South Asia",
    "East Asia", "Southeast Asia", "West Africa", "East Africa", "North Africa",
    "South America", "Central America", "North America", "Oceania", "the Nordics",
)
DECADES = ("1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s")
REGISTERS = (
    "working class", "rural", "urban professional", "students", "retirees",
    "small business owners", "public sector workers", "creative freelancers",
)

SYSTEM_TEMPLATE = """You generate synthetic records for software testing.

Return ONLY a JSON object of the form:
{{"records": [ {{...}}, {{...}} ]}}

Each record must have exactly these fields, and no others:
{fields}

Rules:
- Every value must be plausible and internally consistent (the address should
  match the country, the email should relate to the name, dates must be real).
- Dates use ISO-8601: YYYY-MM-DD.
- All records are invented. Do not use real, identifiable living people.
- Vary the records strongly: no repeated names, no repeated cities unless the
  request asks for them, no clustered birthdays.
- No commentary, no markdown fences — JSON only.

JSON Schema for one record:
{json_schema}
"""


def diversity_hint(rng: random.Random, extra: str = "") -> str:
    """One rotating instruction per batch, so batches don't overlap."""
    parts = [
        f"Skew this batch towards {rng.choice(REGIONS)}",
        f"with dates concentrated in the {rng.choice(DECADES)}",
        f"and a {rng.choice(REGISTERS)} feel",
    ]
    hint = ", ".join(parts) + "."
    if extra:
        hint += f" {extra}"
    return hint + f" (batch nonce {rng.randrange(10**6):06d} — do not include it in the output.)"


def build_system_prompt(schema: type[BaseModel]) -> str:
    return SYSTEM_TEMPLATE.format(
        fields=field_summary(schema),
        json_schema=json.dumps(schema.model_json_schema(), indent=2)[:2500],
    )


def build_user_prompt(
    n: int,
    hint: str,
    avoid: Sequence[str] = (),
) -> str:
    prompt = f"Generate exactly {n} records. {hint}"
    if avoid:
        sample = ", ".join(list(avoid)[:25])
        prompt += f"\nDo not reuse any of these already-generated values: {sample}."
    return prompt


def extract_records(payload: Any) -> list[dict]:
    """Be liberal about the wrapper key the model chose; strict about the rows."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("records", "data", "rows", "items", "results", "people"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    for value in payload.values():  # single unknown key holding the list
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return [payload] if payload else []
