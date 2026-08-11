"""Runtime configuration, loaded from the environment (never from source)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "gpt-4o-mini"
_PLACEHOLDER_KEYS = {"api", "your-key-here", "sk-xxx", "changeme", "todo"}


class MissingAPIKeyError(RuntimeError):
    """Raised when no usable OPENAI_API_KEY is available."""


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str = DEFAULT_MODEL
    temperature: float = 1.0  # high on purpose: we want variety, not the modal answer
    batch_size: int = 25
    concurrency: int = 4
    max_retries: int = 3
    request_timeout: float = 60.0

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv_if_available()
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key.lower() in _PLACEHOLDER_KEYS:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set.\n"
                "  cp .env.example .env   # then paste your key into .env\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "Never commit the key itself."
            )
        return cls(
            api_key=key,
            model=os.getenv("SYNTHGEN_MODEL", DEFAULT_MODEL),
            temperature=float(os.getenv("SYNTHGEN_TEMPERATURE", "1.0")),
            batch_size=int(os.getenv("SYNTHGEN_BATCH_SIZE", "25")),
            concurrency=int(os.getenv("SYNTHGEN_CONCURRENCY", "4")),
            max_retries=int(os.getenv("SYNTHGEN_MAX_RETRIES", "3")),
            request_timeout=float(os.getenv("SYNTHGEN_TIMEOUT", "60")),
        )
