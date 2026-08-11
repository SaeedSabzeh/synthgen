"""Output schemas — the contract the generator holds the model to.

Generated rows are validated against a Pydantic model before they reach a file.
A model asked for a date will happily emit "12/03/1998", "March 12, 1998", or
"the nineties"; the first two are normalised, the third is counted as invalid
and dropped. Without this step a dataset looks fine until something downstream
tries to parse it.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d/%m/%y",
    "%Y/%m/%d",
    "%d %B %Y",
    "%B %d, %Y",
)

_STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True)


def parse_loose_date(value: Any) -> Any:
    """Accept the many date shapes an LLM emits; hand back a real date."""
    if isinstance(value, (date, datetime)) or value is None:
        return value
    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return value  # let Pydantic raise, so the row is counted as invalid


class Person(BaseModel):
    """The default schema: identity-shaped rows for seeding a test database."""

    model_config = _STRICT
    dedup_keys: ClassVar[tuple[str, ...]] = ("full_name", "date_of_birth")

    full_name: str = Field(min_length=3, max_length=80)
    date_of_birth: date
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    street_address: str = Field(min_length=4, max_length=120)
    city: str = Field(min_length=2, max_length=60)
    country: str = Field(min_length=2, max_length=60)
    occupation: str = Field(min_length=2, max_length=60)

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def _coerce_date(cls, value: Any) -> Any:
        return parse_loose_date(value)

    @field_validator("date_of_birth")
    @classmethod
    def _plausible_date(cls, value: date) -> date:
        if not date(1900, 1, 1) <= value <= date.today():
            raise ValueError("date_of_birth outside 1900-today")
        return value


class Transaction(BaseModel):
    model_config = _STRICT
    dedup_keys: ClassVar[tuple[str, ...]] = ("transaction_id",)

    transaction_id: str = Field(min_length=6, max_length=32)
    customer_name: str = Field(min_length=3, max_length=80)
    amount: float = Field(gt=0, lt=100_000)
    currency: Literal["USD", "EUR", "GBP"]
    category: str = Field(min_length=3, max_length=40)
    timestamp: date
    status: Literal["completed", "pending", "refunded", "failed"]

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_date(cls, value: Any) -> Any:
        return parse_loose_date(value)


class SupportTicket(BaseModel):
    """Text-heavy rows — handy for testing NLP pipelines and label balance."""

    model_config = _STRICT
    dedup_keys: ClassVar[tuple[str, ...]] = ("ticket_id", "subject")

    ticket_id: str = Field(min_length=4, max_length=24)
    subject: str = Field(min_length=8, max_length=120)
    body: str = Field(min_length=20, max_length=800)
    product_area: str = Field(min_length=3, max_length=40)
    sentiment: Literal["positive", "neutral", "negative"]
    priority: Literal["low", "medium", "high", "urgent"]


SCHEMAS: dict[str, type[BaseModel]] = {
    "person": Person,
    "transaction": Transaction,
    "ticket": SupportTicket,
}

_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "str": str,
    "integer": int,
    "int": int,
    "number": float,
    "float": float,
    "boolean": bool,
    "bool": bool,
    "date": date,
}


def get_schema(name: str) -> type[BaseModel]:
    try:
        return SCHEMAS[name.lower()]
    except KeyError:
        raise KeyError(f"Unknown schema {name!r}. Available: {sorted(SCHEMAS)}") from None


def schema_from_spec(spec: dict[str, Any] | str | Path) -> type[BaseModel]:
    """Build a model at runtime from a small JSON spec — no Python needed.

    {"name": "Product",
     "dedup_keys": ["sku"],
     "fields": {"sku":      {"type": "string"},
                "price":    {"type": "number", "gt": 0},
                "in_stock": {"type": "boolean"},
                "tier":     {"type": "string", "enum": ["basic", "pro"]}}}
    """
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or "fields" not in spec:
        raise ValueError("Schema spec must be an object with a 'fields' mapping.")

    fields: dict[str, Any] = {}
    for field_name, raw_meta in spec["fields"].items():
        meta = dict(raw_meta or {})
        raw_type = meta.pop("type", "string")
        enum = meta.pop("enum", None)
        annotation = Literal[tuple(enum)] if enum else _TYPE_MAP.get(str(raw_type).lower(), str)
        required = meta.pop("required", True)
        default = ... if required else meta.pop("default", None)
        description = meta.pop("description", None)
        fields[field_name] = (annotation, Field(default, description=description, **meta))

    model = create_model(spec.get("name", "CustomRecord"), __config__=_STRICT, **fields)
    model.dedup_keys = tuple(spec.get("dedup_keys", ()))  # type: ignore[attr-defined]
    return model


def dedup_keys_for(schema: type[BaseModel]) -> tuple[str, ...]:
    keys = tuple(getattr(schema, "dedup_keys", ()) or ())
    return keys or tuple(schema.model_fields)


def field_summary(schema: type[BaseModel]) -> str:
    """Compact, prompt-friendly description of the target shape."""
    lines = []
    for name, info in schema.model_fields.items():
        annotation = getattr(info.annotation, "__name__", str(info.annotation))
        note = f" — {info.description}" if info.description else ""
        lines.append(f"- {name} ({annotation}){note}")
    return "\n".join(lines)


__all__ = [
    "Person",
    "Transaction",
    "SupportTicket",
    "SCHEMAS",
    "get_schema",
    "schema_from_spec",
    "dedup_keys_for",
    "field_summary",
    "parse_loose_date",
]
