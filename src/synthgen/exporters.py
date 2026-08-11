"""Write records out. Format is inferred from the file extension."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

Rows = Sequence[dict]


def _columns(rows: Rows) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def to_csv(rows: Rows, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_columns(rows))
        writer.writeheader()
        writer.writerows(rows)
    return path


def to_jsonl(rows: Rows, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def to_json(rows: Rows, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def to_parquet(rows: Rows, path: str | Path) -> Path:  # pragma: no cover - optional dep
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(path, index=False)
    return path


_WRITERS = {
    ".csv": to_csv,
    ".jsonl": to_jsonl,
    ".ndjson": to_jsonl,
    ".json": to_json,
    ".parquet": to_parquet,
}


def export(rows: Rows, path: str | Path) -> Path:
    suffix = Path(path).suffix.lower()
    try:
        writer = _WRITERS[suffix]
    except KeyError:
        raise ValueError(
            f"Unsupported output format {suffix!r}. Use one of: {', '.join(sorted(_WRITERS))}"
        ) from None
    return writer(rows, path)
