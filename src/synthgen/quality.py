"""A quality report for a generated dataset.

Synthetic data fails quietly: 500 rows that are really 40 rows repeated, or a
"country" column that is 90% one value. This turns that into numbers you can
look at before shipping the dataset anywhere.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class FieldReport:
    name: str
    filled: int
    unique: int
    uniqueness: float
    top_values: list[tuple[str, int]]

    @property
    def warning(self) -> str:
        if self.uniqueness < 0.05:
            return "near-constant"
        if self.top_values and self.top_values[0][1] / max(self.filled, 1) > 0.5:
            return f"dominated by {self.top_values[0][0]!r}"
        return ""


@dataclass
class QualityReport:
    rows: int
    exact_duplicate_rows: int
    fields: list[FieldReport]

    @property
    def duplicate_rate(self) -> float:
        return self.exact_duplicate_rows / self.rows if self.rows else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "duplicate_rate": round(self.duplicate_rate, 3),
            "fields": {
                f.name: {
                    "filled": f.filled,
                    "unique": f.unique,
                    "uniqueness": round(f.uniqueness, 3),
                    "top_values": f.top_values,
                    "warning": f.warning,
                }
                for f in self.fields
            },
        }

    def render(self) -> str:
        lines = [
            f"rows: {self.rows}",
            f"exact duplicate rows: {self.exact_duplicate_rows} ({self.duplicate_rate:.1%})",
            "",
            f"{'field':<20}{'unique':>8}{'uniq%':>8}  top value",
            "-" * 66,
        ]
        for f in self.fields:
            top = f.top_values[0] if f.top_values else ("-", 0)
            flag = f"  <- {f.warning}" if f.warning else ""
            lines.append(
                f"{f.name:<20}{f.unique:>8}{f.uniqueness:>7.0%}  "
                f"{str(top[0])[:24]} ({top[1]}){flag}"
            )
        return "\n".join(lines)


def report(rows: Sequence[dict], top_n: int = 3) -> QualityReport:
    rows = list(rows)
    if not rows:
        return QualityReport(rows=0, exact_duplicate_rows=0, fields=[])

    signatures = Counter(tuple(sorted(r.items(), key=lambda kv: kv[0])) for r in rows)
    duplicates = sum(count - 1 for count in signatures.values() if count > 1)

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    fields: list[FieldReport] = []
    for column in columns:
        values = [str(r.get(column)) for r in rows if r.get(column) not in (None, "")]
        counter = Counter(values)
        fields.append(
            FieldReport(
                name=column,
                filled=len(values),
                unique=len(counter),
                uniqueness=len(counter) / len(values) if values else 0.0,
                top_values=counter.most_common(top_n),
            )
        )

    return QualityReport(rows=len(rows), exact_duplicate_rows=duplicates, fields=fields)
