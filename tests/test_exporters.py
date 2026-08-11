import csv
import json

import pytest

from synthgen.exporters import export

ROWS = [
    {"name": "Ada", "city": "London", "age": 36},
    {"name": "Grace", "city": "New York", "age": 45},
]


def test_csv_roundtrip(tmp_path):
    path = export(ROWS, tmp_path / "out.csv")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["name"] for r in rows] == ["Ada", "Grace"]


def test_jsonl_roundtrip(tmp_path):
    path = export(ROWS, tmp_path / "out.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["city"] == "London"


def test_json_roundtrip(tmp_path):
    path = export(ROWS, tmp_path / "out.json")
    assert json.loads(path.read_text(encoding="utf-8")) == ROWS


def test_nested_directories_are_created(tmp_path):
    path = export(ROWS, tmp_path / "deep" / "nested" / "out.csv")
    assert path.exists()


def test_ragged_rows_keep_every_column(tmp_path):
    path = export([{"a": 1}, {"a": 2, "b": 3}], tmp_path / "out.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "a,b"


def test_unsupported_extension_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="Unsupported output format"):
        export(ROWS, tmp_path / "out.xml")
