from synthgen.quality import report


def test_empty_input():
    assert report([]).rows == 0


def test_duplicate_rows_are_counted():
    row = {"name": "Ada", "city": "London"}
    result = report([row, dict(row), {"name": "Grace", "city": "NYC"}])
    assert result.exact_duplicate_rows == 1
    assert result.duplicate_rate == 1 / 3


def test_near_constant_field_is_flagged():
    rows = [{"country": "Italy", "name": f"P{i}"} for i in range(30)]
    country = next(f for f in report(rows).fields if f.name == "country")
    assert country.uniqueness < 0.05
    assert country.warning == "near-constant"


def test_dominant_value_is_flagged():
    rows = [{"city": "Milan"} for _ in range(8)] + [{"city": f"C{i}"} for i in range(4)]
    city = next(f for f in report(rows).fields if f.name == "city")
    assert "dominated by" in city.warning


def test_high_cardinality_field_has_no_warning():
    rows = [{"name": f"Person {i}"} for i in range(50)]
    assert report(rows).fields[0].warning == ""


def test_render_and_dict_outputs():
    rows = [{"name": "Ada", "city": "London"}]
    result = report(rows)
    assert "rows: 1" in result.render()
    assert result.as_dict()["fields"]["name"]["unique"] == 1
