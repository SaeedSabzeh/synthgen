import pytest
from pydantic import ValidationError

from synthgen.schemas import (
    Person,
    SupportTicket,
    dedup_keys_for,
    field_summary,
    get_schema,
    parse_loose_date,
    schema_from_spec,
)

VALID = {
    "full_name": "Ada Lovelace",
    "date_of_birth": "1990-05-04",
    "email": "ada@example.com",
    "street_address": "12 Analytical Way",
    "city": "London",
    "country": "United Kingdom",
    "occupation": "Mathematician",
}


@pytest.mark.parametrize("raw", ["1990-05-04", "04/05/1990", "4 May 1990", "May 4, 1990"])
def test_loose_dates_are_normalised(raw):
    """Models follow whatever date format the prompt example implied."""
    person = Person.model_validate({**VALID, "date_of_birth": raw})
    assert person.date_of_birth.year == 1990


def test_impossible_date_is_rejected():
    with pytest.raises(ValidationError):
        Person.model_validate({**VALID, "date_of_birth": "1823-01-01"})


def test_bad_email_is_rejected():
    with pytest.raises(ValidationError):
        Person.model_validate({**VALID, "email": "not-an-email"})


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        Person.model_validate({**VALID, "favourite_colour": "blue"})


def test_missing_field_is_rejected():
    incomplete = {k: v for k, v in VALID.items() if k != "city"}
    with pytest.raises(ValidationError):
        Person.model_validate(incomplete)


def test_dedup_keys_declared_and_not_a_field():
    assert dedup_keys_for(Person) == ("full_name", "date_of_birth")
    assert "dedup_keys" not in Person.model_fields


def test_dedup_keys_fall_back_to_all_fields():
    assert dedup_keys_for(SupportTicket) == ("ticket_id", "subject")


def test_get_schema_unknown_name_lists_options():
    with pytest.raises(KeyError, match="person"):
        get_schema("nope")


def test_custom_schema_from_spec():
    model = schema_from_spec(
        {
            "name": "Product",
            "dedup_keys": ["sku"],
            "fields": {
                "sku": {"type": "string"},
                "price": {"type": "number", "gt": 0},
                "tier": {"type": "string", "enum": ["basic", "pro"]},
            },
        }
    )
    assert model(sku="A1", price=9.99, tier="pro").price == 9.99
    assert dedup_keys_for(model) == ("sku",)
    with pytest.raises(ValidationError):
        model(sku="A1", price=-1, tier="pro")
    with pytest.raises(ValidationError):
        model(sku="A1", price=1, tier="enterprise")


def test_example_spec_file_loads():
    model = schema_from_spec("examples/product_schema.json")
    assert "sku" in model.model_fields


def test_spec_without_fields_is_rejected():
    with pytest.raises(ValueError):
        schema_from_spec({"name": "Broken"})


def test_field_summary_is_prompt_ready():
    summary = field_summary(Person)
    assert "full_name" in summary and "date_of_birth" in summary


def test_parse_loose_date_passes_through_unknown():
    assert parse_loose_date("sometime in the 90s") == "sometime in the 90s"
