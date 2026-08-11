import random

from synthgen.prompts import build_system_prompt, build_user_prompt, diversity_hint, extract_records
from synthgen.schemas import Person


def test_system_prompt_contains_fields_and_json_instruction():
    prompt = build_system_prompt(Person)
    assert "records" in prompt
    assert "full_name" in prompt
    assert "ISO-8601" in prompt


def test_user_prompt_states_exact_count_and_avoid_list():
    prompt = build_user_prompt(12, "hint.", avoid=["Ada Lovelace"])
    assert "exactly 12 records" in prompt
    assert "Ada Lovelace" in prompt


def test_diversity_hints_differ_between_batches():
    rng = random.Random(0)
    hints = {diversity_hint(rng) for _ in range(20)}
    assert len(hints) > 15  # this is what stops every batch looking the same


def test_diversity_hint_is_reproducible_with_a_seed():
    assert diversity_hint(random.Random(42)) == diversity_hint(random.Random(42))


def test_extract_records_handles_wrapper_variants():
    row = {"a": 1}
    assert extract_records({"records": [row]}) == [row]
    assert extract_records({"people": [row]}) == [row]          # a key models often invent
    assert extract_records({"anything_else": [row]}) == [row]
    assert extract_records([row]) == [row]
    assert extract_records({}) == []
    assert extract_records("nonsense") == []
