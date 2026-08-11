import json

import pytest
from conftest import constant_person_handler, make_person

from synthgen.generator import SyntheticDataGenerator
from synthgen.schemas import Person


def build(client, **kwargs):
    kwargs.setdefault("batch_size", 10)
    kwargs.setdefault("concurrency", 2)
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("sleep", lambda _s: None)
    return SyntheticDataGenerator(client=client, schema=Person, **kwargs)


def test_generates_the_requested_number(client_factory):
    client = client_factory()
    result = build(client).generate(25)
    assert len(result) == 25
    assert result.stats.returned == 25
    assert all(isinstance(r, Person) for r in result.records)


def test_request_is_split_into_batches(client_factory):
    """A single call for the whole dataset is where silent truncation happens."""
    client = client_factory()
    build(client, batch_size=10).generate(25)
    assert sorted(client.requested_counts) == [5, 10, 10]


def test_duplicates_are_dropped_and_counted(client_factory):
    client = client_factory(handler=constant_person_handler)
    result = build(client, max_topup_rounds=1).generate(10)
    assert len(result) == 1
    assert result.stats.duplicates > 0


def test_invalid_rows_are_dropped_not_stored(client_factory):
    def half_broken(kwargs):
        good = make_person()
        bad = {**make_person(), "email": "definitely not an email"}
        worse = {**make_person(), "date_of_birth": "the nineties"}
        return json.dumps({"records": [good, bad, worse]})

    client = client_factory(handler=half_broken)
    result = build(client, batch_size=3, max_topup_rounds=0).generate(3)
    assert result.stats.invalid == 2  # the bad email and the unparseable date
    assert all(r.email.count("@") == 1 for r in result.records)


def test_topup_rounds_fill_the_gap(client_factory):
    calls = {"n": 0}

    def sparse(kwargs):
        calls["n"] += 1
        # first round returns half of what was asked for
        count = 2 if calls["n"] <= 2 else 5
        return json.dumps({"records": [make_person() for _ in range(count)]})

    client = client_factory(handler=sparse)
    result = build(client, batch_size=5, max_topup_rounds=3).generate(10)
    assert len(result) == 10
    assert client.calls  # more calls than the initial two batches
    assert len(client.calls) > 2


def test_transient_api_errors_are_retried(client_factory):
    client = client_factory(raise_times=2)
    result = build(client, max_retries=3).generate(5)
    assert len(result) == 5
    assert result.stats.retries == 2


def test_permanent_failure_is_recorded_not_raised(client_factory):
    client = client_factory(raise_times=99)
    result = build(client, max_retries=2, max_topup_rounds=0).generate(5)
    assert len(result) == 0
    assert result.errors


def test_json_mode_is_requested(client_factory):
    client = client_factory()
    build(client).generate(5)
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_avoid_list_is_sent_on_topup_rounds(client_factory):
    client = client_factory(handler=constant_person_handler)
    build(client, batch_size=2, max_topup_rounds=2).generate(6)
    later_prompts = [c["messages"][-1]["content"] for c in client.calls[3:]]
    assert any("Do not reuse" in p for p in later_prompts)


def test_stats_are_coherent(client_factory):
    client = client_factory()
    result = build(client).generate(20)
    stats = result.stats
    assert stats.requested == 20
    assert stats.api_calls == 2
    assert 0 < stats.yield_rate <= 1
    assert stats.elapsed_seconds >= 0
    assert set(stats.as_dict()) >= {"requested", "returned", "invalid", "duplicates", "yield_rate"}


def test_zero_rows_is_an_error(client_factory):
    with pytest.raises(ValueError):
        build(client_factory()).generate(0)


def test_to_dicts_is_json_serialisable(client_factory):
    result = build(client_factory()).generate(3)
    json.dumps(result.to_dicts())  # dates must already be strings
