"""Fake OpenAI client so the suite runs offline, with no API key."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import pytest


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]


class FakeCompletions:
    def __init__(self, owner: FakeClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> FakeCompletion:
        with self._owner.lock:
            self._owner.calls.append(kwargs)
            if self._owner.raise_times > 0:
                self._owner.raise_times -= 1
                raise RuntimeError("transient upstream error")
            content = self._owner.handler(kwargs)
        return FakeCompletion([FakeChoice(FakeMessage(content=content))])


class FakeChat:
    def __init__(self, owner: FakeClient) -> None:
        self.completions = FakeCompletions(owner)


@dataclass
class FakeClient:
    handler: Callable[[dict], str]
    raise_times: int = 0
    calls: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.chat = FakeChat(self)

    @property
    def requested_counts(self) -> list[int]:
        """How many records each call asked for, parsed back out of the prompt."""
        import re

        counts = []
        for call in self.calls:
            match = re.search(r"exactly (\d+) records", call["messages"][-1]["content"])
            counts.append(int(match.group(1)) if match else 0)
        return counts


_counter = count(1)


def make_person(index: int | None = None) -> dict:
    i = next(_counter) if index is None else index
    return {
        "full_name": f"Person {i:05d}",
        "date_of_birth": f"19{50 + i % 50:02d}-0{1 + i % 9}-1{i % 10}",
        "email": f"person{i:05d}@example.com",
        "street_address": f"{i} Example Street",
        "city": f"City {i % 40}",
        "country": "Italy",
        "occupation": "Engineer",
    }


def unique_person_handler(kwargs: dict) -> str:
    """Return exactly as many fresh, valid people as the prompt asked for."""
    import re

    prompt = kwargs["messages"][-1]["content"]
    match = re.search(r"exactly (\d+) records", prompt)
    n = int(match.group(1)) if match else 1
    return json.dumps({"records": [make_person() for _ in range(n)]})


def constant_person_handler(kwargs: dict) -> str:
    """Always the same person — exercises the dedup path."""
    import re

    n = int(re.search(r"exactly (\d+) records", kwargs["messages"][-1]["content"]).group(1))
    return json.dumps({"records": [make_person(index=7) for _ in range(n)]})


@pytest.fixture
def client_factory():
    def _factory(handler=unique_person_handler, **kwargs):
        return FakeClient(handler=handler, **kwargs)

    return _factory
