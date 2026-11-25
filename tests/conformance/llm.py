"""Behavioral conformance suite for `spectrag.IndexLLM` (ADR 0002, ADR 0005).

`isinstance(x, IndexLLM)` only checks member names. `check_llm` checks the
contract the pipelines rely on. It takes a factory `make(replies)` that returns
an implementation whose backend answers successive calls with the given reply
strings (a fake client in unit tests). `check_no_llm` checks `NoLLM`-style
components that refuse to generate. Both raise `AssertionError` on a violation.
"""
from __future__ import annotations

import inspect
from typing import Callable

import pytest

from spectrag.llms import LLMUnavailableError
from spectrag.protocols import IndexLLM

SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "score": {"type": "integer", "minimum": 0},
    },
    "required": ["concepts"],
    "additionalProperties": False,
}

GOOD = [
    ('{"concepts": ["graph", "spectrum"]}', {"concepts": ["graph", "spectrum"]}),
    ('```json\n{"concepts": [], "score": 2}\n```', {"concepts": [], "score": 2}),
    ('Here you go:\n{"concepts": ["ünïcödé"]}\nDone.', {"concepts": ["ünïcödé"]}),
]
BAD = [
    "not json at all",
    '{"concepts": ["a", "b"',              # truncated (max_tokens hit)
    '{"score": 1}',                         # missing required
    '{"concepts": "graph"}',                # wrong type
    '{"concepts": [1, 2]}',                 # wrong item type
    '{"concepts": ["a", "b", "c", "d"]}',   # too many items
    '{"concepts": [], "score": true}',      # bool is not an integer
    '{"concepts": [], "score": -1}',        # below minimum
    '{"concepts": [], "extra": 1}',         # additionalProperties: false
    '["concepts"]',                         # array, not object
]


def _check_shape(llm) -> None:
    assert isinstance(llm, IndexLLM), "missing IndexLLM members"
    assert isinstance(llm.model_id, str) and ":" in llm.model_id, (
        "model_id must be '<provider>:<model>' (ADR 0005)"
    )
    for name in ("complete", "complete_json"):
        params = inspect.signature(getattr(llm, name)).parameters
        for kw in ("system", "max_tokens", "temperature"):
            assert params[kw].kind is inspect.Parameter.KEYWORD_ONLY, f"{name}({kw}) keyword-only"
        assert params["max_tokens"].default == 1024 and params["temperature"].default == 0.0


def _check_argument_validation(llm) -> None:
    for call in (lambda: llm.complete(b"bytes"), lambda: llm.complete("p", system=3)):
        with pytest.raises(TypeError):
            call()
    for bad in (
        {"max_tokens": 0}, {"max_tokens": 2.5}, {"max_tokens": True},
        {"temperature": -0.1}, {"temperature": float("nan")},
    ):
        with pytest.raises(ValueError):
            llm.complete("p", **bad)
        with pytest.raises(ValueError):
            llm.complete_json("p", SCHEMA, **bad)
    with pytest.raises(ValueError, match="unsupported"):
        llm.complete_json("p", {"type": "object", "patternProperties": {}})
    with pytest.raises(ValueError, match="unsupported"):
        llm.complete_json("p", {"type": "array", "items": {"oneOf": []}})


def check_llm(make: Callable[[list[str]], object]) -> None:
    _check_shape(make([]))
    # argument and schema errors are raised before any backend call
    _check_argument_validation(make([]))

    llm = make(["Hello.", "  spaced reply \n", ""])
    assert llm.complete("Say hello.") == "Hello."
    assert llm.complete("p", system="Be brief.", max_tokens=5, temperature=0.7) == (
        "  spaced reply \n"
    ), "complete() returns the reply text unchanged"
    assert llm.complete("p") == "", "an empty reply is a valid text reply"

    llm = make([raw for raw, _ in GOOD])
    for _, expected in GOOD:
        assert llm.complete_json("p", SCHEMA, system="JSON only.") == expected

    for raw in BAD:
        with pytest.raises(ValueError):
            make([raw]).complete_json("p", SCHEMA)

    # an empty schema accepts any JSON value
    assert make(["[1, null, \"x\"]"]).complete_json("p", {}) == [1, None, "x"]


def check_no_llm(llm) -> None:
    _check_shape(llm)
    _check_argument_validation(llm)
    with pytest.raises(LLMUnavailableError):
        llm.complete("p")
    with pytest.raises(LLMUnavailableError):
        llm.complete_json("p", SCHEMA)
