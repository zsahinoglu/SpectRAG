"""Helpers shared by the `IndexLLM` implementations (ADR 0005). Not public API.

`LLMBase` gives every implementation the same argument checks and the same
`complete_json` (parse + validate). A backend only implements `_complete`.

The JSON Schema support is a deliberately small subset, so no extra dependency
is needed and nothing is silently ignored: a schema using any other keyword is
rejected before the model is called.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

__all__ = ["LLMBase", "check_schema", "parse_json_reply", "validate"]

_ANNOTATIONS = {"$schema", "$id", "title", "description", "default", "examples"}
_KEYWORDS = {
    "type", "enum", "const",
    "properties", "required", "additionalProperties",
    "items", "minItems", "maxItems", "uniqueItems",
    "minLength", "maxLength",
    "minimum", "maximum",
} | _ANNOTATIONS
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class LLMBase:
    """Argument checks and `complete_json` shared by every implementation."""

    model_id: str

    def _complete(
        self, prompt: str, *, system: str | None, max_tokens: int, temperature: float,
        json_mode: bool,
    ) -> str:
        raise NotImplementedError

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return the model's text reply to a single-turn prompt."""
        _check_args(prompt, system, max_tokens, temperature)
        return self._complete(
            prompt, system=system, max_tokens=max_tokens, temperature=temperature,
            json_mode=False,
        )

    def complete_json(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Any:
        """Return the reply parsed as JSON and validated against `schema`.

        Raises `ValueError` if the schema uses an unsupported keyword (before any
        model call), or if the reply is not JSON or does not validate.
        """
        _check_args(prompt, system, max_tokens, temperature)
        check_schema(schema)
        raw = self._complete(
            prompt, system=system, max_tokens=max_tokens, temperature=temperature,
            json_mode=True,
        )
        data = parse_json_reply(raw)
        validate(data, schema)
        return data

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model_id!r})"


def _check_args(prompt: Any, system: Any, max_tokens: Any, temperature: Any) -> None:
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    if system is not None and not isinstance(system, str):
        raise TypeError(f"system must be str or None, got {type(system).__name__}")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError(f"temperature must be a finite number >= 0, got {temperature!r}")


def parse_json_reply(raw: str) -> Any:
    """Parse a model reply as JSON.

    Accepts a bare JSON value, a ```json fenced block, or one JSON object/array
    surrounded by prose (the first `{` or `[` that starts a complete value).
    Raises `ValueError` otherwise.
    """
    if not isinstance(raw, str):
        raise ValueError(f"reply is not text: {type(raw).__name__}")
    text = _FENCE_RE.sub("", raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            return value
    raise ValueError("reply contains no JSON value")


def check_schema(schema: Any, path: str = "#") -> None:
    """Reject schemas that use keywords outside the supported subset."""
    if not isinstance(schema, Mapping):
        raise ValueError(f"schema at {path} must be a mapping")
    unknown = set(schema) - _KEYWORDS
    if unknown:
        raise ValueError(f"unsupported JSON Schema keyword(s) at {path}: {sorted(unknown)}")
    types = schema.get("type")
    if types is not None:
        for t in [types] if isinstance(types, str) else types:
            if t not in _TYPES:
                raise ValueError(f"unknown type {t!r} at {path}")
    for name, sub in (schema.get("properties") or {}).items():
        check_schema(sub, f"{path}/properties/{name}")
    if "items" in schema:
        check_schema(schema["items"], f"{path}/items")
    extra = schema.get("additionalProperties")
    if extra is not None and not isinstance(extra, bool):
        check_schema(extra, f"{path}/additionalProperties")


def _is_type(value: Any, t: str) -> bool:
    if t == "null":
        return value is None
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer())
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "string":
        return isinstance(value, str)
    if t == "array":
        return isinstance(value, list)
    return isinstance(value, dict)  # "object"


def validate(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate `value` against a schema in the supported subset; raise `ValueError`."""
    def fail(msg: str) -> None:
        raise ValueError(f"reply does not match schema at {path}: {msg}")

    types = schema.get("type")
    if types is not None:
        allowed = [types] if isinstance(types, str) else list(types)
        if not any(_is_type(value, t) for t in allowed):
            fail(f"expected {' or '.join(allowed)}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        fail(f"{value!r} is not one of {schema['enum']!r}")
    if "const" in schema and value != schema["const"]:
        fail(f"{value!r} != {schema['const']!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            fail(f"string shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            fail(f"string longer than {schema['maxLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            fail(f"{value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            fail(f"{value} > maximum {schema['maximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            fail(f"fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            fail(f"more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            seen = [json.dumps(v, sort_keys=True) for v in value]
            if len(set(seen)) != len(seen):
                fail("items are not unique")
        if "items" in schema:
            for i, item in enumerate(value):
                validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for name in schema.get("required") or ():
            if name not in value:
                fail(f"missing required property {name!r}")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in props:
                validate(item, props[name], f"{path}.{name}")
            elif extra is False:
                fail(f"unexpected property {name!r}")
            elif isinstance(extra, Mapping):
                validate(item, extra, f"{path}.{name}")
