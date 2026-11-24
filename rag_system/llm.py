"""Thin shared helpers for the hosted Anthropic (Claude Haiku) backend.

Kept dependency-light: the `anthropic` package is imported lazily so that code
paths and tests that inject a mocked client never need it (or a key).
"""
from __future__ import annotations

import json
import re

from config import AnthropicConfig


def make_anthropic_client(cfg: AnthropicConfig, purpose: str = "this feature"):
    """Build an Anthropic client from config; raise a clear error if no key is set."""
    if not cfg.api_key:
        raise ValueError(
            f"No Anthropic API key for {purpose}: set SPECTRAG_ANTHROPIC_API_KEY "
            "(or ANTHROPIC_API_KEY as a fallback)."
        )
    import anthropic
    kwargs = {"api_key": cfg.api_key, "timeout": cfg.request_timeout}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return anthropic.Anthropic(**kwargs)


def anthropic_complete(client, *, model: str, user: str, system: str | None = None,
                       temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """One Messages-API call; returns the concatenated text blocks."""
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return "".join(
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", "text") == "text"
    )


def parse_concepts_json(raw: str) -> list[str]:
    """Schema-validate concept-extraction output: {"concepts": [str, ...]}.

    Tolerates a ```json fence or prose around the object, but raises ValueError
    on anything that does not match the schema, so the caller retries instead of
    caching a bad result.
    """
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("concept output is not JSON")
        data = json.loads(m.group(0))
    if not isinstance(data, dict) or not isinstance(data.get("concepts"), list):
        raise ValueError("concept output must be an object with a 'concepts' list")
    if not all(isinstance(c, str) for c in data["concepts"]):
        raise ValueError("'concepts' must contain only strings")
    return sorted({c.strip().lower() for c in data["concepts"] if c.strip()})
