"""`IndexLLM` backed by Anthropic's Messages API; the default backend (Claude Haiku)."""
from __future__ import annotations

import os
from typing import Any

from spectrag.llms._common import LLMBase

__all__ = ["AnthropicLLM", "DEFAULT_ANTHROPIC_MODEL"]

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"


def _key_from_env() -> str:
    """SPECTRAG_ANTHROPIC_API_KEY first, ANTHROPIC_API_KEY as a fallback."""
    return os.getenv("SPECTRAG_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or ""


class AnthropicLLM(LLMBase):
    """Claude via the Messages API. Conforms to `spectrag.IndexLLM`.

    The API key is `api_key`, else `SPECTRAG_ANTHROPIC_API_KEY`, else
    `ANTHROPIC_API_KEY`; a missing key fails here, not on the first call.
    Retries of rate limits, 5xx and connection errors are done by the SDK
    (`max_retries`, with its own backoff). `client` is any object with
    `messages.create(**kwargs)`; when given, `api_key`, `base_url`, `timeout`
    and `max_retries` are not used.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        client: Any = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.model = model
        if client is None:
            key = api_key or _key_from_env()
            if not key:
                raise ValueError(
                    "No Anthropic API key: pass api_key= or set SPECTRAG_ANTHROPIC_API_KEY "
                    "(ANTHROPIC_API_KEY is used as a fallback)."
                )
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - core dependency
                raise ImportError(
                    'AnthropicLLM needs the "anthropic" package: pip install "spectrag[anthropic]"'
                ) from exc
            kwargs: dict[str, Any] = {
                "api_key": key, "timeout": timeout, "max_retries": max_retries,
            }
            if base_url:
                kwargs["base_url"] = base_url
            client = anthropic.Anthropic(**kwargs)
        self._client = client

    @property
    def model_id(self) -> str:
        return f"anthropic:{self.model}"

    def _complete(
        self, prompt: str, *, system: str | None, max_tokens: int, temperature: float,
        json_mode: bool,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "text") == "text"
        )
