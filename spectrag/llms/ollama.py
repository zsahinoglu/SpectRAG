"""`IndexLLM` backed by a local Ollama server (extra: `spectrag[ollama]`)."""
from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from spectrag.llms._common import LLMBase

logger = logging.getLogger(__name__)

__all__ = ["OllamaLLM"]


class OllamaLLM(LLMBase):
    """A local model served by Ollama (default `llama3.1:8b`). Conforms to `spectrag.IndexLLM`.

    Uses Ollama's `generate` endpoint; `complete_json` asks for JSON output
    (`format="json"`) and then validates it like every other backend. Transport
    errors and empty replies are retried `max_retries` times in total with
    exponential backoff. `client` is any object with Ollama's
    `generate(model=..., prompt=..., ...)` method; it is built from `host` and
    `timeout` when omitted.
    """

    _wait = wait_exponential(multiplier=1, min=1, max=10)

    def __init__(
        self,
        model: str = "llama3.1:8b",
        *,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        max_retries: int = 3,
        client: Any = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1 (it counts attempts)")
        self.model = model
        self.max_retries = max_retries
        if client is None:
            try:
                import ollama
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    'OllamaLLM needs the "ollama" package: pip install "spectrag[ollama]"'
                ) from exc
            client = ollama.Client(host=host, timeout=timeout)
        self._client = client

    @property
    def model_id(self) -> str:
        return f"ollama:{self.model}"

    def _complete(
        self, prompt: str, *, system: str | None, max_tokens: int, temperature: float,
        json_mode: bool,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            kwargs["system"] = system
        if json_mode:
            kwargs["format"] = "json"
        for attempt in Retrying(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=self._wait,
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        ):
            with attempt:
                resp = self._client.generate(**kwargs)
                text = getattr(resp, "response", None)
                if text is None and isinstance(resp, dict):
                    text = resp.get("response")
                if not isinstance(text, str):
                    raise RuntimeError("Ollama returned no response text")
        return text
