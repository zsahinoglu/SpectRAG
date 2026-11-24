"""Hosted-LLM access for the harness (judge, generator, contamination check) and
token accounting for index cost. Only Claude Haiku is used; never OpenAI."""
from __future__ import annotations

import contextlib
import os
import threading

from eval import pins


class MissingKey(RuntimeError):
    pass


def require_key() -> None:
    """Paid runs need SPECTRAG_ANTHROPIC_API_KEY specifically: the evaluator must not
    silently bill the agent's own ANTHROPIC_API_KEY."""
    if not os.environ.get("SPECTRAG_ANTHROPIC_API_KEY"):
        raise MissingKey("SPECTRAG_ANTHROPIC_API_KEY is not set; `run --dev` and `run --release` "
                         "call Claude Haiku (claude-haiku-5-5). Use `python -m eval smoke` for "
                         "an offline run.")


class HaikuLLM:
    """Pinned-model completion client: `complete(user, system=None) -> str`."""

    def __init__(self, model: str, max_tokens: int = 1024, client=None) -> None:
        from config import AnthropicConfig
        from rag_system.llm import make_anthropic_client
        cfg = AnthropicConfig()
        self.model, self.max_tokens = model, max_tokens
        self._client = client or make_anthropic_client(cfg, "the evaluation harness")

    def complete(self, user: str, system: str | None = None) -> str:
        from rag_system.llm import anthropic_complete
        return anthropic_complete(self._client, model=self.model, user=user, system=system,
                                  temperature=pins.JUDGE_TEMPERATURE, max_tokens=self.max_tokens)


class UsageMeter:
    """Counts Anthropic calls/tokens made anywhere in-process (library code included)
    by wrapping the SDK's Messages.create. Harness-side only; no library change."""

    def __init__(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0
        self._lock = threading.Lock()

    def as_dict(self) -> dict:
        return {"llm_calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}

    @contextlib.contextmanager
    def watch(self):
        try:
            from anthropic.resources.messages import Messages
        except ImportError:  # SDK absent: nothing can be called, nothing to count
            yield self
            return
        original = Messages.create
        meter = self

        def create(self_, *a, **kw):
            resp = original(self_, *a, **kw)
            usage = getattr(resp, "usage", None)
            with meter._lock:
                meter.calls += 1
                meter.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                meter.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            return resp

        Messages.create = create
        try:
            yield self
        finally:
            Messages.create = original
