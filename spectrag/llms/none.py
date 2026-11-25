"""`NoLLM`: the explicit "no language model" component."""
from __future__ import annotations

from spectrag.llms._common import LLMBase

__all__ = ["NoLLM", "LLMUnavailableError"]


class LLMUnavailableError(RuntimeError):
    """Raised when text generation is requested from `NoLLM`."""


class NoLLM(LLMBase):
    """Conforms to `spectrag.IndexLLM` but never calls a model.

    Pass it to run SpectRAG fully offline without LLM-derived signals. Pipelines
    check `isinstance(llm, NoLLM)` up front and skip LLM concepts, utility
    questions and answer generation; calling `complete`/`complete_json` anyway
    validates the arguments and then raises `LLMUnavailableError`, so a missed
    check fails loudly instead of producing empty signals.
    """

    model_id = "none:none"

    def _complete(
        self, prompt: str, *, system: str | None, max_tokens: int, temperature: float,
        json_mode: bool,
    ) -> str:
        raise LLMUnavailableError(
            "NoLLM was configured: this operation needs a language model "
            "(use AnthropicLLM or OllamaLLM)."
        )
