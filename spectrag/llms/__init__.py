"""`IndexLLM` implementations (ADR 0002, ADR 0005).

    AnthropicLLM  Claude Haiku via the Messages API (default)   core install
    OllamaLLM     local model on an Ollama server               extra: spectrag[ollama]
    NoLLM         no model; LLM-derived signals are skipped     core install

Importing this package does not import any backend; a missing extra fails when
the LLM is constructed, with the `pip install` line to fix it.
"""
from __future__ import annotations

from spectrag.llms.anthropic import AnthropicLLM
from spectrag.llms.none import LLMUnavailableError, NoLLM
from spectrag.llms.ollama import OllamaLLM

__all__ = ["AnthropicLLM", "OllamaLLM", "NoLLM", "LLMUnavailableError"]
