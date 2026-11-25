"""Build `spectrag` components from the legacy `Config` (ADR 0006).

This is the only place where the `*_backend` strings of `config.py` are turned
into objects. Every pipeline stage takes an `Embedder` / `IndexLLM` by
injection and never branches on a backend name; the pipelines call these
functions only when the caller did not inject a component.

Internal and provisional: the public construction API is the `SpectRAG`
object and its presets (ARCH3, I4).
"""
from __future__ import annotations

from config import Config
from spectrag.embedders import OllamaEmbedder
from spectrag.llms import AnthropicLLM, NoLLM, OllamaLLM
from spectrag.protocols import Embedder, IndexLLM

__all__ = ["LLM_BACKENDS", "build_embedder", "build_llm", "generation_sampling"]

#: Accepted values of `generation_backend`, `indexing.concept_llm_backend` and
#: `indexing.utility_question_llm_backend`.
LLM_BACKENDS = ("anthropic", "ollama", "none")


def build_embedder(cfg: Config) -> Embedder:
    """The configured Ollama embedder (same model, prefixes and batching as before)."""
    o = cfg.ollama
    return OllamaEmbedder(
        o.embedding_model, host=o.base_url, timeout=o.request_timeout,
        batch_size=o.embed_batch_size, dimension=o.embedding_dim,
    )


def build_llm(cfg: Config, backend: str) -> IndexLLM:
    """An `IndexLLM` for one backend name. Transport retries: `indexing.llm_max_retries`."""
    retries = max(1, cfg.indexing.llm_max_retries)
    if backend == "anthropic":
        a = cfg.anthropic
        return AnthropicLLM(a.model, api_key=a.api_key or None, base_url=a.base_url,
                            timeout=a.request_timeout, max_retries=retries)
    if backend == "ollama":
        o = cfg.ollama
        return OllamaLLM(o.llm_model, host=o.base_url, timeout=o.request_timeout,
                         max_retries=retries)
    if backend == "none":
        return NoLLM()
    if backend == "openai":
        raise ValueError("The OpenAI backend was removed (ARCH2c); use 'anthropic', "
                         "'ollama' or 'none'.")
    raise ValueError(f"Unknown LLM backend {backend!r}; expected one of {LLM_BACKENDS}")


def generation_sampling(cfg: Config) -> tuple[float, int]:
    """(temperature, max_tokens) for answer generation with `cfg.generation_backend`."""
    if cfg.generation_backend == "ollama":
        return cfg.ollama.llm_temperature, cfg.ollama.llm_max_tokens
    return cfg.anthropic.temperature, cfg.anthropic.max_tokens
