"""`Embedder` backed by a local Ollama server (extra: `spectrag[ollama]`)."""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from spectrag.embedders._common import check_inputs, to_unit_rows

logger = logging.getLogger(__name__)

__all__ = ["OllamaEmbedder"]

# nomic-embed-text is trained with task-instruction prefixes: documents must be
# embedded as "search_document: ..." and queries as "search_query: ...".
_NOMIC_PREFIXES = {"query": "search_query: ", "document": "search_document: "}


class OllamaEmbedder:
    """Embeds text with an Ollama embedding model (default `nomic-embed-text`).

    Conforms to `spectrag.Embedder`. Batches requests, retries transient failures
    with exponential backoff, and returns L2-normalized float32 rows.

    `query_prefix` / `document_prefix` default to the nomic task prefixes for
    nomic models and to no prefix otherwise. `dimension` is the expected output
    size; the model's real size wins (with a warning) if they differ. `client` is
    any object with Ollama's `embed(model=..., input=...)` method; it is built
    from `host` and `timeout` when omitted.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        batch_size: int = 32,
        dimension: int | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        client: Any = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model = model
        self.batch_size = batch_size
        self._expected_dim = dimension
        self._dim: int | None = None  # observed from the first response
        nomic = "nomic" in model.lower()
        self._prefixes = {
            "query": query_prefix if query_prefix is not None
            else (_NOMIC_PREFIXES["query"] if nomic else ""),
            "document": document_prefix if document_prefix is not None
            else (_NOMIC_PREFIXES["document"] if nomic else ""),
        }
        if client is None:
            try:
                import ollama
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    'OllamaEmbedder needs the "ollama" package: pip install "spectrag[ollama]"'
                ) from exc
            client = ollama.Client(host=host, timeout=timeout)
        self._client = client

    @property
    def model_id(self) -> str:
        return f"ollama:{self.model}"

    @property
    def dimension(self) -> int:
        """Output size reported by the model (one probe request if nothing was embedded yet)."""
        if self._dim is None:
            self.embed(["dimension probe"], kind="query")
        assert self._dim is not None
        return self._dim

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        """One Ollama call, retried; a short or malformed reply is retryable."""
        resp = self._client.embed(model=self.model, input=batch)
        embs = getattr(resp, "embeddings", None)
        if embs is None and isinstance(resp, dict):
            embs = resp.get("embeddings")
        if embs is None:
            raise RuntimeError("Ollama returned no embeddings")
        return to_unit_rows(embs, len(batch))

    def _observe_dim(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
            if self._expected_dim is not None and dim != self._expected_dim:
                logger.warning(
                    "Embedding dim %d of %s differs from the expected %d; using %d.",
                    dim, self.model, self._expected_dim, dim,
                )
        elif dim != self._dim:
            raise RuntimeError(f"Embedding dimension changed mid-run: {dim} != {self._dim}")

    def embed(self, texts: Sequence[str], kind: str = "document") -> np.ndarray:
        """Return a float32 (len(texts), dimension) array of L2-normalized rows."""
        texts = check_inputs(texts, kind)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        prefix = self._prefixes[kind]
        parts = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if prefix:
                batch = [prefix + t for t in batch]
            arr = self._embed_batch(batch)
            self._observe_dim(arr.shape[1])
            parts.append(arr)
            logger.debug("Embedded %d / %d texts", min(start + self.batch_size, len(texts)), len(texts))
        return np.concatenate(parts, axis=0)
