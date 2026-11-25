"""`Embedder` backed by a local sentence-transformers model.

Needs the `sentence-transformers` package (currently shipped in the
`spectrag[rerank]` extra; see ADR 0004). Runs in-process, no server.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from spectrag.embedders._common import check_inputs, to_unit_rows

__all__ = ["SentenceTransformerEmbedder"]


class SentenceTransformerEmbedder:
    """Embeds text with a `sentence_transformers.SentenceTransformer` model.

    Conforms to `spectrag.Embedder`. `model` is a Hugging Face model name or a
    local path and has no default: choosing one is a quality decision. Pin
    `revision` (a commit hash) for reproducible indexes; it becomes part of
    `model_id`. Asymmetric models (E5, BGE, nomic) need `query_prefix` /
    `document_prefix`. `client` is an already-loaded model object with
    `encode()`; it is loaded from `model`, `revision` and `device` when omitted.
    """

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        query_prefix: str = "",
        document_prefix: str = "",
        client: Any = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model = model
        self.revision = revision
        self.batch_size = batch_size
        self._prefixes = {"query": query_prefix, "document": document_prefix}
        if client is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "SentenceTransformerEmbedder needs the \"sentence-transformers\" package: "
                    'pip install "spectrag[rerank]"'
                ) from exc
            client = SentenceTransformer(model, revision=revision, device=device)
        self._client = client
        dim = getattr(client, "get_sentence_embedding_dimension", lambda: None)()
        self._dim: int | None = int(dim) if dim else None

    @property
    def model_id(self) -> str:
        rev = f"@{self.revision}" if self.revision else ""
        return f"sentence-transformers:{self.model}{rev}"

    @property
    def dimension(self) -> int:
        if self._dim is None:  # model does not report it; probe once
            self.embed(["dimension probe"], kind="query")
        assert self._dim is not None
        return self._dim

    def embed(self, texts: Sequence[str], kind: str = "document") -> np.ndarray:
        """Return a float32 (len(texts), dimension) array of L2-normalized rows."""
        texts = check_inputs(texts, kind)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        prefix = self._prefixes[kind]
        if prefix:
            texts = [prefix + t for t in texts]
        raw = self._client.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = to_unit_rows(raw, len(texts))
        if self._dim is None:
            self._dim = arr.shape[1]
        elif arr.shape[1] != self._dim:
            raise RuntimeError(f"Embedding dimension changed: {arr.shape[1]} != {self._dim}")
        return arr
