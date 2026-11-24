"""Deterministic, network-free stand-ins used by unit and golden tests."""
from __future__ import annotations

import hashlib
import re

import faiss
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashed bag-of-words embedder: no model, no network, no randomness.

    Each token is hashed (SHA-256, not Python's salted ``hash``) to a bucket;
    the count vector is L2-normalised. Texts sharing vocabulary get high cosine
    similarity, so topical structure in a test corpus is preserved. Mirrors the
    ``OllamaEmbedder`` surface used by the pipeline (``embed``, ``embed_chunks``,
    ``build_faiss_index``). ``kind`` is accepted and ignored.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in _TOKEN_RE.findall(text.lower()):
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % self.dim
            v[bucket] += 1.0  # unsigned: cosines stay >= 0, like term-frequency vectors
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v).astype(np.float32)

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def embed_chunks(self, chunks) -> None:
        for c, e in zip(chunks, self.embed([c.text for c in chunks])):
            c.embedding = e

    def build_faiss_index(self, chunks) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(self.dim)
        index.add(np.stack([c.embedding for c in chunks]).astype(np.float32))
        return index


class WhitespaceEncoder:
    """Stand-in for a tiktoken encoding (which may download data on first use)."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))
