"""Deterministic offline stand-ins for smoke runs: no model, no network, no randomness."""
from __future__ import annotations

import hashlib
import re

import faiss
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the of in on and or to is are was were what which who how does do did "
                  "for with by from as at that this it its be".split())


class HashEmbedder:
    """Hashed bag-of-words (SHA-256 buckets, stop words dropped), L2-normalised."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in _TOKEN_RE.findall(text.lower()):
            if tok not in _STOP:
                h = hashlib.sha256(tok.encode("utf-8")).digest()
                v[int.from_bytes(h[:4], "big") % self.dim] += 1.0
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
    """tiktoken stand-in (tiktoken may download data on first use)."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class ExtractiveGenerator:
    """Answers with the first sentence of the top-ranked chunk."""

    def generate(self, question: str, chunks) -> str:
        if not chunks:
            return ""
        return re.split(r"(?<=[.!?])\s", chunks[0].text.strip(), maxsplit=1)[0]


class LexicalJudgeLLM:
    """Fake `complete()` client. As judge: prefers the answer sharing more words with the
    question (ties → TIE). As closed-book answerer: always UNKNOWN."""

    def complete(self, user: str, system: str | None = None) -> str:
        if "Answer A:" not in user:
            return "UNKNOWN"
        q = user.split("Question:\n", 1)[1].split("\n\nAnswer A:", 1)[0]
        a = user.split("Answer A:\n", 1)[1].split("\n\nAnswer B:", 1)[0]
        b = user.split("Answer B:\n", 1)[1].split("\n\nWhich answer", 1)[0]
        words = set(_TOKEN_RE.findall(q.lower())) - _STOP
        sa = len(words & set(_TOKEN_RE.findall(a.lower())))
        sb = len(words & set(_TOKEN_RE.findall(b.lower())))
        return "A" if sa > sb else "B" if sb > sa else "TIE"
