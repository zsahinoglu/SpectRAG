from __future__ import annotations
import hashlib
import logging
from pathlib import Path

import numpy as np
import faiss
from rag_system.models import Chunk
from spectrag.embedders.ollama import OllamaEmbedder as _OllamaEmbedder
from config import OllamaConfig

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _EmbeddingCache:
    """Content-hash disk cache of chunk embeddings, one file per model.

    Keyed by the SHA-256 of the chunk text, so re-indexing a corpus after adding
    documents does not re-embed unchanged chunks. Deterministic: a hit returns the
    exact vector a fresh embed would produce, so results never change — only the
    recompute is skipped.
    """

    def __init__(self, cache_dir: str, model: str) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Model in the filename so switching embedders never serves stale vectors.
        # Plain-array .npz (keys + vectors), read with allow_pickle=False. Legacy
        # `<model>.pkl` caches are ignored, never loaded: they are just cache misses.
        self._file = self._dir / f"{model.replace('/', '_')}.npz"
        self._cache: dict[str, np.ndarray] = {}
        if self._file.exists():
            try:
                with np.load(self._file, allow_pickle=False) as npz:
                    keys, vecs = npz["keys"], npz["vectors"]
                if vecs.ndim != 2 or len(keys) != len(vecs):
                    raise ValueError("misaligned embedding cache")
                self._cache = {str(k): v for k, v in zip(keys, vecs)}
            except Exception:  # noqa: BLE001 — a corrupt cache must not abort indexing
                self._cache = {}
        self._dirty = False

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def get(self, key: str) -> np.ndarray | None:
        return self._cache.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        self._cache[key] = np.asarray(vec, dtype=np.float32)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            tmp = self._file.with_name(self._file.name + ".tmp")
            keys = list(self._cache)
            vecs = (np.stack([self._cache[k] for k in keys]).astype(np.float32)
                    if keys else np.empty((0, 0), dtype=np.float32))
            with open(tmp, "wb") as fh:
                np.savez(fh, keys=np.array(keys, dtype=str), vectors=vecs)
            tmp.replace(self._file)
            self._dirty = False
        except OSError:
            pass


class OllamaEmbedder(_OllamaEmbedder):
    """`spectrag.embedders.OllamaEmbedder` configured from `OllamaConfig`.

    Embedding itself (nomic task prefixes, batching, retries, normalization,
    dimension checks) lives in the `spectrag` class. This subclass only adds the
    pipeline helpers the indexing and query pipelines still call (`embed_chunks`
    with its content-hash cache, `build_faiss_index`); ARCH2c moves those into
    the pipelines so any `spectrag.Embedder` can be injected.
    """

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg
        super().__init__(
            cfg.embedding_model,
            host=cfg.base_url,
            # Apply the configured timeout so a stalled Ollama can't hang the run forever.
            timeout=cfg.request_timeout,
            batch_size=cfg.embed_batch_size,
            dimension=cfg.embedding_dim,
        )
        self._emb_cache: _EmbeddingCache | None = None

    def embed(self, texts, kind: str = "document") -> np.ndarray:
        # Legacy behavior: empty input needs no server round-trip; fall back to
        # the configured dimension until the model has reported its own.
        if not isinstance(texts, (str, bytes)) and len(texts) == 0 and self._dim is None:
            return np.empty((0, self.cfg.embedding_dim), dtype=np.float32)
        return super().embed(texts, kind)

    def embed_chunks(self, chunks: list[Chunk]) -> None:
        """Embed all chunks in-place, reusing cached vectors for unchanged text.

        Only chunks whose text is not already in the content-hash cache are sent to
        Ollama; the rest are served from disk. This makes re-indexing after adding
        documents pay embedding cost only for genuinely new chunks.
        """
        if not chunks:
            return
        if self._emb_cache is None:
            self._emb_cache = _EmbeddingCache(
                self.cfg.embedding_cache_dir, self.cfg.embedding_model
            )

        keys = [_text_hash(c.text) for c in chunks]
        miss_idx = [i for i, k in enumerate(keys) if k not in self._emb_cache]

        if miss_idx:
            miss_embs = self.embed([chunks[i].text for i in miss_idx], kind="document")
            for pos, i in enumerate(miss_idx):
                self._emb_cache.put(keys[i], miss_embs[pos])
            self._emb_cache.flush()
            logger.info(
                "Embeddings: %d computed, %d reused from cache",
                len(miss_idx), len(chunks) - len(miss_idx),
            )
        else:
            logger.info("Embeddings: all %d reused from cache", len(chunks))

        for chunk, key in zip(chunks, keys):
            chunk.embedding = self._emb_cache.get(key)

    def build_faiss_index(self, chunks: list[Chunk]) -> faiss.IndexFlatIP:
        """Build an exact inner-product FAISS index from already-embedded chunks."""
        missing = [c.chunk_id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(
                f"{len(missing)} chunk(s) not embedded (e.g. {missing[:3]}) — "
                "call embed_chunks() before build_faiss_index()."
            )
        if not chunks:
            return faiss.IndexFlatIP(self._dim or self.cfg.embedding_dim)
        matrix = np.stack([c.embedding for c in chunks]).astype(np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])   # use the real embedding dim
        index.add(matrix)
        return index
