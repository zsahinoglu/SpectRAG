from __future__ import annotations
import hashlib
import logging
from pathlib import Path

import numpy as np
import faiss
from spectrag.models import Chunk
from spectrag.embedders.ollama import OllamaEmbedder as _OllamaEmbedder
from spectrag.protocols import Embedder
from config import OllamaConfig

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


class _EmbeddingCache:
    """Content-hash disk cache of chunk embeddings, one file per model.

    Keyed by the SHA-256 of the chunk text, so re-indexing a corpus after adding
    documents does not re-embed unchanged chunks. Deterministic: a hit returns the
    exact vector a fresh embed would produce, so results never change — only the
    recompute is skipped.
    """

    def __init__(self, cache_dir: str, model: str, legacy_model: str | None = None) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Model in the filename so switching embedders never serves stale vectors.
        # Plain-array .npz (keys + vectors), read with allow_pickle=False. Legacy
        # `<model>.pkl` caches are ignored, never loaded: they are just cache misses.
        # `legacy_model` names a pre-ARCH2c file that is read if `model`'s is absent.
        self._file = self._dir / f"{_safe(model)}.npz"
        self._cache: dict[str, np.ndarray] = {}
        candidates = [self._file]
        if legacy_model:
            candidates.append(self._dir / f"{legacy_model.replace('/', '_')}.npz")
        for f in candidates:
            if not f.exists():
                continue
            try:
                with np.load(f, allow_pickle=False) as npz:
                    keys, vecs = npz["keys"], npz["vectors"]
                if vecs.ndim != 2 or len(keys) != len(vecs):
                    raise ValueError("misaligned embedding cache")
                self._cache = {str(k): v for k, v in zip(keys, vecs)}
            except Exception:  # noqa: BLE001 — a corrupt cache must not abort indexing
                self._cache = {}
            break
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


def embed_chunks(embedder: Embedder, chunks: list[Chunk], cache_dir: str | None = None) -> None:
    """Set `chunk.embedding` for every chunk using any `spectrag.Embedder`.

    With `cache_dir`, vectors are cached on disk by text hash in one file per
    `embedder.model_id`, so re-indexing only embeds genuinely new chunks.
    """
    if not chunks:
        return
    if cache_dir is None:
        vecs = embedder.embed([c.text for c in chunks], kind="document")
        for chunk, vec in zip(chunks, vecs):
            chunk.embedding = vec
        return
    model_id = embedder.model_id
    # Pre-ARCH2c caches are named after the bare model ("nomic-embed-text.npz").
    cache = _EmbeddingCache(cache_dir, model_id, legacy_model=model_id.split(":", 1)[-1])
    keys = [_text_hash(c.text) for c in chunks]
    miss_idx = [i for i, k in enumerate(keys) if k not in cache]
    if miss_idx:
        miss_embs = embedder.embed([chunks[i].text for i in miss_idx], kind="document")
        for pos, i in enumerate(miss_idx):
            cache.put(keys[i], miss_embs[pos])
        cache.flush()
    logger.info("Embeddings: %d computed, %d reused from cache",
                len(miss_idx), len(chunks) - len(miss_idx))
    for chunk, key in zip(chunks, keys):
        chunk.embedding = cache.get(key)


def build_faiss_index(chunks: list[Chunk], dimension: int) -> faiss.IndexFlatIP:
    """Exact inner-product FAISS index over already-embedded chunks (row i = chunks[i])."""
    missing = [c.chunk_id for c in chunks if c.embedding is None]
    if missing:
        raise ValueError(
            f"{len(missing)} chunk(s) not embedded (e.g. {missing[:3]}) — "
            "call embed_chunks() before build_faiss_index()."
        )
    if not chunks:
        return faiss.IndexFlatIP(dimension)
    matrix = np.stack([c.embedding for c in chunks]).astype(np.float32)
    index = faiss.IndexFlatIP(matrix.shape[1])   # use the real embedding dim
    index.add(matrix)
    return index


class OllamaEmbedder(_OllamaEmbedder):
    """Deprecated: `spectrag.embedders.OllamaEmbedder` configured from `OllamaConfig`.

    Kept for callers that build an embedder from `OllamaConfig` and use the
    `embed_chunks`/`build_faiss_index` methods; the pipelines now use the
    module-level functions with any `spectrag.Embedder` (ARCH2c).
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

    def embed(self, texts, kind: str = "document") -> np.ndarray:
        # Legacy behavior: empty input needs no server round-trip; fall back to
        # the configured dimension until the model has reported its own.
        if not isinstance(texts, (str, bytes)) and len(texts) == 0 and self._dim is None:
            return np.empty((0, self.cfg.embedding_dim), dtype=np.float32)
        return super().embed(texts, kind)

    def embed_chunks(self, chunks: list[Chunk]) -> None:
        """Module-level `embed_chunks` with the configured `embedding_cache_dir`."""
        embed_chunks(self, chunks, self.cfg.embedding_cache_dir)

    def build_faiss_index(self, chunks: list[Chunk]) -> faiss.IndexFlatIP:
        """Module-level `build_faiss_index`."""
        return build_faiss_index(chunks, self._dim or self.cfg.embedding_dim)
