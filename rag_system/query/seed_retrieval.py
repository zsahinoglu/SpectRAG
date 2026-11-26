from __future__ import annotations
import re
import numpy as np
import faiss
from spectrag.models import Chunk
from spectrag.protocols import Embedder
from config import Config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for BM25 (lexical, no stemming)."""
    return _TOKEN_RE.findall(text.lower())


class SparseBM25:
    """BM25Okapi with a query path that never loops over the corpus in Python.

    ``rank_bm25.BM25Okapi.get_scores`` builds a per-term term-frequency array with
    a Python comprehension over *every* document, for every query term. Here the
    index (idf, doc lengths, avgdl) is still built by ``BM25Okapi`` once, so those
    values are identical; only the query path changes: term frequencies come from
    per-term postings (doc ids + tf), and only the documents containing a term are
    touched. The arithmetic per touched document is the same expression, in the
    same order, so scores equal ``BM25Okapi.get_scores`` element for element
    (documents without the term get the same 0 contribution).
    """

    def __init__(self, corpus: list[list[str]]) -> None:
        from rank_bm25 import BM25Okapi
        ref = BM25Okapi(corpus)
        self.idf = ref.idf
        self.k1, self.b, self.avgdl = ref.k1, ref.b, ref.avgdl
        self.corpus_size = ref.corpus_size
        self.doc_len = np.asarray(ref.doc_len)
        ids: dict[str, list[int]] = {}
        tfs: dict[str, list[int]] = {}
        for di, freqs in enumerate(ref.doc_freqs):
            for word, tf in freqs.items():
                ids.setdefault(word, []).append(di)
                tfs.setdefault(word, []).append(tf)
        self._postings = {
            w: (np.asarray(ids[w], dtype=np.int64), np.asarray(tfs[w]))
            for w in ids
        }

    def get_scores(self, query: list[str]) -> np.ndarray:
        score = np.zeros(self.corpus_size)
        for q in query:
            post = self._postings.get(q)
            if post is None:
                continue
            doc_ids, tf = post
            idf = self.idf.get(q) or 0
            score[doc_ids] += idf * (
                tf * (self.k1 + 1)
                / (tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc_ids] / self.avgdl))
            )
        return score


class SeedRetriever:
    """Stage 1 — produce initial seed chunk indices.

    Dense FAISS cosine, optionally augmented with utility-question matching:
    each chunk carries LLM-generated "what question does this answer?" strings
    (built at index time for the utility-question edges). Those are query-shaped
    surrogates of the chunk, so matching the user query against them
    (question↔question) bridges the abstractive-query ↔ passage gap that plain
    dense retrieval struggles with. The dense ranking and the utility-question
    ranking are fused with Reciprocal Rank Fusion (rank-based ⇒ no score-scale
    tuning). No query-time LLM cost — the questions already exist on the chunks.
    """

    def __init__(self, cfg: Config, embedder: Embedder) -> None:
        rc = cfg.retrieval
        self.k = rc.seed_k
        self.embedder = embedder
        self.uq_enabled = rc.uq_seed_enabled
        self.uq_k = rc.uq_seed_k
        self.rrf_k = rc.rrf_k
        self.bm25_enabled = rc.bm25_seed_enabled
        self.bm25_k = rc.bm25_seed_k
        self.hybrid_lex_w = rc.hybrid_lexical_weight
        # Utility-question FAISS index, rebuilt when the corpus changes.
        self._uq_index: faiss.Index | None = None
        self._uq_to_chunk: list[int] = []
        self._uq_key: tuple | None = None
        # BM25 lexical index, rebuilt when the corpus changes.
        self._bm25 = None
        self._bm25_key: tuple | None = None
        # Dense embedding matrix, rebuilt when the corpus changes.
        self._emb_matrix: np.ndarray | None = None
        self._emb_key: tuple | None = None
        self._emb_src: list | None = None

    @staticmethod
    def _corpus_key(chunks: list[Chunk]) -> tuple:
        """Content-based cache key for the corpus. Keyed on size + boundary chunk
        ids (unique per corpus) rather than id(chunks): Python reuses an object's
        id() after it is freed, so a reallocated-but-different list could otherwise
        be mistaken for the same corpus and serve a stale index (see CASE-10)."""
        if not chunks:
            return (0,)
        return (len(chunks), chunks[0].chunk_id, chunks[-1].chunk_id)

    def _rrf(self, ranked_lists: list[list[int]]) -> list[int]:
        """Fuse ranked index lists by Reciprocal Rank Fusion (higher = better)."""
        scores: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, idx in enumerate(lst):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def _build_uq_index(self, chunks: list[Chunk]) -> bool:
        """Embed every chunk's utility questions into one FAISS index (cached)."""
        key = self._corpus_key(chunks)
        if self._uq_key == key:
            return self._uq_index is not None

        self._uq_key = key
        self._uq_index = None
        self._uq_to_chunk = []

        questions: list[str] = []
        mapping: list[int] = []
        for ci, c in enumerate(chunks):
            for q in c.utility_questions:
                questions.append(q)
                mapping.append(ci)

        if not questions:
            return False

        # Embed as queries so user-query ↔ utility-question stays symmetric.
        q_embs = self.embedder.embed(questions, kind="query")
        index = faiss.IndexFlatIP(q_embs.shape[1])
        index.add(q_embs.astype(np.float32))
        self._uq_index = index
        self._uq_to_chunk = mapping
        return True

    def _uq_seed_ranking(self, q_emb: np.ndarray, chunks: list[Chunk]) -> list[int]:
        """Chunk indices ranked by best matching utility question (dedup, best rank)."""
        if not self._build_uq_index(chunks):
            return []
        n = min(self.uq_k * 4, len(self._uq_to_chunk))   # over-fetch; dedup to chunks
        _, idx_arr = self._uq_index.search(q_emb, n)
        seen: set[int] = set()
        ranked: list[int] = []
        for qi in idx_arr[0]:
            if qi < 0:
                continue
            ci = self._uq_to_chunk[qi]
            if ci not in seen:
                seen.add(ci)
                ranked.append(ci)
            if len(ranked) >= self.uq_k:
                break
        return ranked

    def _build_bm25(self, chunks: list[Chunk]) -> bool:
        """Build a BM25 index over chunk texts (cached until the corpus changes)."""
        key = self._corpus_key(chunks)
        if self._bm25_key == key:
            return self._bm25 is not None
        self._bm25_key = key
        self._bm25 = None
        corpus = [_tokenize(c.text) for c in chunks]
        if not any(corpus):
            return False
        self._bm25 = SparseBM25(corpus)
        return True

    def _dense_matrix(self, chunks: list[Chunk]) -> np.ndarray:
        """(N, D) float32 embedding matrix, stacked once per corpus rather than
        once per query. The key adds the boundary embeddings' identity to the
        corpus key so re-embedded chunks invalidate the cache. A strong reference
        to the source list is also required (`is`), so id() reuse cannot alias.
        TODO(H4): in-place edits of a middle chunk's embedding are not detected;
        incremental indexing must invalidate explicitly."""
        key = self._corpus_key(chunks) + (
            (id(chunks[0].embedding), id(chunks[-1].embedding)) if chunks else ()
        )
        if (self._emb_key != key or self._emb_matrix is None
                or self._emb_src is not chunks):
            self._emb_matrix = np.stack([c.embedding for c in chunks]).astype(np.float32)
            self._emb_key = key
            self._emb_src = chunks
        return self._emb_matrix

    def _bm25_raw(self, query: str, chunks: list[Chunk]) -> np.ndarray | None:
        """Raw full-corpus BM25 scores, computed once per query and shared by the
        seed ranking and the hybrid relevance blend. None if BM25 is unavailable."""
        if not self._build_bm25(chunks):
            return None
        return np.asarray(self._bm25.get_scores(_tokenize(query)))

    def _bm25_ranking(self, scores: np.ndarray | None) -> list[int]:
        """Chunk indices ranked by BM25 lexical score (positive scores only)."""
        if scores is None:
            return []
        order = np.argsort(-scores)
        return [int(i) for i in order[: self.bm25_k] if scores[i] > 0.0]

    @staticmethod
    def _bm25_scores_full(scores: np.ndarray | None) -> np.ndarray | None:
        """Raw BM25 scores min-floored at 0 and max-normalised to [0,1].
        None if BM25 is unavailable or the query matched nothing lexically."""
        if scores is None:
            return None
        scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
        smax = scores.max()
        return scores / smax if smax > 0 else None

    def retrieve(
        self,
        query: str,
        chunks: list[Chunk],
        faiss_index: faiss.Index,
        trace: dict | None = None,
    ) -> tuple[list[int], np.ndarray]:
        """
        Returns:
            seed_indices   — top-K chunk indices (dense, or RRF-fused with UQ hits)
            all_scores     — (N,) dense cosine vector across all chunks (Stage 3 input)

        `trace`, if given, is filled with {"rankings": {source: [chunk idx, ...]}}
        for explain(); it never affects the result.
        """
        q_emb = self.embedder.embed([query], kind="query")    # (1, D) normalized
        k = min(self.k, len(chunks))

        # Full dense similarity vector — cosine, consumed by diffusion + selection.
        all_embs = self._dense_matrix(chunks)
        all_scores = (all_embs @ q_emb.T).squeeze()    # (N,)

        # Hybrid lexical relevance: fold normalized BM25 into the relevance vector
        # so lexical precision (exact terms) drives Stage-3 diffusion and Stage-4
        # selection, not just the seed choice. Naive keeps a separate pure-dense
        # path, so this is a genuine SpectRAG-only signal.
        bm25_raw = self._bm25_raw(query, chunks) if self.bm25_enabled else None
        if self.hybrid_lex_w > 0 and self.bm25_enabled:
            bm25_all = self._bm25_scores_full(bm25_raw)
            if bm25_all is not None:
                dmax = float(all_scores.max())
                dense_n = all_scores / dmax if dmax > 0 else all_scores
                all_scores = (1.0 - self.hybrid_lex_w) * dense_n + self.hybrid_lex_w * bm25_all

        # Dense ranking (over-fetch a little to give fusion room to work)
        dense_k = min(max(self.k, self.uq_k), len(chunks))
        _, dense_idx = faiss_index.search(q_emb, dense_k)
        dense_rank = [int(i) for i in dense_idx[0] if i >= 0]

        # Fuse dense with the utility-question and BM25 rankings (whichever are
        # enabled and non-empty) via Reciprocal Rank Fusion.
        ranked_lists = [dense_rank]
        named = {"dense": dense_rank}
        if self.uq_enabled:
            uq_rank = self._uq_seed_ranking(q_emb, chunks)
            if uq_rank:
                ranked_lists.append(uq_rank)
                named["utility_question"] = uq_rank
        if self.bm25_enabled:
            bm25_rank = self._bm25_ranking(bm25_raw)
            if bm25_rank:
                ranked_lists.append(bm25_rank)
                named["bm25"] = bm25_rank
        if trace is not None:
            trace["rankings"] = {k: list(v) for k, v in named.items()}

        if len(ranked_lists) == 1:
            return dense_rank[:k], all_scores

        seed_indices = self._rrf(ranked_lists)[:k]
        return seed_indices, all_scores
