"""H6: vectorised query-path pieces must return exactly what the old per-query
full-corpus Python loops returned. The old implementations are kept here as the
reference."""
from __future__ import annotations

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from rag_system.query.graph_expansion import GraphExpander
from rag_system.query.seed_retrieval import SeedRetriever, SparseBM25, _tokenize
from spectrag.models import Chunk
from tests.fakes import FakeEmbedder
from tests.golden_corpus import QUESTIONS, build_index, make_chunks, make_config

_WORDS = ("star fusion helium bread dough yeast soup sauce flour planet orbit "
          "galaxy gravity salt water energy plasma tomato garlic milk").split()


def _synthetic_chunks(n: int, seed: int = 0) -> list[Chunk]:
    rng = np.random.default_rng(seed)
    emb = FakeEmbedder()
    out = []
    for i in range(n):
        text = " ".join(rng.choice(_WORDS, size=int(rng.integers(0, 14))))
        out.append(Chunk(chunk_id=f"c{i}", doc_id=f"d{i // 10}", text=text,
                         position=i, section_path=[], references=[]))
    vecs = emb.embed([c.text for c in out])
    for c, v in zip(out, vecs):
        c.embedding = v
    return out


@pytest.mark.parametrize("query", [
    "star fusion", "bread bread dough", "zzz unknown", "", "salt water salt energy flour",
])
def test_sparse_bm25_scores_equal_rank_bm25(query):
    chunks = _synthetic_chunks(300)
    corpus = [_tokenize(c.text) for c in chunks]
    ref, new = BM25Okapi(corpus), SparseBM25(corpus)
    q = _tokenize(query)
    np.testing.assert_array_equal(new.get_scores(q), ref.get_scores(q))


def test_sparse_bm25_ranking_equal_rank_bm25():
    chunks = _synthetic_chunks(500, seed=3)
    corpus = [_tokenize(c.text) for c in chunks]
    ref, new = BM25Okapi(corpus), SparseBM25(corpus)
    for query in ("star fusion helium", "dough yeast", "galaxy gravity orbit planet"):
        q = _tokenize(query)
        assert list(np.argsort(-new.get_scores(q))) == list(np.argsort(-ref.get_scores(q)))


def test_dense_scores_equal_per_query_stack():
    chunks = _synthetic_chunks(200)
    cfg = make_config()
    cfg.retrieval.hybrid_lexical_weight = 0.0
    sr = SeedRetriever(cfg, FakeEmbedder())
    import faiss
    index = faiss.IndexFlatIP(64)
    index.add(np.stack([c.embedding for c in chunks]).astype(np.float32))
    for q in ("star fusion", "bread dough"):
        _seeds, scores = sr.retrieve(q, chunks, index)
        q_emb = FakeEmbedder().embed([q], kind="query")
        old = (np.stack([c.embedding for c in chunks]).astype(np.float32) @ q_emb.T).squeeze()
        np.testing.assert_array_equal(scores, old)


def test_dense_matrix_cache_invalidates_on_new_corpus():
    sr = SeedRetriever(make_config(), FakeEmbedder())
    a = _synthetic_chunks(20, seed=1)
    b = _synthetic_chunks(20, seed=2)
    m1 = sr._dense_matrix(a)
    assert sr._dense_matrix(a) is m1            # cached, not rebuilt
    m2 = sr._dense_matrix(b)                    # same size and ids, new embeddings
    np.testing.assert_array_equal(m2, np.stack([c.embedding for c in b]))


class _Neo:
    def __init__(self, ids):
        self.ids = ids

    def bfs_expand(self, seeds, hops, cap):
        return self.ids


def test_expander_matches_old_lookup_and_survives_corpus_change():
    cfg = make_config()
    ex = GraphExpander(cfg)
    a = _synthetic_chunks(30)
    ids = ["c5", "c9", "missing", "c1"]
    old = lambda cs: [{c.chunk_id: i for i, c in enumerate(cs)}[x]
                      for x in ids if x in {c.chunk_id for c in cs}]
    assert ex.expand([0], a, _Neo(ids)) == old(a)
    # Same size and boundary ids, different order in the middle: cache must not lie.
    b = [a[0]] + list(reversed(a[1:-1])) + [a[-1]]
    assert ex.expand([0], b, _Neo(ids)) == old(b)


def test_expander_sees_new_middle_id_with_same_length_and_boundaries():
    ex = GraphExpander(make_config())
    a = _synthetic_chunks(30)
    ids = ["c5", "cNEW", "c1"]
    ex.expand([0], a, _Neo(ids))                      # warm the cache
    b = list(a)
    b[10] = Chunk(chunk_id="cNEW", doc_id="d1", text="x", position=10,
                  section_path=[], references=[])
    old = lambda cs: [{c.chunk_id: i for i, c in enumerate(cs)}[x]
                      for x in ids if x in {c.chunk_id for c in cs}]
    assert ex.expand([0], b, _Neo(ids)) == old(b)


def test_expander_rebuilds_on_unknown_id_after_in_place_mutation():
    """Same list object, same length and boundary ids: only the unknown-id rebuild
    can notice the new middle id (the identity/key checks both pass)."""
    ex = GraphExpander(make_config())
    b = _synthetic_chunks(30)
    ids = ["c5", "cNEW2", "c1"]
    old = lambda cs: [{c.chunk_id: i for i, c in enumerate(cs)}[x]
                      for x in ids if x in {c.chunk_id for c in cs}]
    assert ex.expand([0], b, _Neo(ids)) == old(b)     # warms the cache on this exact list
    cached = ex._map
    b[12] = Chunk(chunk_id="cNEW2", doc_id="d1", text="y", position=12,
                  section_path=[], references=[])
    assert ex._map_src is b and ex._map is cached     # cache is stale for b
    result = ex.expand([0], b, _Neo(ids))
    assert result == old(b)
    assert 12 in result
    assert ex._map is not cached                      # the rebuild branch ran


def test_golden_questions_unchanged_through_retrieve():
    """Seeds for the golden corpus equal those from a fresh retriever (no cache state)."""
    cfg, emb, chunks, faiss_index, _store = build_index()
    warm = SeedRetriever(cfg, emb)
    for q in QUESTIONS:
        fresh = SeedRetriever(cfg, emb)
        w, ws = warm.retrieve(q, chunks, faiss_index)
        f, fs = fresh.retrieve(q, chunks, faiss_index)
        assert w == f
        np.testing.assert_array_equal(ws, fs)
