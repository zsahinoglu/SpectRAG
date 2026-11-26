"""I5: QueryResult.explain() — provenance per selected chunk, no effect on ranking."""
from __future__ import annotations

import dataclasses

import pytest

from spectrag.models import ChunkExplanation
from tests.golden_corpus import QUESTIONS, build_index
from tests.test_golden_retrieval import _pipeline

SOURCES = {"dense", "utility_question", "bm25", "graph_expansion"}


def _run(q=QUESTIONS[0]):
    cfg, emb, chunks, fidx, store = build_index()
    return _pipeline(cfg, emb).query(q, chunks, fidx, store)


@pytest.mark.parametrize("q", QUESTIONS)
def test_every_selected_chunk_has_source_and_score_parts(q):
    r = _run(q)
    ex = r.explain()
    assert r.chunks and [e.chunk_id for e in ex] == [c.chunk_id for c in r.chunks]
    for e in ex:
        assert isinstance(e, ChunkExplanation)
        assert e.seed_source and set(e.seed_source) <= SOURCES
        if e.seed_source == ("graph_expansion",):
            assert e.seed_ranks == {}
        else:
            assert set(e.seed_ranks) == set(e.seed_source)
            assert all(rank >= 1 for rank in e.seed_ranks.values())
        assert {"relevance", "diffused", "final"} <= set(e.score_parts)
        assert e.score_parts["final"] == pytest.approx(r.scores[e.chunk_id])


def test_proximity_parts_present_in_hybrid_mode():
    ex = _run().explain()
    assert any("spectral_proximity" in e.score_parts for e in ex)
    assert any("ppr_proximity" in e.score_parts for e in ex)


def test_explanation_is_frozen():
    e = _run().explain()[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.chunk_id = "x"


def test_explain_does_not_change_result():
    # Same selection and scores whether or not anyone calls explain().
    a, b = _run(), _run()
    b.explain()
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]
    assert a.scores == b.scores
