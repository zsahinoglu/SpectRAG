"""Golden + per-stage tests for the query path. Offline: fake embedder, in-memory graph."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rag_system.query.cluster_selection import ClusterAwareSelector
from rag_system.query.diffusion import RelevanceDiffuser
from rag_system.query.graph_expansion import GraphExpander
from rag_system.query.pipeline import QueryPipeline
from rag_system.query.seed_retrieval import SeedRetriever
from tests.fakes import FakeEmbedder
from tests.golden_corpus import QUESTIONS, build_index, offline_encoder

GOLDEN = Path(__file__).parent / "golden" / "retrieval.json"


class _FakeGenerator:
    def generate(self, question, chunks):
        return "ANSWER:" + "|".join(c.chunk_id for c in chunks)

    def close(self):
        pass


def _pipeline(cfg, embedder) -> QueryPipeline:
    p = QueryPipeline.__new__(QueryPipeline)  # skip Ollama/OpenAI construction
    p.embedder = embedder
    p.seed_retriever = SeedRetriever(cfg, embedder)
    p.expander = GraphExpander(cfg)
    p.diffuser = RelevanceDiffuser(cfg)
    p.selector = ClusterAwareSelector(cfg)
    p.selector._enc = offline_encoder()  # avoid tiktoken's data download
    p.generator = _FakeGenerator()
    return p


@pytest.fixture()
def index():
    return build_index()


# ---- fake embedder ---------------------------------------------------------

def test_fake_embedder_deterministic_and_normalised():
    a, b = FakeEmbedder().embed(["red star"]), FakeEmbedder().embed(["red star"])
    assert a.tobytes() == b.tobytes()
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5
    assert FakeEmbedder().embed([]).shape == (0, 64)


def test_fake_embedder_similarity_follows_vocabulary():
    e = FakeEmbedder()
    v = e.embed(["stars fusion helium", "stars fusion energy", "bread yeast dough"])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


# ---- indexing is reproducible ---------------------------------------------

def test_index_build_is_byte_identical(index):
    _, _, c1, _, s1 = index
    _, _, c2, _, s2 = build_index()
    assert [c.embedding.tobytes() for c in c1] == [c.embedding.tobytes() for c in c2]
    assert [c.spectral_coords.tobytes() for c in c1] == [c.spectral_coords.tobytes() for c in c2]
    assert [c.cluster_label for c in c1] == [c.cluster_label for c in c2]
    assert sorted(s1._graph.edges(data="weight")) == sorted(s2._graph.edges(data="weight"))


# ---- Stage 1: seed retrieval ----------------------------------------------

def test_stage1_seeds_are_topical(index):
    cfg, emb, chunks, fidx, _ = index
    seeds, scores = SeedRetriever(cfg, emb).retrieve(QUESTIONS[1], chunks, fidx)
    assert len(seeds) == cfg.retrieval.seed_k
    assert scores.shape == (len(chunks),)
    assert chunks[seeds[0]].doc_id == "cook" and "dough" in chunks[seeds[0]].text.lower()


def test_stage1_dense_only_matches_faiss_order(index):
    cfg, emb, chunks, fidx, _ = index
    cfg.retrieval.uq_seed_enabled = False
    cfg.retrieval.bm25_seed_enabled = False
    cfg.retrieval.hybrid_lexical_weight = 0.0
    seeds, _ = SeedRetriever(cfg, emb).retrieve(QUESTIONS[0], chunks, fidx)
    _, expect = fidx.search(emb.embed([QUESTIONS[0]]), cfg.retrieval.seed_k)
    assert seeds == [int(i) for i in expect[0]]


def test_stage1_utility_question_hit_surfaces_chunk(index):
    cfg, emb, chunks, fidx, _ = index
    cfg.retrieval.bm25_seed_enabled = False
    seeds, _ = SeedRetriever(cfg, emb).retrieve("How do massive stars die?", chunks, fidx)
    assert chunks.index(next(c for c in chunks if c.chunk_id == "astro-2")) in seeds[:2]


# ---- Stage 2: graph expansion ---------------------------------------------

def test_stage2_expansion_contains_seeds_and_respects_cap(index):
    cfg, emb, chunks, fidx, store = index
    seeds, _ = SeedRetriever(cfg, emb).retrieve(QUESTIONS[0], chunks, fidx)
    cands = GraphExpander(cfg).expand(seeds, chunks, store)
    assert set(seeds) <= set(cands)
    cfg.retrieval.max_candidates = len(seeds)
    assert len(GraphExpander(cfg).expand(seeds, chunks, store)) == len(seeds)


def test_stage2_zero_hops_returns_only_seeds(index):
    cfg, emb, chunks, fidx, store = index
    cfg.retrieval.graph_expansion_hops = 0
    assert set(GraphExpander(cfg).expand([0, 3], chunks, store)) == {0, 3}


# ---- Stage 3: diffusion ----------------------------------------------------

def _stage3(index, mode):
    cfg, emb, chunks, fidx, store = index
    cfg.retrieval.diffusion_mode = mode
    seeds, all_scores = SeedRetriever(cfg, emb).retrieve(QUESTIONS[0], chunks, fidx)
    cands = GraphExpander(cfg).expand(seeds, chunks, store)
    sub = store.subgraph({chunks[i].chunk_id for i in cands})
    out = RelevanceDiffuser(cfg).diffuse(cands, seeds, all_scores, chunks, sub)
    return cands, all_scores, out


@pytest.mark.parametrize("mode", ["spectral", "ppr", "hybrid"])
def test_stage3_scores_every_candidate(index, mode):
    cands, _, out = _stage3(index, mode)
    assert set(out) == set(cands)
    assert all(np.isfinite(v) for v in out.values())


def test_stage3_alpha_zero_is_raw_scores(index):
    index[0].retrieval.diffusion_spectral_alpha = 0.0
    cands, all_scores, out = _stage3(index, "hybrid")
    for gi in cands:
        assert out[gi] == pytest.approx(float(all_scores[gi]))


def test_stage3_empty_candidates(index):
    cfg, _, chunks, _, _ = index
    assert RelevanceDiffuser(cfg).diffuse([], [0], np.zeros(len(chunks)), chunks) == {}


# ---- Stage 4: selection ----------------------------------------------------

@pytest.mark.parametrize("mode", ["token_budget", "topk_diffused", "mmr", "protected_mmr", "greedy"])
def test_stage4_selection_modes(index, mode):
    cfg, emb, chunks, fidx, store = index
    cfg.retrieval.selection_mode = mode
    cfg.retrieval.final_context_k = 3
    seeds, all_scores = SeedRetriever(cfg, emb).retrieve(QUESTIONS[0], chunks, fidx)
    cands = GraphExpander(cfg).expand(seeds, chunks, store)
    sub = store.subgraph({chunks[i].chunk_id for i in cands})
    div = RelevanceDiffuser(cfg).diffuse(cands, seeds, all_scores, chunks, sub)
    sel = ClusterAwareSelector(cfg)
    sel._enc = offline_encoder()
    picked = sel.select(cands, div, all_scores, chunks, sub, QUESTIONS[0])
    assert picked and len(picked) == len(set(picked)) and set(picked) <= set(cands)
    if mode != "token_budget":
        assert len(picked) <= 3


def test_stage4_empty_candidates(index):
    cfg, _, chunks, _, _ = index
    assert ClusterAwareSelector(cfg).select([], {}, np.zeros(len(chunks)), chunks, None) == []


def test_stage4_token_budget_is_respected(index):
    cfg, emb, chunks, fidx, store = index
    cfg.retrieval.token_budget = 20  # whitespace tokens; each chunk is ~12–16
    sel = ClusterAwareSelector(cfg)
    sel._enc = offline_encoder()
    scores = {i: 1.0 for i in range(len(chunks))}
    picked = sel.select(list(range(len(chunks))), scores, np.ones(len(chunks)), chunks, None)
    assert 1 <= len(picked) <= 2


def test_coverage_order_negative_scores_no_crash(index):
    # F1/F4 regression: best < 0 made the tie band empty -> IndexError.
    cfg, _, chunks, _, _ = index
    sel = ClusterAwareSelector(cfg)
    sel.token_tie_band = 0.1
    order = sel._coverage_order([0, 1], {0: -0.1, 1: -0.2}, chunks)
    assert sorted(order) == [0, 1]


def test_token_budget_select_negative_scores(index):
    cfg, _, chunks, _, _ = index
    sel = ClusterAwareSelector(cfg)
    sel._enc = offline_encoder()
    sel.token_tie_band = 0.1
    scores = {0: -0.1, 1: -0.2, 2: -0.15}
    picked = sel._token_budget_select([0, 1, 2], scores, chunks)
    assert picked and picked[0] == 0 and set(picked) <= {0, 1, 2}


# ---- Full pipeline: golden --------------------------------------------------

def _run_all(cfg, emb, chunks, fidx, store):
    p = _pipeline(cfg, emb)
    out = {}
    for q in QUESTIONS:
        r = p.query(q, chunks, fidx, store)
        out[q] = {
            "selected": [c.chunk_id for c in r.chunks],
            "scores": {k: round(v, 5) for k, v in r.scores.items()},
            "answer": r.answer,
        }
    return out


def test_pipeline_is_byte_identical_across_runs():
    first = json.dumps(_run_all(*build_index()), sort_keys=True)
    second = json.dumps(_run_all(*build_index()), sort_keys=True)
    assert first == second


def test_pipeline_matches_golden():
    got = _run_all(*build_index())
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
    if not GOLDEN.exists():
        pytest.fail("missing golden file; actual output:\n" + json.dumps(got, indent=2, sort_keys=True))
    assert got == json.loads(GOLDEN.read_text())


def test_pipeline_answers_are_on_topic():
    got = _run_all(*build_index())
    assert got[QUESTIONS[1]]["selected"][0].startswith("cook-")
    assert got[QUESTIONS[0]]["selected"][0].startswith("astro-")
