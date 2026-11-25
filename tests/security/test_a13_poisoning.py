"""R1 / A13 — poisoning and hub concepts. Must be made to pass by H7.

Threat: an attacker gets one document into the corpus (an upload, a scraped page,
a shared drive). It is lexically unrelated to what users ask, so dense/BM25 seeds
never pick it. Instead it buys graph edges to everything — by steering the
index-time concept extractor into listing every popular concept, and by citing
"[1] [2] [3] ..." which collide with every other document's local numbering —
and then rides graph expansion + diffusion into the context of unrelated queries.

Separately, a *benign* concept that occurs in most chunks (a hub) turns the
entity signal into a near-complete graph, washing out the spectral structure.

Tests marked ``H7`` fail today (strict xfail). H7 makes them pass and removes the
marker; an unmarked test here is a control that must keep passing.
"""
from __future__ import annotations

import pytest

from config import Config
from rag_system.indexing.edges.citation import CitationEdgeExtractor
from rag_system.indexing.edges.entity import EntityEdgeExtractor
from rag_system.indexing.graph_builder import GraphBuilder
from spectrag.llms import AnthropicLLM
from tests.golden_corpus import QUESTIONS
from tests.test_golden_retrieval import _pipeline
from tests.security._fixtures import (
    POISON_ID, ScriptedLLM, build_poisoned_index, chunk, concepts_json,
)

H7 = pytest.mark.xfail(strict=True, reason="R1/A13 open: H7 (untrusted-content defenses) "
                                           "must make this pass, then drop this marker")


def _selected(monkeypatch, tmp_path, linked: bool) -> dict[str, list[str]]:
    cfg, emb, chunks, fidx, store, _ = build_poisoned_index(monkeypatch, tmp_path, linked=linked)
    p = _pipeline(cfg, emb)
    return {q: [c.chunk_id for c in p.query(q, chunks, fidx, store).chunks] for q in QUESTIONS}


# ---- a poisoned document --------------------------------------------------------

def test_control_unlinked_ad_stays_out_of_context(monkeypatch, tmp_path):
    """Sanity check for the attack below: without the injection and citations the
    ad is irrelevant to every golden question and is never selected. If this
    fails, the poisoning test proves nothing."""
    for q, ids in _selected(monkeypatch, tmp_path, linked=False).items():
        assert POISON_ID not in ids, q


@H7
def test_poisoned_hub_document_does_not_enter_unrelated_contexts(monkeypatch, tmp_path):
    hits = sorted(q for q, ids in _selected(monkeypatch, tmp_path, linked=True).items()
                  if POISON_ID in ids)
    assert not hits, (
        f"injected ad {POISON_ID!r} was put into the LLM context for "
        f"{len(hits)}/{len(QUESTIONS)} unrelated questions: {hits}"
    )


@H7
def test_injected_document_cannot_buy_edges_to_most_of_the_corpus(monkeypatch, tmp_path):
    """Graph-level view of the same attack: one chunk must not become adjacent to
    a majority of the corpus because its text asked for it."""
    _, _, chunks, _, _, graph = build_poisoned_index(monkeypatch, tmp_path, linked=True)
    others = len(chunks) - 1
    deg = graph.degree(POISON_ID)
    assert deg <= others // 2, f"{POISON_ID} has degree {deg}/{others}"


@H7
def test_bare_numeric_citations_do_not_link_different_documents():
    """"[1]" in document A and "[1]" in document B are different references.
    Keying them globally as ``ref:1`` links every document that uses numeric
    citations to every other one, and lets "[1] [2] ... [99]" link to all."""
    a = chunk("a-0", "a", "Fusion powers stars [1].")
    b = chunk("b-0", "b", "Knead the dough well [1].")
    edges = CitationEdgeExtractor().extract([a, b])
    assert edges == [], f"unrelated documents linked by a shared local citation number: {edges}"


def test_same_doi_still_links_documents():
    """Control for the citation fix: a real shared identifier must still link."""
    a = chunk("a-0", "a", "See doi:10.1000/xyz123 for data.")
    b = chunk("b-0", "b", "As in doi:10.1000/xyz123 we find more.")
    assert len(CitationEdgeExtractor().extract([a, b])) == 1


# ---- hub concepts ------------------------------------------------------------

N_HUB = 40


def _hub_graph(monkeypatch, tmp_path, concepts_for: dict[int, list[str]]):
    chunks = [chunk(f"d{i}-0", f"d{i}", f"passage number {i} body") for i in range(N_HUB)]

    def reply(prompt: str) -> str:
        for i in range(N_HUB):
            if f"passage number {i} body" in prompt:
                return concepts_json(concepts_for.get(i, []))
        return concepts_json([])

    cfg = Config()
    cfg.indexing.concept_cache_dir = str(tmp_path / "concepts")
    entity = EntityEdgeExtractor(cfg.indexing, cfg.ner,
                                 llm=AnthropicLLM(client=ScriptedLLM(reply))).extract(chunks)
    return GraphBuilder(cfg).build(chunks, [], [], [], entity, [], [])


@H7
def test_hub_concept_does_not_make_a_clique(monkeypatch, tmp_path):
    """Every chunk of a domain corpus mentions "machine learning". That shared
    term carries no information, but today it creates C(40,2) = 780 edges at the
    full entity weight. Bound: no chunk gets more than N/4 entity neighbours."""
    concepts = {i: ["machine learning", f"topic {i}"] for i in range(N_HUB)}
    graph = _hub_graph(monkeypatch, tmp_path, concepts)
    max_deg = max(d for _, d in graph.degree())
    assert max_deg <= N_HUB // 4, (
        f"hub concept shared by all {N_HUB} chunks gave {graph.number_of_edges()} edges, "
        f"max degree {max_deg}"
    )


def test_specific_shared_concept_still_links(monkeypatch, tmp_path):
    """Control for the hub cap: a concept shared by two chunks must still link."""
    concepts = {i: [f"topic {i}"] for i in range(N_HUB)}
    concepts[3] = concepts[7] = ["pulsar timing"]
    graph = _hub_graph(monkeypatch, tmp_path, concepts)
    assert graph.has_edge("d3-0", "d7-0")
    assert graph.number_of_edges() == 1
