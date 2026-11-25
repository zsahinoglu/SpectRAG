"""H7 review fixes: duplicate NER terms, unvalidated caches."""
from __future__ import annotations

import hashlib
import json

from config import IndexingConfig
from rag_system.indexing.edges.entity import EntityEdgeExtractor
from rag_system.indexing.edges.llm_concept import LLMConceptExtractor, cache_name
from rag_system.untrusted import MAX_CONCEPTS
from spectrag.models import Chunk


def _chunk(i):
    return Chunk(chunk_id=f"c{i}", doc_id="d", text=f"t{i}", position=i,
                 section_path=[], references=[])


def _entity(terms_by_text):
    ex = object.__new__(EntityEdgeExtractor)
    ex._min_freq, ex._log_interval, ex._skip, ex._concept = 2, 1000, False, None
    ex._terms = lambda chunk: terms_by_text[chunk.text]
    return ex


def test_duplicate_terms_do_not_crash_or_inflate_hubs():
    # 'x' is listed twice per chunk (two NER labels) in 3 of 20 chunks: not a hub
    # (max_df=5); the old code counted 6 mentions and dropped it.
    terms = {f"t{i}": [f"u{i}"] for i in range(20)}
    for i in range(3):
        terms[f"t{i}"] = ["x", "x"]
    edges = _entity(terms).extract([_chunk(i) for i in range(20)])
    assert {(e.i, e.j) for e in edges} == {(0, 1), (0, 2), (1, 2)}
    assert all(0 < e.score <= 1 for e in edges)


def test_two_chunks_with_only_a_repeated_term():
    edges = _entity({"t0": ["x", "x"], "t1": ["x", "x"]}).extract([_chunk(0), _chunk(1)])
    assert [(e.i, e.j) for e in edges] == [(0, 1)]
    assert 0 < edges[0].score <= 1


def test_concept_cache_hit_is_revalidated(tmp_path):
    class L:
        model_id = "m"
    cfg = IndexingConfig(concept_cache_dir=str(tmp_path))
    snippet = "poisoned passage"
    key = hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:16]
    poisoned = [f"{'x' * 100}{i}" for i in range(25)] + [f"ok{i}" for i in range(50)]
    (tmp_path / f"{cache_name('m')}.json").write_text(
        json.dumps({key: poisoned}), encoding="utf-8")
    got = LLMConceptExtractor(L(), cfg).extract(snippet)
    assert 0 < len(got) <= MAX_CONCEPTS and all(len(c) <= 64 for c in got)
