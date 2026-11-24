"""ARCH1: pins the five component Protocols (ADR 0002).

Member names and parameter lists are asserted exactly; changing them is an
API change and needs an ADR. Minimal fakes check that structural conformance
works and that incomplete implementations are rejected.
"""
from __future__ import annotations

import inspect
from typing import Protocol

import numpy as np
import pytest

from spectrag import protocols
from spectrag.protocols import Embedder, GraphStore, IndexLLM, Reranker, VectorStore

# protocol -> (data attributes, {method: parameter names excluding self})
EXPECTED = {
    Embedder: ({"model_id", "dimension"}, {
        "embed": ["texts", "kind"],
    }),
    IndexLLM: ({"model_id"}, {
        "complete": ["prompt", "system", "max_tokens", "temperature"],
        "complete_json": ["prompt", "schema", "system", "max_tokens", "temperature"],
    }),
    VectorStore: (set(), {
        "add": ["ids", "vectors"],
        "remove": ["ids"],
        "search": ["query", "k"],
        "__len__": [],
    }),
    GraphStore: (set(), {
        "write_graph": ["chunks", "nx_graph"],
        "bfs_expand": ["seed_chunk_ids", "hops", "max_candidates"],
        "subgraph": ["chunk_ids"],
        "node_count": [],
        "edge_count": [],
        "has_graph_data": [],
        "close": [],
    }),
    Reranker: ({"model_id"}, {
        "rerank": ["query", "passages"],
    }),
}


def test_protocols_all_is_pinned():
    assert sorted(protocols.__all__) == sorted(
        ["EmbedKind", "Embedder", "IndexLLM", "VectorStore", "GraphStore", "Reranker"]
    )


@pytest.mark.parametrize("proto", list(EXPECTED), ids=lambda p: p.__name__)
def test_protocol_members_are_pinned(proto):
    assert Protocol in proto.__mro__
    assert getattr(proto, "_is_runtime_protocol", False), "must be runtime_checkable"
    attrs, methods = EXPECTED[proto]
    assert set(getattr(proto, "__annotations__", {})) == attrs
    declared = {
        name for name, val in vars(proto).items()
        if inspect.isfunction(val) and not name.startswith("_") or name == "__len__"
    }
    assert declared == set(methods)
    for name, params in methods.items():
        sig = inspect.signature(getattr(proto, name))
        assert list(sig.parameters)[1:] == params, name


def test_llm_options_are_keyword_only():
    for name in ("complete", "complete_json"):
        sig = inspect.signature(getattr(IndexLLM, name))
        for p in ("system", "max_tokens", "temperature"):
            assert sig.parameters[p].kind is inspect.Parameter.KEYWORD_ONLY


# Minimal structural fakes: no inheritance from the Protocols.

class FakeEmbedder:
    model_id = "fake-embed-v1"
    dimension = 4

    def embed(self, texts, kind="document"):
        out = np.ones((len(texts), self.dimension), dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class FakeLLM:
    model_id = "fake-llm-v1"

    def complete(self, prompt, *, system=None, max_tokens=1024, temperature=0.0):
        return "ok"

    def complete_json(self, prompt, schema, *, system=None, max_tokens=1024, temperature=0.0):
        return {}


class FakeVectorStore:
    def __init__(self):
        self._v = {}

    def add(self, ids, vectors):
        self._v.update(zip(ids, vectors))

    def remove(self, ids):
        for i in ids:
            self._v.pop(i, None)

    def search(self, query, k):
        scored = sorted(((i, float(v @ query)) for i, v in self._v.items()),
                        key=lambda t: -t[1])
        return scored[:k]

    def __len__(self):
        return len(self._v)


class FakeReranker:
    model_id = "fake-rerank-v1"

    def rerank(self, query, passages):
        return [0.0 for _ in passages]


@pytest.mark.parametrize("fake, proto", [
    (FakeEmbedder(), Embedder),
    (FakeLLM(), IndexLLM),
    (FakeVectorStore(), VectorStore),
    (FakeReranker(), Reranker),
], ids=lambda x: type(x).__name__ if not isinstance(x, type) else x.__name__)
def test_fakes_conform(fake, proto):
    assert isinstance(fake, proto)


def test_incomplete_implementation_rejected():
    class NoDim:
        model_id = "x"

        def embed(self, texts, kind="document"):
            return np.zeros((0, 0))

    class NoRemove:
        def add(self, ids, vectors): ...
        def search(self, query, k): ...
        def __len__(self): return 0

    assert not isinstance(NoDim(), Embedder)
    assert not isinstance(NoRemove(), VectorStore)
    assert not isinstance(FakeEmbedder(), IndexLLM)


def test_fake_vector_store_roundtrip():
    vs = FakeVectorStore()
    emb = FakeEmbedder()
    vs.add(["a", "b"], emb.embed(["x", "y"]))
    assert len(vs) == 2
    vs.remove(["a", "missing"])
    assert [i for i, _ in vs.search(emb.embed(["q"], kind="query")[0], 5)] == ["b"]


def test_existing_in_memory_graph_store_conforms():
    """The current in-memory store already satisfies the GraphStore contract."""
    pytest.importorskip("networkx")
    from rag_system.store.memory_graph import InMemoryGraphStore

    assert isinstance(InMemoryGraphStore(), GraphStore)
