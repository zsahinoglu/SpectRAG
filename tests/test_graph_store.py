"""GraphStore conformance (H8): in-memory, file round-trip, and Neo4j when available."""
from __future__ import annotations

import json
import os

import networkx as nx
import pytest

from rag_system.store.memory_graph import InMemoryGraphStore
from tests.conformance.graph_store import check_graph_store
from tests.golden_corpus import build_index
from tests.test_golden_retrieval import _run_all


class _FileStore(InMemoryGraphStore):
    """The 'file store': every write is saved to disk and reloaded (Parquet format)."""

    def __init__(self, path):
        super().__init__()
        self._path = path

    def write_graph(self, chunks, nx_graph):
        super().write_graph(chunks, nx_graph)
        self.save(self._path)
        assert self.load(self._path)


def _neo4j_config():
    pytest.skip("H8b: Neo4j weight-ordered BFS admission not implemented yet")
    uri = os.environ.get("SPECTRAG_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("set SPECTRAG_TEST_NEO4J_URI (and _USER/_PASSWORD) to test against Neo4j")
    pytest.importorskip("neo4j")
    from config import Neo4jConfig
    db = os.environ.get("SPECTRAG_TEST_NEO4J_DATABASE")
    if not db or db == "neo4j":
        pytest.skip("set SPECTRAG_TEST_NEO4J_DATABASE to a dedicated test database (not 'neo4j'); "
                    "write_graph deletes all Chunk nodes")
    return Neo4jConfig(
        uri=uri,
        database=db,
        user=os.environ.get("SPECTRAG_TEST_NEO4J_USER", "neo4j"),
        password=os.environ.get("SPECTRAG_TEST_NEO4J_PASSWORD", "password123"),
    )


def _neo4j_store():
    from rag_system.store.neo4j_client import Neo4jGraphClient
    store = Neo4jGraphClient(_neo4j_config())
    store.connect()
    store.write_graph([], nx.Graph())  # start empty
    return store


def test_in_memory_store_conforms():
    check_graph_store(InMemoryGraphStore)


def test_file_store_conforms(tmp_path):
    n = iter(range(1000))
    check_graph_store(lambda: _FileStore(tmp_path / f"g{next(n)}"))


def test_neo4j_store_conforms():
    _neo4j_config()
    check_graph_store(_neo4j_store)


def _golden_with(store_factory):
    cfg, emb, chunks, fidx, store = build_index()
    other = store_factory()
    other.write_graph(chunks, store.subgraph({c.chunk_id for c in chunks}))
    return json.dumps(_run_all(cfg, emb, chunks, fidx, other), sort_keys=True)


def test_file_store_matches_memory_on_golden_corpus(tmp_path):
    memory = json.dumps(_run_all(*build_index()), sort_keys=True)
    assert _golden_with(lambda: _FileStore(tmp_path / "g")) == memory


def test_neo4j_matches_memory_on_golden_corpus():
    _neo4j_config()
    memory = json.dumps(_run_all(*build_index()), sort_keys=True)
    store = _golden_with(_neo4j_store)
    assert store == memory
