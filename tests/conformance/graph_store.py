"""Behavioral conformance suite for `spectrag.GraphStore` (ADR 0002, ADR 0007).

`check_graph_store(factory)` raises `AssertionError` on the first contract
violation. `factory()` must return a fresh, empty, ready-to-use store; it is
called several times. Run it on every implementation (Neo4j only when a server
is available), so the pipelines can treat stores as interchangeable.
"""
from __future__ import annotations

import networkx as nx

from spectrag.models import Chunk
from spectrag.protocols import GraphStore


def _chunks(ids: list[str]) -> list[Chunk]:
    return [
        Chunk(chunk_id=i, doc_id=i.split("-")[0], text=i, position=n,
              section_path=[], references=[], utility_questions=[])
        for n, i in enumerate(ids)
    ]


def sample_graph() -> tuple[list[Chunk], nx.Graph]:
    """Path a-0 .. a-4, a hub h-0 joined to a-0..a-3, and an isolated node z-0."""
    chunks = _chunks(["a-0", "a-1", "a-2", "a-3", "a-4", "h-0", "z-0"])
    g = nx.Graph()
    g.add_nodes_from(c.chunk_id for c in chunks)
    for n in range(4):
        g.add_edge(f"a-{n}", f"a-{n + 1}", weight=0.5 + n / 10, semantic=0.5)
    for n in range(4):
        g.add_edge("h-0", f"a-{n}", weight=0.25, adjacency=1.0)
    return chunks, g


def check_graph_store(factory) -> None:
    store = factory()
    assert isinstance(store, GraphStore), "missing GraphStore members"

    # empty store
    assert store.node_count() == 0 and store.edge_count() == 0
    assert store.has_graph_data() is False
    assert store.bfs_expand(["a-0"], 2, 10) == {"a-0"}
    assert set(store.subgraph(set()).nodes) == set()

    chunks, g = sample_graph()
    store.write_graph(chunks, g)
    assert store.node_count() == 7
    assert store.edge_count() == g.number_of_edges() == 8
    assert store.has_graph_data() is True

    # write_graph replaces, never appends; a rewrite is idempotent
    store.write_graph(chunks, g)
    assert (store.node_count(), store.edge_count()) == (7, 8)
    c2 = _chunks(["x-0", "x-1"])
    g2 = nx.Graph()
    g2.add_edge("x-0", "x-1", weight=0.9)
    store.write_graph(c2, g2)
    assert (store.node_count(), store.edge_count()) == (2, 1)
    assert store.bfs_expand(["a-0"], 2, 10) == {"a-0"}, "old graph must be gone"
    store.write_graph(chunks, g)

    # bfs_expand: seeds always included, hop-limited, undirected
    assert store.bfs_expand(["a-0"], 0, 10) == {"a-0"}
    assert store.bfs_expand(["a-0"], 1, 10) == {"a-0", "a-1", "h-0"}
    assert store.bfs_expand(["a-4"], 1, 10) == {"a-4", "a-3"}, "edges are followed both ways"
    assert store.bfs_expand(["a-0"], 2, 10) == {"a-0", "a-1", "a-2", "a-3", "h-0"}
    assert store.bfs_expand(["z-0"], 3, 10) == {"z-0"}
    assert store.bfs_expand(["a-4", "z-0"], 1, 10) == {"a-4", "a-3", "z-0"}
    assert store.bfs_expand(["ghost"], 2, 10) == {"ghost"}, "unknown seeds are kept, not an error"
    assert store.bfs_expand([], 2, 10) == set()
    assert store.bfs_expand(["a-0"], 9, 100) == {c.chunk_id for c in chunks if c.chunk_id != "z-0"}

    # cap: within a hop, new nodes are admitted in ascending chunk_id order
    assert store.bfs_expand(["a-0"], 1, 2) == {"a-0", "a-1"}
    assert store.bfs_expand(["a-0"], 2, 4) == {"a-0", "a-1", "h-0", "a-2"}
    assert store.bfs_expand(["a-0"], 1, 1) == {"a-0"}
    # seeds are never dropped, even when they exceed the cap
    assert store.bfs_expand(["a-0", "a-4", "z-0"], 2, 2) == {"a-0", "a-4", "z-0"}

    # subgraph: induced, weighted, requested nodes always present
    sub = store.subgraph({"a-0", "a-1", "h-0", "z-0", "ghost"})
    assert isinstance(sub, nx.Graph) and not sub.is_directed()
    assert set(sub.nodes) == {"a-0", "a-1", "h-0", "z-0", "ghost"}
    assert {frozenset(e) for e in sub.edges} == {
        frozenset(("a-0", "a-1")), frozenset(("h-0", "a-0")), frozenset(("h-0", "a-1")),
    }
    assert abs(sub["a-0"]["a-1"]["weight"] - 0.5) < 1e-9
    assert abs(sub["h-0"]["a-0"]["weight"] - 0.25) < 1e-9
    assert all("weight" in d for _, _, d in sub.edges(data=True))
    full = store.subgraph({c.chunk_id for c in chunks})
    assert full.number_of_edges() == 8
    sub.add_edge("a-0", "z-0", weight=1.0)
    assert store.subgraph({"a-0", "z-0"}).number_of_edges() == 0, "result is a copy"

    # edges to chunks that were not written are ignored, not turned into nodes
    g3 = g.copy()
    g3.add_edge("a-0", "outsider", weight=0.7)
    store.write_graph(chunks, g3)
    assert (store.node_count(), store.edge_count()) == (7, 8)
    assert "outsider" not in store.bfs_expand(["a-0"], 3, 100)

    # duplicate chunk ids collapse to one node
    store.write_graph(chunks + _chunks(["a-0"]), g)
    assert store.node_count() == 7

    store.write_graph([], nx.Graph())
    assert store.has_graph_data() is False

    store.close()
    store.close()  # idempotent
