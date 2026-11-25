"""H3: round trips, integrity checks and the one-way legacy migration."""
from __future__ import annotations

import pickle

import faiss
import numpy as np
import pytest

from rag_system.models import Chunk
from rag_system.store.index_store import IndexStore
from rag_system.store.memory_graph import InMemoryGraphStore
from spectrag.formats import IndexFormatError, read_manifest
from spectrag.formats.migrate import main as migrate_main, migrate


def _chunks() -> list[Chunk]:
    a = Chunk("a", "d1", "alpha text", 0, ["1. Intro"], ["[1]"], utility_questions=["why?"],
              embedding=np.arange(4, dtype=np.float32), cluster_label=2,
              metadata={"page": 3, "src": "x.pdf"})
    b = Chunk("b", "d1", "beta text", 1, [], [], language="de")   # no embedding / coords
    c = Chunk("c", "d2", "gamma", 0, [], [], embedding=np.ones(4, dtype=np.float32),
              spectral_coords=np.array([0.5, -0.5], dtype=np.float32))
    return [a, b, c]


def _faiss() -> faiss.Index:
    idx = faiss.IndexFlatIP(4)
    idx.add(np.eye(4, dtype=np.float32))
    return idx


def test_index_round_trip_keeps_alignment_with_missing_vectors(tmp_path):
    IndexStore(tmp_path).save(_chunks(), _faiss())
    chunks, idx = IndexStore(tmp_path).load()
    a, b, c = chunks
    assert idx.ntotal == 4
    assert [x.chunk_id for x in chunks] == ["a", "b", "c"]
    np.testing.assert_array_equal(a.embedding, np.arange(4, dtype=np.float32))
    assert b.embedding is None and b.spectral_coords is None and b.language == "de"
    assert a.spectral_coords is None
    np.testing.assert_array_equal(c.spectral_coords, [0.5, -0.5])   # not shifted onto a or b
    assert (a.section_path, a.references, a.utility_questions) == (["1. Intro"], ["[1]"], ["why?"])
    assert a.cluster_label == 2 and a.metadata == {"page": 3, "src": "x.pdf"}


def test_manifest_has_version_and_checksums(tmp_path):
    IndexStore(tmp_path).save(_chunks(), _faiss())
    m = read_manifest(tmp_path)
    assert m["format_version"] == 1 and m["n_chunks"] == 3
    assert set(m["files"]) == {"chunks.parquet", "embeddings.npy", "spectral.npy", "faiss.index"}
    assert all(len(v["sha256"]) == 64 for v in m["files"].values())


@pytest.mark.parametrize("victim", ["chunks.parquet", "embeddings.npy", "faiss.index"])
def test_tampered_file_is_refused(tmp_path, victim):
    IndexStore(tmp_path).save(_chunks(), _faiss())
    with open(tmp_path / victim, "ab") as fh:
        fh.write(b"x")
    with pytest.raises(IndexFormatError, match="checksum"):
        IndexStore(tmp_path).load()


def test_unsupported_version_and_missing_manifest(tmp_path):
    IndexStore(tmp_path).save(_chunks(), _faiss())
    mpath = tmp_path / "manifest.json"
    mpath.write_text(mpath.read_text().replace('"format_version": 1', '"format_version": 99'))
    with pytest.raises(IndexFormatError, match="unsupported"):
        IndexStore(tmp_path).load()
    with pytest.raises(FileNotFoundError):
        IndexStore(tmp_path / "empty").load()


def test_manifest_cannot_point_outside_the_directory(tmp_path):
    IndexStore(tmp_path).save(_chunks(), _faiss())
    mpath = tmp_path / "manifest.json"
    mpath.write_text(mpath.read_text().replace("chunks.parquet", "../chunks.parquet"))
    with pytest.raises(IndexFormatError, match="unsafe"):
        IndexStore(tmp_path).load()


def test_graph_round_trip(tmp_path):
    import networkx as nx
    g = nx.Graph()
    g.add_edge("a", "b", weight=0.5, semantic=0.25, citation=1.0)
    g.add_node("c", doc_id="d2", position=0)
    g.nodes["a"].update(doc_id="d1", position=0)
    store = InMemoryGraphStore()
    store._graph = g
    store.save(tmp_path / "graph")
    other = InMemoryGraphStore()
    assert other.load(tmp_path / "graph") is True
    assert other.node_count() == 3 and other.edge_count() == 1
    row = other.get_edges_for_nodes({"a", "b"})[0]
    assert row["weight"] == 0.5 and row["semantic"] == 0.25 and row["citation"] == 1.0
    assert other._graph.nodes["a"]["doc_id"] == "d1"
    assert InMemoryGraphStore().load(tmp_path / "nothing") is False


def _write_legacy(d, chunks, graph=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunks.pkl").write_bytes(pickle.dumps(chunks, protocol=5))
    faiss.write_index(_faiss(), str(d / "faiss.index"))
    if graph is not None:
        (d / "graph.pkl").write_bytes(pickle.dumps(graph, protocol=5))


def test_migration_converts_legacy_index_and_graph(tmp_path):
    import networkx as nx
    g = nx.Graph()
    g.add_edge("a", "b", weight=0.5)
    g.nodes["a"]["doc_id"] = "d1"
    sdir = tmp_path / "store" / "sessions" / "s1"
    _write_legacy(sdir, _chunks(), g)
    with pytest.raises(IndexFormatError, match="migrate"):     # load refuses legacy files
        IndexStore(sdir).load()

    assert migrate(tmp_path / "store") == {str(sdir): ["chunks.pkl", "graph.pkl"]}
    chunks, _ = IndexStore(sdir).load()
    assert [c.chunk_id for c in chunks] == ["a", "b", "c"]
    np.testing.assert_array_equal(chunks[2].spectral_coords, [0.5, -0.5])
    loaded = InMemoryGraphStore()
    assert loaded.load(sdir / "graph") and loaded.edge_count() == 1
    assert migrate(tmp_path / "store") == {}                    # idempotent


def test_migration_refuses_forbidden_globals_and_runs_no_payload(tmp_path):
    sentinel = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (open, (str(sentinel), "w"))

    _write_legacy(tmp_path / "idx", [Evil()])
    assert migrate_main([str(tmp_path / "idx")]) == 1
    assert not sentinel.exists()
    assert not (tmp_path / "idx" / "manifest.json").exists()


def test_embedding_cache_round_trip_and_legacy_ignored(tmp_path):
    from rag_system.indexing.embedder import _EmbeddingCache
    (tmp_path / "m.pkl").write_bytes(b"not loaded")
    c = _EmbeddingCache(str(tmp_path), "m")
    assert "k" not in c
    c.put("k", np.arange(3))
    c.flush()
    again = _EmbeddingCache(str(tmp_path), "m")
    np.testing.assert_array_equal(again.get("k"), np.arange(3, dtype=np.float32))
