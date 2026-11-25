"""Reading and writing the safe on-disk format (see ``spectrag.formats``)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spectrag.models import Chunk

FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
CHUNKS_FILE = "chunks.parquet"
EMBEDDINGS_FILE = "embeddings.npy"
SPECTRAL_FILE = "spectral.npy"
FAISS_FILE = "faiss.index"
NODES_FILE = "nodes.parquet"
EDGES_FILE = "edges.parquet"

_EDGE_SIGNALS = ("weight", "semantic", "adjacency", "section", "entity",
                 "utility_question", "citation")


class IndexFormatError(ValueError):
    """The directory is not a valid index: missing/unsupported manifest, checksum mismatch."""


# ---- manifest -------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(directory: Path | str, files: list[str], **meta: Any) -> dict:
    """Checksum ``files`` (names inside ``directory``) and write ``manifest.json`` last."""
    d = Path(directory)
    manifest = {
        "format_version": FORMAT_VERSION,
        **meta,
        "files": {n: {"sha256": _sha256(d / n), "bytes": (d / n).stat().st_size}
                  for n in sorted(files)},
    }
    tmp = d / (MANIFEST_FILE + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, d / MANIFEST_FILE)
    return manifest


def read_manifest(directory: Path | str, *, verify: bool = True) -> dict:
    """Load the manifest; with ``verify`` every listed file must match its sha256."""
    d = Path(directory)
    mpath = d / MANIFEST_FILE
    if not mpath.is_file():
        legacy = " (legacy pickle index? run `python -m spectrag.formats.migrate`)" \
            if (d / "chunks.pkl").exists() or (d / "graph.pkl").exists() else ""
        raise IndexFormatError(f"no {MANIFEST_FILE} in {d}{legacy}")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        version, files = manifest["format_version"], manifest["files"]
    except (ValueError, KeyError, TypeError) as exc:
        raise IndexFormatError(f"unreadable manifest {mpath}: {exc}") from exc
    if version != FORMAT_VERSION:
        raise IndexFormatError(f"unsupported index format_version {version!r} "
                               f"(this build reads {FORMAT_VERSION})")
    if verify:
        for name, info in files.items():
            if Path(name).name != name:          # no path components, ever
                raise IndexFormatError(f"manifest lists unsafe file name {name!r}")
            f = d / name
            if not f.is_file():
                raise IndexFormatError(f"{name} listed in manifest but missing in {d}")
            if _sha256(f) != info.get("sha256"):
                raise IndexFormatError(f"checksum mismatch for {name} in {d}")
    return manifest


# ---- chunks ---------------------------------------------------------------------

def _json_default(o: Any):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"metadata value of type {type(o).__name__} is not JSON-serializable")


def _matrix(chunks: list[Chunk], attr: str) -> tuple[np.ndarray | None, list[bool]]:
    present = [getattr(c, attr) is not None for c in chunks]
    if not any(present):
        return None, present
    rows = [np.asarray(getattr(c, attr), dtype=np.float32).ravel()
            for c in chunks if getattr(c, attr) is not None]
    dim = rows[0].shape[0]
    if any(r.shape[0] != dim for r in rows):
        raise ValueError(f"chunks have {attr} of differing lengths")
    mat = np.zeros((len(chunks), dim), dtype=np.float32)
    mat[np.flatnonzero(present)] = np.stack(rows)
    return mat, present


def write_chunks(directory: Path | str, chunks: list[Chunk], faiss_file: str | None = None,
                 **meta: Any) -> dict:
    """Write chunks (+ embeddings/spectral coords) and a manifest covering ``faiss_file``
    (a file already present in ``directory``). Extra ``meta`` goes into the manifest."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    files = [CHUNKS_FILE]
    emb, has_emb = _matrix(chunks, "embedding")
    spec, has_spec = _matrix(chunks, "spectral_coords")
    table = pa.table({
        "chunk_id": pa.array([c.chunk_id for c in chunks], pa.string()),
        "doc_id": pa.array([c.doc_id for c in chunks], pa.string()),
        "text": pa.array([c.text for c in chunks], pa.string()),
        "position": pa.array([c.position for c in chunks], pa.int64()),
        "section_path": pa.array([list(c.section_path) for c in chunks], pa.list_(pa.string())),
        "references": pa.array([list(c.references) for c in chunks], pa.list_(pa.string())),
        "language": pa.array([c.language for c in chunks], pa.string()),
        "utility_questions": pa.array([list(c.utility_questions) for c in chunks],
                                      pa.list_(pa.string())),
        "cluster_label": pa.array([int(c.cluster_label) for c in chunks], pa.int64()),
        "metadata": pa.array([json.dumps(c.metadata, default=_json_default) for c in chunks],
                             pa.string()),
        "has_embedding": pa.array(has_emb, pa.bool_()),
        "has_spectral": pa.array(has_spec, pa.bool_()),
    })
    pq.write_table(table, d / CHUNKS_FILE)
    for name, mat in ((EMBEDDINGS_FILE, emb), (SPECTRAL_FILE, spec)):
        if mat is not None:
            np.save(d / name, mat, allow_pickle=False)
            files.append(name)
        elif (d / name).exists():
            (d / name).unlink()                  # stale file from an earlier save
    if faiss_file:
        files.append(faiss_file)
    return write_manifest(d, files, n_chunks=len(chunks), **meta)


def read_chunks(directory: Path | str) -> list[Chunk]:
    d = Path(directory)
    manifest = read_manifest(d)
    if CHUNKS_FILE not in manifest["files"]:
        raise IndexFormatError(f"{CHUNKS_FILE} not listed in manifest of {d}")
    cols = pq.read_table(d / CHUNKS_FILE).to_pydict()
    n = len(cols["chunk_id"])
    mats = {}
    for key, name in (("has_embedding", EMBEDDINGS_FILE), ("has_spectral", SPECTRAL_FILE)):
        if any(cols[key]):
            if name not in manifest["files"]:
                raise IndexFormatError(f"{name} not listed in manifest of {d}")
            m = np.load(d / name, allow_pickle=False)
            if m.ndim != 2 or m.shape[0] != n:
                raise IndexFormatError(f"{name} shape {m.shape} does not match {n} chunks")
            mats[key] = m
    chunks = []
    for i in range(n):
        chunks.append(Chunk(
            chunk_id=cols["chunk_id"][i], doc_id=cols["doc_id"][i], text=cols["text"][i],
            position=cols["position"][i], section_path=list(cols["section_path"][i]),
            references=list(cols["references"][i]), language=cols["language"][i],
            embedding=mats["has_embedding"][i].copy() if cols["has_embedding"][i] else None,
            spectral_coords=mats["has_spectral"][i].copy() if cols["has_spectral"][i] else None,
            utility_questions=list(cols["utility_questions"][i]),
            cluster_label=cols["cluster_label"][i],
            metadata=json.loads(cols["metadata"][i]),
        ))
    return chunks


# ---- graph ----------------------------------------------------------------------

def write_graph(directory: Path | str, graph: nx.Graph) -> dict:
    """Persist a chunk graph: nodes (id, doc_id, position) and edges (known signals)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / MANIFEST_FILE).unlink(missing_ok=True)   # a crash mid-write must read as "no graph"
    nodes = list(graph.nodes(data=True))
    pq.write_table(pa.table({
        "id": pa.array([str(n) for n, _ in nodes], pa.string()),
        "doc_id": pa.array([a.get("doc_id") for _, a in nodes], pa.string()),
        "position": pa.array([a.get("position") for _, a in nodes], pa.int64()),
    }), d / NODES_FILE)
    edges = list(graph.edges(data=True))
    cols: dict[str, Any] = {
        "source": pa.array([str(u) for u, _, _ in edges], pa.string()),
        "target": pa.array([str(v) for _, v, _ in edges], pa.string()),
    }
    for s in _EDGE_SIGNALS:
        cols[s] = pa.array([float(a.get(s, 0.0)) for _, _, a in edges], pa.float64())
    pq.write_table(pa.table(cols), d / EDGES_FILE)
    return write_manifest(d, [NODES_FILE, EDGES_FILE],
                          n_nodes=len(nodes), n_edges=len(edges))


def read_graph(directory: Path | str) -> nx.Graph:
    d = Path(directory)
    manifest = read_manifest(d)
    for name in (NODES_FILE, EDGES_FILE):
        if name not in manifest["files"]:
            raise IndexFormatError(f"{name} not listed in manifest of {d}")
    nodes = pq.read_table(d / NODES_FILE).to_pydict()
    edges = pq.read_table(d / EDGES_FILE).to_pydict()
    g = nx.Graph()
    for i, nid in enumerate(nodes["id"]):
        g.add_node(nid, doc_id=nodes["doc_id"][i], position=nodes["position"][i])
    for i, (u, v) in enumerate(zip(edges["source"], edges["target"])):
        g.add_edge(u, v, **{s: edges[s][i] for s in _EDGE_SIGNALS})
    return g
