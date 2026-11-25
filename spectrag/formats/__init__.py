"""On-disk index format and manifests (H3). No `pickle` in any load path.

Layout of an index directory (format version 1)::

    manifest.json      format version + sha256/size of every other file
    chunks.parquet     one row per chunk; list/JSON columns, no Python objects
    embeddings.npy     (n, d) float32, row i belongs to chunks row i
    spectral.npy       (n, k) float32, same row alignment
    faiss.index        FAISS index (written by ``faiss.write_index``)

``has_embedding`` / ``has_spectral`` columns say which rows of the matrices are
real, so alignment never depends on counting (see CASE-08). The manifest is
written last, so a half-written index has no manifest and is refused.
Graphs use the same manifest mechanism (``nodes.parquet`` + ``edges.parquet``).

Everything here is data-only: Parquet, ``.npy`` read with ``allow_pickle=False``
and JSON. Legacy pickle files are never read here; see ``spectrag.formats.migrate``.
"""
from spectrag.formats.files import (
    FORMAT_VERSION,
    MANIFEST_FILE,
    IndexFormatError,
    read_chunks,
    read_graph,
    read_manifest,
    write_chunks,
    write_graph,
)

__all__ = [
    "FORMAT_VERSION", "MANIFEST_FILE", "IndexFormatError",
    "read_chunks", "read_graph", "read_manifest", "write_chunks", "write_graph",
]
