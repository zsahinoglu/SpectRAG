"""SpectRAG: spectral graph retrieval-augmented generation.

Package layout (ADR 0001):

    spectrag.models      core data types (Chunk, QueryResult)
    spectrag.settings    configuration dataclasses, later presets
    spectrag.protocols   pluggable interfaces (ADR 0002)
    spectrag.ingest      document loading and chunking
    spectrag.index       graph construction, edge signals, spectral embedding
    spectrag.retrieve    seeding, expansion, diffusion, selection
    spectrag.stores      VectorStore / GraphStore implementations
    spectrag.embedders   Embedder implementations
    spectrag.llms        IndexLLM implementations
    spectrag.rerankers   Reranker implementations
    spectrag.formats     on-disk index format and manifests

Importing `spectrag` is cheap: it does not import optional backends or
`spectrag.settings`.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

from spectrag.models import Chunk, QueryResult
from spectrag.protocols import Embedder, GraphStore, IndexLLM, Reranker, VectorStore

try:
    __version__ = _version("spectrag")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "Chunk",
    "QueryResult",
    "Embedder",
    "IndexLLM",
    "VectorStore",
    "GraphStore",
    "Reranker",
]
