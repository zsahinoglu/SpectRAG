"""Pluggable component interfaces (ADR 0002).

Every external dependency SpectRAG talks to is reached through one of these
`typing.Protocol` classes. Implementations do not inherit from them; any object
with the right attributes and methods conforms (structural typing). All five
are `runtime_checkable` so `isinstance` can be used for a cheap shape check;
that check only verifies member names, not signatures.

Status: provisional until the v1.0 API freeze (ARCH3). ARCH2 wires embedders
and LLMs through these; H8 owns the `GraphStore` conformance suite.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np

    from spectrag.models import Chunk

__all__ = [
    "EmbedKind",
    "Embedder",
    "IndexLLM",
    "VectorStore",
    "GraphStore",
    "Reranker",
]

EmbedKind = Literal["document", "query"]


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors.

    `model_id` and `dimension` are written to the index manifest so an index is
    never queried with a different embedder than the one that built it.
    """

    model_id: str
    dimension: int

    def embed(self, texts: Sequence[str], kind: EmbedKind = "document") -> np.ndarray:
        """Return a float32 array of shape (len(texts), dimension), rows L2-normalized.

        `kind` lets asymmetric models apply their query/document instructions.
        An empty input returns shape (0, dimension).
        """
        ...


@runtime_checkable
class IndexLLM(Protocol):
    """Text-generation model used at index time (concepts, utility questions)
    and for answer generation. The default backend is Claude Haiku.

    Implementations own transport concerns: retries, rate limits, timeouts and
    API-key lookup. Caching of index-time outputs stays in the pipeline, keyed
    by `model_id` and the prompt version.
    """

    model_id: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return the model's text reply to a single-turn prompt."""
        ...

    def complete_json(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Any:
        """Return a reply parsed as JSON and validated against `schema` (JSON Schema).

        Raises `ValueError` if the reply cannot be parsed or does not validate;
        callers never receive unvalidated model output from this method.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Nearest-neighbour index over chunk vectors, addressed by chunk id.

    Scores are inner products of L2-normalized vectors (cosine similarity).
    Persistence is not part of this interface; the on-disk format belongs to
    `spectrag.formats` (H3).
    """

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Insert or replace vectors; `vectors` has shape (len(ids), dim)."""
        ...

    def remove(self, ids: Sequence[str]) -> None:
        """Delete vectors by id; unknown ids are ignored."""
        ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to `k` (chunk_id, score) pairs, best first, for one query vector."""
        ...

    def __len__(self) -> int: ...


@runtime_checkable
class GraphStore(Protocol):
    """Weighted chunk graph used for expansion and spectral scoring.

    Mirrors the surface already shared by `InMemoryGraphStore` and
    `Neo4jGraphClient`. Visualization helpers are app concerns and are
    deliberately not part of the contract. Stores are handed over ready to
    use: connecting is the factory's job, `close()` releases resources.
    """

    def write_graph(self, chunks: Sequence[Chunk], nx_graph: nx.Graph) -> None:
        """Replace the stored graph with `nx_graph` (nodes keyed by chunk_id)."""
        ...

    def bfs_expand(self, seed_chunk_ids: list[str], hops: int, max_candidates: int) -> set[str]:
        """Seeds plus nodes reachable within `hops`, capped at `max_candidates`."""
        ...

    def subgraph(self, chunk_ids: set[str]) -> nx.Graph:
        """Induced subgraph on `chunk_ids` with a `weight` attribute on every edge."""
        ...

    def node_count(self) -> int: ...

    def edge_count(self) -> int: ...

    def has_graph_data(self) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    """Optional second-stage scorer for (query, passage) pairs."""

    model_id: str

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage, in input order (higher is better)."""
        ...
