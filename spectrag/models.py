"""Core data types shared by indexing, retrieval and stores.

Canonical home of `Chunk` and `QueryResult` since ARCH1 (moved from
`rag_system.models`, which remains as a re-export shim; see ADR 0003).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["Chunk", "ChunkExplanation", "QueryResult"]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int                   # 0-indexed position within document
    section_path: list[str]         # e.g. ["1. Introduction", "1.2 Background"]
    references: list[str]           # raw citation strings found in text
    language: str = "en"                 # ISO 639-1 code detected at index time
    embedding: Optional[np.ndarray] = None
    spectral_coords: Optional[np.ndarray] = None
    utility_questions: list[str] = field(default_factory=list)  # LLM "what does this answer?" (Stage 1 query-side)
    cluster_label: int = -1              # assigned offline by SpectralClusterAssigner
    metadata: dict = field(default_factory=dict)  # source path, page number, etc.

    def __hash__(self) -> int:
        return hash(self.chunk_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.chunk_id == other.chunk_id


@dataclass(frozen=True)
class ChunkExplanation:
    """Why one chunk was selected (provisional names until ARCH3 freezes the API).

    seed_source: retrievers that put the chunk in the seed set, from
        'dense', 'utility_question', 'bm25'; ('graph_expansion',) if it entered
        only through graph expansion.
    seed_ranks:  source -> 1-based rank in that source's ranking.
    score_parts: relevance (hybrid dense/BM25 relevance), spectral_proximity and
        ppr_proximity (absent when not computed), diffused (the combined score),
        final (the score reported in QueryResult.scores).
    """
    chunk_id: str
    seed_source: tuple[str, ...]
    seed_ranks: dict[str, int]
    score_parts: dict[str, float]


@dataclass
class QueryResult:
    chunks: list[Chunk]
    answer: str
    scores: dict[str, float]        # chunk_id → final relevance score
    explanations: list[ChunkExplanation] = field(
        default_factory=list, repr=False, compare=False)

    def explain(self) -> list[ChunkExplanation]:
        """One ChunkExplanation per selected chunk, in selection order."""
        return list(self.explanations)
