from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import faiss

from spectrag.models import Chunk, ChunkExplanation, QueryResult
from spectrag.protocols import Embedder, IndexLLM
from rag_system.components import build_embedder
from rag_system.query.seed_retrieval import SeedRetriever
from rag_system.query.graph_expansion import GraphExpander
from rag_system.query.diffusion import RelevanceDiffuser
from rag_system.query.cluster_selection import ClusterAwareSelector
from rag_system.generation.generator import Generator
from config import Config

if TYPE_CHECKING:
    from rag_system.store.neo4j_client import Neo4jGraphClient

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Chains the five runtime stages:
      Stage 1  SeedRetriever         — FAISS ANN top-K
      Stage 2  GraphExpander         — Neo4j Cypher BFS
      Stage 3  RelevanceDiffuser     — spectral proximity re-scoring (eigenspace distance to seeds)
      Stage 4  ClusterAwareSelector  — greedy diverse selection
      Stage 5  Generator             — LLM answer

    `embedder` must be the embedder the index was built with; `llm` answers the
    question (`NoLLM` → retrieval only, empty answer). Both default to `cfg`.
    """

    def __init__(self, cfg: Config, *, embedder: Embedder | None = None,
                 llm: IndexLLM | None = None) -> None:
        self.embedder = embedder if embedder is not None else build_embedder(cfg)
        self.seed_retriever = SeedRetriever(cfg, self.embedder)
        self.expander = GraphExpander(cfg)
        self.diffuser = RelevanceDiffuser(cfg)
        self.selector = ClusterAwareSelector(cfg)
        self.generator = Generator(cfg, llm)

    def query(
        self,
        question: str,
        chunks: list[Chunk],
        faiss_index: faiss.Index,
        neo4j_client: Neo4jGraphClient,
    ) -> QueryResult:
        logger.info("Stage 1 — seed retrieval")
        seed_trace: dict = {}
        seed_indices, all_scores = self.seed_retriever.retrieve(
            question, chunks, faiss_index, trace=seed_trace
        )
        logger.info("  → %d seeds", len(seed_indices))

        logger.info("Stage 2 — graph expansion (Neo4j BFS)")
        candidate_indices = self.expander.expand(seed_indices, chunks, neo4j_client)
        logger.info("  → %d candidates", len(candidate_indices))

        # Fetch the candidate subgraph from Neo4j once; stages 3+4 use it as nx.Graph
        candidate_chunk_ids = {chunks[i].chunk_id for i in candidate_indices}
        subgraph = neo4j_client.subgraph(candidate_chunk_ids)

        logger.info("Stage 3 — proximity re-scoring (spectral / PPR)")
        diffuse_trace: dict = {}
        diffused_scores = self.diffuser.diffuse(
            candidate_indices, seed_indices, all_scores, chunks, subgraph,
            trace=diffuse_trace,
        )

        logger.info("Stage 4 — cluster-aware selection")
        selected_indices = self.selector.select(
            candidate_indices, diffused_scores, all_scores, chunks, subgraph, question
        )
        logger.info("  → %d context chunks selected", len(selected_indices))

        logger.info("Stage 5 — generation")
        selected_chunks = [chunks[i] for i in selected_indices]
        answer = self.generator.generate(question, selected_chunks)

        scores = {
            chunks[i].chunk_id: diffused_scores.get(i, 0.0)
            for i in selected_indices
        }
        explanations = self._explain(
            selected_indices, seed_indices, all_scores, diffused_scores,
            seed_trace, diffuse_trace, chunks,
        )
        return QueryResult(chunks=selected_chunks, answer=answer, scores=scores,
                           explanations=explanations)

    @staticmethod
    def _explain(selected, seed_indices, all_scores, diffused, seed_trace,
                 diffuse_trace, chunks) -> list[ChunkExplanation]:
        """Read-only provenance for QueryResult.explain(); touches no ranking."""
        seeds = set(seed_indices)
        rankings = seed_trace.get("rankings", {})
        out = []
        for i in selected:
            ranks = {src: lst.index(i) + 1 for src, lst in rankings.items()
                     if i in lst} if i in seeds else {}
            parts = {"relevance": float(all_scores[i])}
            for key, name in (("spectral", "spectral_proximity"),
                              ("ppr", "ppr_proximity")):
                prox = diffuse_trace.get(key)
                if prox is not None and i in prox:
                    parts[name] = float(prox[i])
            parts["diffused"] = float(diffused.get(i, 0.0))
            parts["final"] = parts["diffused"]
            out.append(ChunkExplanation(
                chunk_id=chunks[i].chunk_id,
                seed_source=tuple(ranks) or ("graph_expansion",),
                seed_ranks=ranks, score_parts=parts))
        return out

    def close(self) -> None:
        self.generator.close()
