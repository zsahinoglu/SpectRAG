from __future__ import annotations
from typing import TYPE_CHECKING

from spectrag.models import Chunk
from config import Config

if TYPE_CHECKING:
    from rag_system.store.neo4j_client import Neo4jGraphClient


class GraphExpander:
    """
    Stage 2 — BFS from seeds to collect multi-hop neighbors.
    Delegates traversal to Neo4j via Cypher so the full graph never needs to
    be loaded into Python memory.
    """

    def __init__(self, cfg: Config) -> None:
        self.hops = cfg.retrieval.graph_expansion_hops
        self.max_candidates = cfg.retrieval.max_candidates

    def expand(
        self,
        seed_indices: list[int],
        chunks: list[Chunk],
        neo4j_client: Neo4jGraphClient,
    ) -> list[int]:
        chunk_id_to_idx = self._id_map(chunks)
        seed_chunk_ids = [chunks[i].chunk_id for i in seed_indices]

        expanded_ids = neo4j_client.bfs_expand(
            seed_chunk_ids, self.hops, self.max_candidates
        )

        pairs = [(cid, chunk_id_to_idx[cid]) for cid in expanded_ids if cid in chunk_id_to_idx]
        if any(chunks[i].chunk_id != cid for cid, i in pairs):
            # Cached map is stale (corpus changed under the same size/boundaries):
            # rebuild it once and redo the lookup.
            self._map_key = None
            chunk_id_to_idx = self._id_map(chunks)
            pairs = [(cid, chunk_id_to_idx[cid]) for cid in expanded_ids if cid in chunk_id_to_idx]
        return [i for _cid, i in pairs]

    def _id_map(self, chunks: list[Chunk]) -> dict[str, int]:
        """chunk_id → index, built once per corpus instead of once per query."""
        key = (len(chunks), chunks[0].chunk_id, chunks[-1].chunk_id) if chunks else (0,)
        if getattr(self, "_map_key", None) != key:
            self._map = {c.chunk_id: i for i, c in enumerate(chunks)}
            self._map_key = key
        return self._map
