from __future__ import annotations
import logging
from collections import defaultdict
from spectrag.models import Chunk
from spectrag.llms import NoLLM
from spectrag.protocols import IndexLLM
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge
from rag_system.ner.extractor import HybridNERExtractor
from config import IndexingConfig, LanguageConfig, NERConfig, OllamaConfig

logger = logging.getLogger(__name__)

#: A term in more than max(HUB_MIN_CHUNKS, HUB_MAX_FRACTION * n_chunks) chunks makes no edges.
HUB_MAX_FRACTION = 0.25
HUB_MIN_CHUNKS = 5


class EntityEdgeExtractor(EdgeExtractor):
    """Cross-chunk edges via shared entities/concepts (the primary multi-hop link).

    Terms come from either spaCy/GLiNER NER ("ner") or LLM concept extraction
    ("llm_concepts", see llm_concept.py) per `entity_extraction_mode`. The edge
    logic is identical — chunks sharing a term get linked — only the term source
    changes. "llm_concepts" needs `llm` (any `spectrag.IndexLLM`); with `NoLLM`
    the signal is skipped and yields no edges."""

    def __init__(
        self,
        cfg: IndexingConfig,
        ner_cfg: NERConfig,
        lang_cfg: LanguageConfig | None = None,
        ollama_cfg: OllamaConfig | None = None,
        *,
        llm: IndexLLM | None = None,
    ) -> None:
        self._min_freq     = cfg.entity_min_freq
        self._log_interval = cfg.entity_log_interval
        self._mode         = cfg.entity_extraction_mode
        self._skip         = False
        self._concept = self._ner = None
        if self._mode == "llm_concepts":
            if llm is None:
                raise TypeError("entity_extraction_mode='llm_concepts' needs llm= (an IndexLLM)")
            if isinstance(llm, NoLLM):
                logger.warning("NoLLM configured: LLM concept edges are skipped")
                self._skip = True
            else:
                from rag_system.indexing.edges.llm_concept import LLMConceptExtractor
                self._concept = LLMConceptExtractor(llm, cfg)
        else:
            self._ner = HybridNERExtractor(ner_cfg, lang_cfg, ollama_cfg)

    def _terms(self, chunk: Chunk) -> list[str]:
        if self._concept is not None:
            return self._concept.extract(chunk.text)
        return [ent.normalized for ent in self._ner.extract(chunk.text, language=chunk.language)]

    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        if self._skip:
            return []
        entity_to_chunks: dict[str, list[int]] = defaultdict(list)
        if self._concept is not None:
            # Parallel LLM concept extraction (llm_parallel_workers threads);
            # cache hits return instantly, misses retry transient API errors.
            per_chunk = self._concept.extract_batch(
                [c.text for c in chunks],
                progress_cb=lambda done, total: logger.info(
                    "Concept/entity extraction: %d / %d chunks", done, total),
                progress_every=self._log_interval,
            )
            for i, terms in enumerate(per_chunk):
                for term in set(terms):
                    entity_to_chunks[term].append(i)
            self._concept.close()
        else:
            for i, chunk in enumerate(chunks):
                if (i + 1) % self._log_interval == 0:
                    logger.info("Concept/entity extraction: %d / %d chunks", i + 1, len(chunks))
                # distinct terms: NER can return one term under several labels
                for term in set(self._terms(chunk)):
                    entity_to_chunks[term].append(i)

        # A term shared by a large share of the corpus ("machine learning") is a
        # hub: it would add C(n,2) edges and carries no information. Skip it.
        max_df = max(HUB_MIN_CHUNKS, int(HUB_MAX_FRACTION * len(chunks)))
        edge_counts: dict[tuple[int, int], float] = {}
        for term, chunk_idxs in entity_to_chunks.items():
            if len(chunk_idxs) < self._min_freq:
                continue
            if len(chunk_idxs) > max_df:
                logger.info("Skipping hub term %r (in %d of %d chunks)",
                            term, len(chunk_idxs), len(chunks))
                continue
            for a in range(len(chunk_idxs)):
                for b in range(a + 1, len(chunk_idxs)):
                    key = (chunk_idxs[a], chunk_idxs[b])
                    edge_counts[key] = edge_counts.get(key, 0.0) + 1.0

        if not edge_counts:
            return []

        # Jaccard similarity of the two chunks' term sets instead of a raw count: a
        # chunk that lists many terms (a steered extractor, or a boilerplate page)
        # shares only a small fraction of its terms with any one neighbour, so it
        # cannot buy strong links to everything by listing everything.
        n_terms = [0] * len(chunks)
        for chunk_idxs in entity_to_chunks.values():
            for i in chunk_idxs:
                n_terms[i] += 1
        scored = {(i, j): c / (n_terms[i] + n_terms[j] - c)
                  for (i, j), c in edge_counts.items()}
        max_score = max(scored.values())
        return [RawEdge(i, j, s / max_score) for (i, j), s in scored.items()]
