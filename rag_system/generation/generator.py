from __future__ import annotations
import logging
from spectrag.models import Chunk
from spectrag.llms import NoLLM
from spectrag.protocols import IndexLLM
from rag_system.components import build_llm, generation_sampling
from rag_system.language.detector import detect_language, language_name
from config import Config
from prompts import GENERATION_SYSTEM, GENERATION_CONTEXT_BLOCK

logger = logging.getLogger(__name__)


class Generator:
    """Stage 5 — formats retrieved context and asks the LLM (Claude Haiku by default).

    `llm` is any `spectrag.IndexLLM`; when omitted it is built from
    `cfg.generation_backend`. With `NoLLM` no answer is generated (`""`).
    """

    def __init__(self, cfg: Config, llm: IndexLLM | None = None) -> None:
        self._lang_cfg = cfg.language
        self._llm = llm if llm is not None else build_llm(cfg, cfg.generation_backend)
        self._temperature, self._max_tokens = generation_sampling(cfg)

    def generate(self, question: str, context_chunks: list[Chunk]) -> str:
        if isinstance(self._llm, NoLLM):
            return ""
        lang_code = detect_language(question, fallback=self._lang_cfg.fallback_language)
        lang      = language_name(lang_code)
        logger.debug("Query language detected: %s (%s)", lang, lang_code)

        # Group context by source document, with a header per source, so the
        # generator can see — and synthesize across — multiple sources. Applied
        # identically to SpectRAG and the NaiveRAG baseline (both share this
        # generator), so it stays a fair formatting change: it simply reflects
        # each system's real source spread (SpectRAG's graph expansion typically
        # spans more documents than naive top-k).
        groups: dict[str, list[Chunk]] = {}
        for chunk in context_chunks:
            src = chunk.metadata.get("source", chunk.doc_id)
            groups.setdefault(src, []).append(chunk)

        blocks: list[str] = []
        n = 0
        for src, group_chunks in groups.items():
            blocks.append(f"=== Source: {src} ===")
            for chunk in group_chunks:
                n += 1
                blocks.append(GENERATION_CONTEXT_BLOCK.format(
                    i=n, source=src, pos=chunk.position, text=chunk.text,
                ))
        context_text = "\n".join(blocks)

        system = GENERATION_SYSTEM.format(language=lang)
        user   = f"{context_text}\nQuestion: {question}\n\nAnswer:"
        return self._llm.complete(
            user, system=system, temperature=self._temperature, max_tokens=self._max_tokens,
        ).strip()

    def close(self) -> None:
        pass  # LLM components manage their own HTTP pools
