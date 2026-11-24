from __future__ import annotations
import logging
from rag_system.models import Chunk
from rag_system.language.detector import detect_language, language_name
from rag_system.llm import anthropic_complete, make_anthropic_client
from config import Config
from prompts import GENERATION_SYSTEM, GENERATION_CONTEXT_BLOCK

logger = logging.getLogger(__name__)


class Generator:
    """Stage 5 — formats retrieved context and calls the LLM (Claude Haiku by default)."""

    def __init__(self, cfg: Config, client=None) -> None:
        """`client` lets tests inject a mocked LLM client for the chosen backend."""
        self._backend  = cfg.generation_backend
        self._lang_cfg = cfg.language
        if self._backend == "anthropic":
            self._llm_cfg = cfg.anthropic
            self._client  = client or make_anthropic_client(
                cfg.anthropic, "the final-answer generator")
        elif self._backend == "openai":
            self._llm_cfg = cfg.openai
            if not cfg.openai.api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set — generation_backend='openai' requires it."
                )
            from openai import OpenAI
            self._client = client or OpenAI(
                api_key=cfg.openai.api_key,
                base_url=cfg.openai.base_url,
                timeout=cfg.openai.request_timeout,
            )
        else:
            raise ValueError(f"Unknown generation_backend {self._backend!r}")

    def generate(self, question: str, context_chunks: list[Chunk]) -> str:
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

        if self._backend == "anthropic":
            return anthropic_complete(
                self._client, model=self._llm_cfg.model, system=system, user=user,
                temperature=self._llm_cfg.temperature,
                max_tokens=self._llm_cfg.max_tokens,
            ).strip()

        resp = self._client.chat.completions.create(
            model=self._llm_cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=self._llm_cfg.temperature,
            max_tokens=self._llm_cfg.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def close(self) -> None:
        pass  # clients manage their own HTTP pools
