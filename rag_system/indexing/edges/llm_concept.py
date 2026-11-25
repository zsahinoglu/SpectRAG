"""LLM concept extraction for the entity/concept edge signal.

Instead of spaCy/GLiNER named entities (generic persons/orgs, weakly extracted),
this asks an LLM (Claude Haiku by default) for the domain CONCEPTS a passage is about, as
canonical short phrases. Two chunks that discuss the same concept in different
words — which cosine similarity misses — then share a concept key and get linked.
This is the cross-document signal genuinely ORTHOGONAL to the embedding space,
which is the graph's whole reason to exist.

Results are cached on disk by content hash, in one file per `IndexLLM.model_id`,
so re-indexing the same corpus (the per-domain eval loop does this a lot) is free
after the first pass. Transport errors are retried inside the `IndexLLM`; a reply
that fails schema validation is re-asked up to `llm_max_retries` times here
(ADR 0005). Failures are NEVER cached — a transient error must not permanently
strip a chunk of its concepts. extract_batch() runs the calls in a thread pool
(llm_parallel_workers) to cut indexing wall-clock.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import IndexingConfig
from prompts import CONCEPT_EXTRACTION_PROMPT
from spectrag.llms import LLMUnavailableError
from spectrag.protocols import IndexLLM

logger = logging.getLogger(__name__)

#: Output contract of CONCEPT_EXTRACTION_PROMPT (validated by `IndexLLM.complete_json`).
CONCEPTS_SCHEMA = {
    "type": "object",
    "properties": {"concepts": {"type": "array", "items": {"type": "string"}}},
    "required": ["concepts"],
}


def cache_name(model_id: str) -> str:
    """Filesystem-safe form of a `model_id` for cache file names."""
    return model_id.replace("/", "_").replace(":", "_")


class LLMConceptExtractor:
    def __init__(self, llm: IndexLLM, cfg: IndexingConfig) -> None:
        self._llm = llm
        self._max_chars = cfg.concept_max_text_chars
        self._max_tokens = cfg.concept_llm_max_tokens
        self._workers = max(1, cfg.llm_parallel_workers)
        self._max_retries = max(1, cfg.llm_max_retries)

        self._cache_dir = Path(cfg.concept_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / f"{cache_name(llm.model_id)}.json"
        # Caches written before ARCH2c are named after the bare model ("claude-haiku-5-5.json").
        legacy = self._cache_dir / f"{llm.model_id.split(':', 1)[-1].replace('/', '_')}.json"
        self._cache: dict[str, list[str]] = {}
        for f in (self._cache_file, legacy):
            if f.exists():
                try:
                    self._cache = json.loads(f.read_text(encoding="utf-8"))
                    break
                except (json.JSONDecodeError, OSError):
                    self._cache = {}
        self._dirty = 0
        self._lock = threading.Lock()  # guards _cache/_dirty across pool threads

    def _call_llm(self, snippet: str) -> list[str]:
        """One extraction; an invalid reply is re-asked. Raises after the final attempt."""
        prompt = CONCEPT_EXTRACTION_PROMPT.format(text=snippet)
        last_exc: Exception | None = None
        for _ in range(self._max_retries):
            try:
                data = self._llm.complete_json(
                    prompt, CONCEPTS_SCHEMA, temperature=0.0, max_tokens=self._max_tokens)
            except ValueError as exc:  # reply failed parsing/validation → ask again
                last_exc = exc
                continue
            return sorted({c.strip().lower() for c in data["concepts"] if c.strip()})
        raise last_exc  # type: ignore[misc]

    def extract(self, text: str) -> list[str]:
        """Concept keys for one chunk (cached by content hash; failures NOT cached)."""
        snippet = text[: self._max_chars]
        key = hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:16]
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        try:
            concepts = self._call_llm(snippet)
        except LLMUnavailableError:
            raise  # NoLLM must be handled by the caller, never turn into "no concepts"
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not abort indexing
            logger.warning("Concept extraction failed (%s); no concepts for this chunk "
                           "(not cached — retried next run)", exc)
            return []

        with self._lock:
            self._cache[key] = concepts
            self._dirty += 1
            if self._dirty >= 25:
                self._flush_locked()
        return concepts

    def extract_batch(self, texts: list[str],
                      progress_cb=None, progress_every: int = 50) -> list[list[str]]:
        """Concepts for many chunks, LLM calls in parallel. Order preserved."""
        results: list[list[str]] = [[] for _ in texts]
        done = 0
        done_lock = threading.Lock()

        def _one(i: int) -> None:
            nonlocal done
            results[i] = self.extract(texts[i])
            if progress_cb:
                with done_lock:
                    done += 1
                    if done % progress_every == 0:
                        progress_cb(done, len(texts))

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            list(pool.map(_one, range(len(texts))))
        return results

    def _flush_locked(self) -> None:
        """Write the cache to disk. Caller must hold self._lock."""
        try:
            self._cache_file.write_text(json.dumps(self._cache), encoding="utf-8")
            self._dirty = 0
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
