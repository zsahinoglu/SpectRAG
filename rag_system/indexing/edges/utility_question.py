from __future__ import annotations
import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
from spectrag.models import Chunk
from spectrag.llms import LLMUnavailableError, OllamaLLM
from spectrag.protocols import Embedder, IndexLLM
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge
from rag_system.indexing.edges.llm_concept import cache_name
from rag_system.language.detector import language_name
from config import IndexingConfig
from prompts import UTILITY_QUESTION_PROMPT
from rag_system.untrusted import clean_questions, fence

logger = logging.getLogger(__name__)


class UtilityQuestionEdgeExtractor(EdgeExtractor):
    """
    Functional-similarity edges: the LLM generates questions per chunk; chunks
    that answer similar questions receive an edge.

    `llm` is any `spectrag.IndexLLM` (transport retries are its job), `embedder`
    any `spectrag.Embedder`. Pass `NoLLM` only if the caller skips this signal.
    """

    def __init__(self, cfg: IndexingConfig, llm: IndexLLM, embedder: Embedder) -> None:
        self._n_questions    = cfg.utility_questions_per_chunk
        self._threshold      = cfg.utility_question_threshold
        self._search_k       = cfg.utility_question_search_k
        self._max_text_chars = cfg.utility_question_max_text_chars
        self._temperature    = cfg.utility_question_llm_temperature
        self._max_tokens     = cfg.utility_question_llm_max_tokens
        self._log_interval   = cfg.utility_question_log_interval
        self._llm            = llm
        self._embedder       = embedder
        # Hosted API calls parallelize well; a local Ollama server is kept
        # sequential (it serializes generations anyway).
        self._workers        = (1 if isinstance(llm, OllamaLLM)
                                else max(1, cfg.llm_parallel_workers))

        # Content-hash disk cache of generated questions, keyed by chunk text (+
        # language). The filename encodes model_id/count so a config change
        # never serves mismatched questions. Re-indexing after adding docs only
        # pays the LLM for genuinely new chunks; unchanged chunks are free.
        self._cache_dir = Path(cfg.utility_question_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / (
            f"{cache_name(llm.model_id)}_{self._n_questions}q.json"
        )
        self._cache: dict[str, list[str]] = {}
        if self._cache_file.exists():
            try:
                self._cache = json.loads(self._cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        self._cache_lock = threading.Lock()
        self._cache_dirty = 0

    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        # Generate questions for each chunk — LLM calls in parallel (hosted
        # backends); transient API errors are retried inside the IndexLLM.
        all_questions: list[list[str]] = [[] for _ in chunks]
        done = 0
        done_lock = threading.Lock()

        def _one(i: int) -> None:
            nonlocal done
            all_questions[i] = self._generate_questions(
                chunks[i].text, language_name(chunks[i].language))
            with done_lock:
                done += 1
                if done % self._log_interval == 0:
                    logger.info("Utility questions: %d / %d chunks", done, len(chunks))

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            list(pool.map(_one, range(len(chunks))))

        self._flush_cache()

        # Persist the questions on their chunk — reused at query time as
        # query-shaped surrogates for retrieval (see SeedRetriever).
        for ci, qs in enumerate(all_questions):
            chunks[ci].utility_questions = qs

        # Flatten, tracking which chunk each question belongs to
        flat_questions: list[str] = []
        q_to_chunk: list[int] = []
        for ci, qs in enumerate(all_questions):
            for q in qs:
                flat_questions.append(q)
                q_to_chunk.append(ci)

        if not flat_questions:
            return []

        # Embed all questions in one batched pass
        q_embeddings = self._embedder.embed(flat_questions)

        # ANN search over question embeddings
        k = min(self._search_k, len(flat_questions))
        index = faiss.IndexFlatIP(q_embeddings.shape[1])
        index.add(q_embeddings)
        scores_mat, idx_mat = index.search(q_embeddings, k)

        edges: dict[tuple[int, int], float] = {}
        for qi, (nbrs, sims) in enumerate(zip(idx_mat, scores_mat)):
            ci = q_to_chunk[qi]
            for qj, sim in zip(nbrs, sims):
                cj = q_to_chunk[qj]
                if ci == cj or sim < self._threshold:
                    continue
                key = (min(ci, cj), max(ci, cj))
                if sim > edges.get(key, -1.0):
                    edges[key] = float(sim)

        return [RawEdge(i, j, s) for (i, j), s in edges.items()]

    @staticmethod
    def _parse_questions(raw: str, n: int) -> list[str]:
        """Questions from model output (list markers stripped, non-questions dropped)."""
        return clean_questions(raw, n)

    def _generate_questions(self, text: str, language: str = "English") -> list[str]:
        snippet = text[: self._max_text_chars]
        # Cache key = language + text snippet (questions depend on both). A hit
        # returns without any LLM call.
        key = hashlib.sha256(f"{language}|{snippet}".encode("utf-8")).hexdigest()[:16]
        with self._cache_lock:
            if key in self._cache:
                return list(self._cache[key])

        prompt = UTILITY_QUESTION_PROMPT.format(
            n=self._n_questions,
            language=language,
            text=fence(snippet),
        )
        try:
            raw = self._llm.complete(
                prompt, temperature=self._temperature, max_tokens=self._max_tokens)
        except LLMUnavailableError:
            raise  # NoLLM must be handled by the caller, never turn into "no questions"
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not abort indexing
            # Transport retries already happened inside the IndexLLM. Failures are NOT
            # cached: the chunk is retried on the next index run.
            logger.warning("Question generation failed: %s", exc)
            return []
        questions = self._parse_questions(raw, self._n_questions)
        with self._cache_lock:
            self._cache[key] = questions
            self._cache_dirty += 1
            if self._cache_dirty >= 25:
                self._flush_cache_locked()
        return questions

    def _flush_cache_locked(self) -> None:
        """Write the question cache to disk. Caller must hold self._cache_lock."""
        try:
            tmp = self._cache_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            tmp.replace(self._cache_file)
            self._cache_dirty = 0
        except OSError:
            pass

    def _flush_cache(self) -> None:
        with self._cache_lock:
            self._flush_cache_locked()
