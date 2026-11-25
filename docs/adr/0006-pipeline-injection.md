# ADR 0006: Pipelines take components by injection

- Status: accepted (ARCH2c); the pipeline constructors are internal and provisional until the
  `SpectRAG` object replaces them as the public entry point (ARCH3)
- Date: 2026-11-25

## Context

ADR 0004 and ADR 0005 delivered `Embedder` and `IndexLLM` implementations. The pipelines still
built their own clients: `IndexingPipeline` and `QueryPipeline` constructed an Ollama embedder,
and each LLM call site (`llm_concept.py`, `utility_question.py`, `generator.py`) branched on a
`*_backend` string (`anthropic`/`openai`/`ollama`), with its own retry loop. The pipelines also
called `embed_chunks`/`build_faiss_index`, which only the legacy Ollama subclass had, so no other
embedder could be used. The human dropped the OpenAI backends (ARCH2d) and asked ARCH2c to remove
every OpenAI code path and the `[openai]` extra.

## Decision

1. **Every stage takes components, not configs, for its external dependencies.**
   `LLMConceptExtractor(llm, cfg)`, `UtilityQuestionEdgeExtractor(cfg, llm, embedder)`,
   `EntityEdgeExtractor(..., *, llm=None)`, `Generator(cfg, llm=None)`,
   `SeedRetriever(cfg, embedder)`. The pipelines take keyword-only `embedder=` and `llm=`:
   `IndexingPipeline(cfg, graph_store, *, embedder=None, llm=None)`,
   `QueryPipeline(cfg, *, embedder=None, llm=None)`. One injected `llm` serves both index-time
   signals (concepts and utility questions); the query pipeline's `llm` writes the answer.
2. **Backend names are resolved in one place**, `rag_system/components.py`
   (`build_embedder`, `build_llm`, `generation_sampling`), and only when the caller injected
   nothing. Accepted names: `anthropic` (default), `ollama`, `none`. `openai` fails with a message
   saying the backend was removed. The factory lives in `rag_system`, not `spectrag`: the public
   way to build components from settings is the `SpectRAG` object and its presets (ARCH3, I4).
3. **`NoLLM` handling** (ADR 0005 rule 9): `IndexingPipeline` skips the utility-question signal,
   `EntityEdgeExtractor` returns no concept edges, `Generator` returns `""` without a call.
   If `NoLLM` reaches a call site anyway, `LLMUnavailableError` propagates; it is never turned
   into "no concepts" / "no questions" like an ordinary per-chunk failure.
4. **Retries**: transport retries happen inside the `IndexLLM` (budget `indexing.llm_max_retries`).
   The concept extractor re-asks only when `complete_json` raises `ValueError` (invalid reply), up
   to `llm_max_retries` times. Utility questions use `complete` (free text), so a failure is final
   for that run. Failures are never cached, as before.
5. **Embedding helpers work with any `Embedder`**: `embed_chunks(embedder, chunks, cache_dir)` and
   `build_faiss_index(chunks, dimension)` are module functions in
   `rag_system/indexing/embedder.py`; the legacy `OllamaEmbedder(cfg)` methods delegate to them.
6. **Caches are keyed on `model_id`** (hard rule 6): `<model_id>.npz` for embeddings,
   `<model_id>.json` for concepts, `<model_id>_<n>q.json` for utility questions, with `/` and `:`
   replaced by `_`. Embedding and concept caches written before ARCH2c (named after the bare model)
   are still read when the new file does not exist; the utility-question name is unchanged for
   Anthropic. Existing caches therefore stay valid and nothing is re-requested from the LLM.
7. **The `rag_system.models` shim now warns** (ADR 0003 rule 3): every caller in the library,
   server and tests imports `spectrag.models`. `spectrag.formats.migrate` resolves legacy pickles
   that name `rag_system.models.Chunk` to `spectrag.models.Chunk` without importing the shim.
8. **OpenAI removed**: `OpenAIConfig`, `Config.openai`, the `openai` branches, the `[openai]`
   extra and the `requirements.txt` entry. `spectrag.settings` no longer exports `OpenAIConfig`.

## Consequences

- Any conforming `Embedder`/`IndexLLM` can be passed to the pipelines; a fully offline run is
  `IndexingPipeline(cfg, store, embedder=..., llm=NoLLM())` and needs no Ollama or API key.
- Retrieval output is unchanged with the default configuration (same prompts, models, sampling and
  caches); `tests/test_injection.py` checks the injected query pipeline against the golden results.
- Transport errors are retried `llm_max_retries` times in total instead of
  (SDK retries) × `llm_max_retries`; the worst-case cost of one failing chunk is lower.
- `rag_system/llm.py` stays (unused by the library) because `eval/llm.py` imports it; it is removed
  once the evaluation harness uses `AnthropicLLM`.
- The LLM-NER fallback in `rag_system/ner/extractor.py` still builds an Ollama client from
  `OllamaConfig`; it has no backend branch and is not one of the pipeline LLM call sites listed in
  ARCH2. Moving it to an injected `IndexLLM` is a follow-up.
- `tests/test_injection.py` pins the constructor keywords and checks that the call sites contain no
  backend branches, `import ollama`, `rag_system.llm` or OpenAI references.
