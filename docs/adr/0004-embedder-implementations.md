# ADR 0004: Embedder implementations

- Status: accepted (ARCH2a); constructor signatures provisional until the v1.0 freeze (ARCH3)
- Date: 2026-11-25

## Context

ADR 0002 defined the `Embedder` Protocol. The only implementation, `OllamaEmbedder` in
`rag_system/indexing/embedder.py`, has no `model_id`/`dimension`, is built from `OllamaConfig`,
imports `ollama` at module level, and also holds pipeline concerns: a pickle-backed embedding
cache (`embed_chunks`) and FAISS index building. ARCH2 asks for Ollama, sentence-transformers and
OpenAI embedders that conform to the Protocol and can be injected into the pipelines. ARCH2 is
split into ARCH2a (embedders, this ADR), ARCH2b (LLMs), ARCH2c (injection into the pipelines) and
ARCH2d (OpenAI backends, pending a human decision).

## Decision

1. **`spectrag.embedders` exports `OllamaEmbedder` and `SentenceTransformerEmbedder`.** Each
   lives in its own module; the backend package is imported inside the constructor, so
   `import spectrag.embedders` imports no backend and a missing extra fails at construction
   with the `pip install "spectrag[...]"` line.
2. **Constructors take plain arguments, not config objects.** The model name is the only
   positional argument; everything else is keyword-only. No `spectrag.embedders` module imports
   `config`/`spectrag.settings`. Building components from settings is a factory concern (ARCH2c).
3. **`client=` is the injection point on every implementation**: an already-built backend object
   (Ollama client, loaded `SentenceTransformer`). Tests pass fakes; applications can share a
   configured client. The same keyword is used by the LLM implementations (ARCH2b).
4. **`model_id` is `"<provider>:<model>[@<revision>]"`**, e.g. `ollama:nomic-embed-text`,
   `sentence-transformers:BAAI/bge-small-en-v1.5@<commit>`. The provider prefix keeps ids from
   different backends distinct in index manifests (hard rule 6). Ollama tags are mutable, so the
   Ollama id cannot pin weights; digest pinning is left to the manifest work (H3).
5. **`dimension` is what the model actually returns.** If the backend does not report it, the
   first access embeds one probe text and caches the result. An expected `dimension=` argument
   (Ollama) only triggers a warning on mismatch, as before; a size change mid-run is an error.
6. **Shared input and output rules** (`spectrag/embedders/_common.py`): a bare `str` is a
   `TypeError`, an unknown `kind` is a `ValueError`, any sequence of `str` is accepted, output is
   float32 with L2-normalized rows, and a reply with the wrong number of rows is an error
   (retried for Ollama), so a vector can never be mapped onto the wrong text.
7. **Task prefixes are constructor arguments.** Ollama keeps today's behavior (nomic prefixes for
   nomic models, none otherwise) as the default when a prefix is `None`; sentence-transformers
   defaults to no prefix because the right one depends on the model.
8. **`SentenceTransformerEmbedder.model` has no default.** Picking a local model is a quality
   decision for presets (I4), not for the class.
9. **The legacy class stays as a subclass.** `rag_system.indexing.embedder.OllamaEmbedder(cfg)`
   now subclasses `spectrag.embedders.OllamaEmbedder` and keeps `embed_chunks` and
   `build_faiss_index` until ARCH2c moves them into the pipelines; the pickle cache is H3's to
   replace. The old import path therefore yields a conforming `Embedder` today.
10. **Conformance suite**: `tests/conformance/embedder.py::check_embedder` checks the behavior
    `isinstance` cannot (dtype, shape, unit rows, order and batch independence, empty input,
    argument validation, stable `dimension`). Every implementation, the legacy subclass and the
    test `FakeEmbedder` run it.

## Consequences

- Pipelines can be handed any conforming embedder once ARCH2c lands; nothing in the pipelines
  changes in ARCH2a and retrieval output is unchanged (same prefixes, batching, normalization).
- sentence-transformers is shipped in the `[rerank]` extra today; whether it gets its own
  extra name is settled with the extras list in the v1.0 freeze (ARCH3).
- The OpenAI embedder follows the same rules in `spectrag/embedders/openai.py` if ARCH2d is
  approved.
- `tests/api/test_embedders_surface.py` pins the exports and constructor signatures.
