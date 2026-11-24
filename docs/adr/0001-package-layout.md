# ADR 0001: `spectrag` package layout

- Status: accepted (ARCH1)
- Date: 2026-11-24

## Context

The research code is spread across top-level modules (`config.py`, `prompts.py`, `main.py`,
`server.py`) and two packages (`rag_system/`, `rag_server/`). Library code, the demo web app and
the CLI share one namespace, nothing can be installed, and backends (Neo4j, Ollama, OpenAI) are
imported unconditionally. Section 3 of the Production Library Plan targets one importable
package with optional extras and the web app split out as `spectrag-app` at v1.0.

## Decision

One top-level package, `spectrag/`, with these modules:

| Module | Responsibility | Receives (from) |
| --- | --- | --- |
| `spectrag.models` | `Chunk`, `QueryResult` | `rag_system/models.py` (moved in ARCH1) |
| `spectrag.settings` | config dataclasses, later presets (I4) | `config.py` (re-exported now, moved later) |
| `spectrag.protocols` | the five component interfaces (ADR 0002) | new |
| `spectrag.ingest` | loading, chunking, language detection | `rag_system/indexing/chunker.py`, `rag_system/language/`, `rag_system/ner/` |
| `spectrag.index` | edge signals, graph build, calibration, spectral embedding | `rag_system/indexing/` |
| `spectrag.retrieve` | seeding, expansion, diffusion, selection | `rag_system/query/` |
| `spectrag.stores` | `VectorStore` / `GraphStore` implementations | `rag_system/store/` |
| `spectrag.embedders` | `Embedder` implementations | `rag_system/indexing/embedder.py` |
| `spectrag.llms` | `IndexLLM` implementations | LLM calls in edges and `rag_system/generation/` |
| `spectrag.rerankers` | `Reranker` implementations | new |
| `spectrag.formats` | on-disk format, manifests (H3) | `rag_system/store/index_store.py` |

Rules:

- `import spectrag` stays cheap: it imports only `models` and `protocols`. It must not import
  `spectrag.settings` (which still loads `.env`), optional backends, `faiss`, `networkx` or
  `rag_system`. `tests/api/test_package_surface.py` checks this.
- Backend code (Neo4j, Ollama, OpenAI, Anthropic, GLiNER, sentence-transformers) is only
  imported inside the implementing module under `stores/`, `embedders/`, `llms/`, `rerankers/`,
  so a missing extra fails at construction time with a clear message, not at `import spectrag`.
- `rag_server/`, `server.py`, `frontend/`, `landing/` are app code. They do not move into
  `spectrag/`; ARCH3 moves them to `spectrag-app`.
- The single `SpectRAG` facade object is added once components are injectable (ARCH2) and frozen
  in ARCH3. ARCH1 does not define it, so no name is pinned before its shape is known.
- `__version__` comes from installed package metadata (`importlib.metadata`), so the version
  has one source: `pyproject.toml` (H2a; owned by the release manager).

## Consequences

- Builders can start targeting `spectrag.*` import paths now; the eight subpackages exist as
  empty placeholders with docstrings that state what each will hold.
- Modules move one at a time, each in a small submission with a shim (ADR 0003), so moves do not
  collide with in-flight builder work on `rag_system/`.
- H2a packages `spectrag` (and, until the moves finish, `rag_system`, `config`, `prompts`).
