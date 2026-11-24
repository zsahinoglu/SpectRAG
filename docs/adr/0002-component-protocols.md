# ADR 0002: Component interfaces as `typing.Protocol`

- Status: accepted (ARCH1); member lists are provisional until the v1.0 freeze (ARCH3)
- Date: 2026-11-24

## Context

The pipelines construct their dependencies themselves (`ollama.Client`, `OpenAI`, Neo4j driver)
and choose between them with string flags such as `utility_question_llm_backend`. Tests cannot
substitute fakes without patching, and every new backend adds branches. The plan names five
pluggable components: `Embedder`, `IndexLLM`, `VectorStore`, `GraphStore`, `Reranker`.

## Decision

Define the five as `@runtime_checkable` `typing.Protocol` classes in `spectrag.protocols`,
re-exported from `spectrag`. Implementations do not subclass them (structural typing), so
third-party classes conform without importing SpectRAG.

| Protocol | Members |
| --- | --- |
| `Embedder` | `model_id: str`, `dimension: int`, `embed(texts, kind="document") -> ndarray` (float32, `(n, dimension)`, rows L2-normalized; `kind` is `"document"` or `"query"`) |
| `IndexLLM` | `model_id: str`, `complete(prompt, *, system, max_tokens, temperature) -> str`, `complete_json(prompt, schema, *, system, max_tokens, temperature)` (parsed and validated against a JSON Schema, else `ValueError`) |
| `VectorStore` | `add(ids, vectors)`, `remove(ids)`, `search(query, k) -> list[(chunk_id, score)]`, `__len__` |
| `GraphStore` | `write_graph(chunks, nx_graph)`, `bfs_expand(seed_chunk_ids, hops, max_candidates) -> set[str]`, `subgraph(chunk_ids) -> nx.Graph`, `node_count()`, `edge_count()`, `has_graph_data()`, `close()` |
| `Reranker` | `model_id: str`, `rerank(query, passages) -> list[float]` (input order) |

Choices and reasons:

- **`model_id` on embedders, LLMs and rerankers**: the index manifest pins model versions
  (hard rule 6); the component reports what to pin.
- **`complete_json` is part of `IndexLLM` from the start**: schema-validated outputs are
  required (C1, H7, I2). Adding a method to a Protocol later breaks every outside implementation,
  and backends with native structured output can use it.
- **LLM options are keyword-only**, so arguments can be added later without breaking positional
  callers.
- **Transport belongs to the implementation** (retries, timeouts, rate limits, key lookup such as
  `SPECTRAG_ANTHROPIC_API_KEY`); caching of index-time outputs stays in the pipeline, keyed by
  `model_id` and prompt version, so every backend gets it.
- **`GraphStore` mirrors the surface `InMemoryGraphStore` and `Neo4jGraphClient` already share**,
  so today's stores conform unchanged (tested for the in-memory store). The visualization helpers
  (`get_top_nodes_by_degree`, `get_edges_for_nodes`) are app concerns and left out. `connect()`
  is left out: factories hand over ready stores and `close()` releases them.
- **Persistence is not part of `VectorStore` or `GraphStore`**; the on-disk format belongs to
  `spectrag.formats` (H3), so stores cannot introduce their own (pickle-based) formats.
- `isinstance` checks only member names. Behavior is checked by conformance suites (H8 for
  `GraphStore`; ARCH2 for embedders and LLMs).

Diffusion and selection are *not* in this set; they become swappable strategy components inside
`spectrag.retrieve` in ARCH3, with their own ADR.

## Consequences

- `tests/api/test_protocols.py` pins member names and parameter lists; changing them requires a
  new ADR.
- ARCH2 adapts `OllamaEmbedder` (add `model_id`, `dimension`) and the LLM call sites to these
  interfaces, and the pipelines take components by injection.
- H8 may refine `GraphStore` (e.g. incremental `add`/`remove` for H4) by ADR before ARCH3 freezes
  it.
