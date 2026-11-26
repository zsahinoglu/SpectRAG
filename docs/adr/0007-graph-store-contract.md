# ADR 0007: GraphStore behavioral contract

- Status: proposed (H8; awaiting architect acceptance)
- Date: 2026-11-25

## Context

ADR 0002 fixed the `GraphStore` member names. `InMemoryGraphStore` (also the "file store" once
saved/loaded as Parquet) and `Neo4jGraphClient` had drifted on details that change retrieval:
which neighbours survive the `max_candidates` cap (set order vs. database order), edges that
name chunks that were never written (kept as new nodes vs. dropped), and duplicate chunk ids.

## Decision

1. `write_graph` replaces the whole graph. Nodes are exactly the unique `chunk_id`s in `chunks`;
   edges with an endpoint outside `chunks` are dropped.
2. `bfs_expand` follows edges in both directions. Seeds are always returned (even unknown ones,
   even beyond the cap). Within a hop, each new node is ranked by the **maximum `weight` of its
   edges to the hop's frontier, descending, ties broken by ascending chunk_id**, and admitted in
   that order until `len(result) >= max_candidates` (missing weight counts as 0.0). Human-approved
   (option B); independent of chunk-id naming.
3. `subgraph` returns a fresh undirected graph containing every requested id, plus the induced
   edges with a `weight` attribute.
4. `close()` is idempotent.

Executable form: `tests/conformance/graph_store.py::check_graph_store(factory)`. Run against
in-memory, file round-trip, and Neo4j (set `SPECTRAG_TEST_NEO4J_URI`; skipped otherwise). The
golden-corpus parity test compares full pipeline output across stores.

Scope: H8a covers the contract, in-memory and file stores. `Neo4jGraphClient.bfs_expand` does not
yet implement the weight ordering (H8b); its tests are skipped until then. Tie-break uses
code-point order, so ids should be ASCII (Cypher compares UTF-16 code units).
The Neo4j test factory requires a dedicated database (`SPECTRAG_TEST_NEO4J_DATABASE`, must not
be `neo4j`) because `write_graph` deletes every `Chunk` node; it empties that database first.

## Consequences

Retrieval is deterministic when the cap truncates a hop. Golden output is unchanged (the golden
corpus does not truncate mid-hop), but real corpora that hit the cap (default 200) will change:
the evaluator's dev-split delta is required before merge. Neo4j conformance is only exercised where a server exists.
