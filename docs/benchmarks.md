# Stage-1 retrieval benchmarks

Command: `python benchmarks/bench_query.py 10000 100000 1000000 --reps 20`

Setup: fake embedder (no network, no LLM), synthetic corpus of 30-word chunks over a
2000-word vocabulary, `hybrid_lexical_weight=0.3`, UQ seeds off, 4 queries x 20
repetitions (80 timed calls per cell) after a warm-up pass for both paths. Machine:
4 cores, 15 GB RAM, single process. The 10^6 row fit in memory, so it is included.

- **new**: the full `SeedRetriever.retrieve` (embed query, dense scores from the cached
  matrix, BM25 scored once per query via postings, hybrid blend, FAISS search, RRF).
- **old**: a step-for-step reconstruction of the pre-H6 `retrieve` (per-query `np.stack` of
  all embeddings, rank_bm25 `get_scores` twice: blend and seed ranking). The benchmark
  asserts seeds and relevance vectors are identical to the new path at every size.
- "one-off cache build" is the first query's cost (BM25 postings, embedding matrix).

| chunks | old median ms | old p95 ms | new median ms | new p95 ms | speedup (median) | one-off cache build s |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 34.4 | 43.4 | 16.0 | 34.0 | 2.1x | 0.2 |
| 100,000 | 234.6 | 282.2 | 20.0 | 28.1 | 11.7x | 2.2 |
| 1,000,000 | 2233.5 | 2973.0 | 52.0 | 60.1 | 43.0x | 22.4 |

The new path still scales linearly in N (dense matmul and the FAISS flat search); the
per-query Python loops over the corpus are gone. Timings are machine-dependent; a 10^4
run earlier on the same box gave 32.9 / 16.3 ms, so run-to-run noise is a few ms.
