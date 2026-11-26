"""Stage-1 query latency at scale, with the fake embedder (no network, no LLM).

    python benchmarks/bench_query.py 10000 100000 1000000 [--reps 20]

For each corpus size it times the full ``SeedRetriever.retrieve`` (embed query, dense
scores, BM25 hybrid blend, FAISS search, RRF) against a reconstruction of the full
pre-H6 retrieve: per-query ``np.stack`` of all embeddings, and rank_bm25's per-document
Python loop run twice (once for the seed ranking, once for the hybrid blend). It asserts
both return identical seeds and relevance vectors, then reports median and p95 over
``reps`` repetitions of every query, after a warm-up pass for both paths.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from rag_system.query.seed_retrieval import SeedRetriever, _tokenize  # noqa: E402
from spectrag.models import Chunk  # noqa: E402
from tests.fakes import FakeEmbedder  # noqa: E402

VOCAB = [f"w{i}" for i in range(2000)]
QUERIES = ["w1 w7 w42", "w500 w501", "w1999 w3 w3 w88", "w10 w20 w30 w40"]


def make_corpus(n: int):
    rng = np.random.default_rng(0)
    emb = FakeEmbedder()
    chunks = []
    for i in range(n):
        words = rng.choice(VOCAB, size=30)
        chunks.append(Chunk(chunk_id=f"c{i}", doc_id=f"d{i // 20}", text=" ".join(words),
                            position=i, section_path=[], references=[]))
    for lo in range(0, n, 5000):
        part = chunks[lo:lo + 5000]
        for c, v in zip(part, emb.embed([c.text for c in part])):
            c.embedding = v
    index = faiss.IndexFlatIP(emb.dim)
    for lo in range(0, n, 50000):
        index.add(np.stack([c.embedding for c in chunks[lo:lo + 50000]]).astype(np.float32))
    return emb, chunks, index


def old_retrieve(retr: SeedRetriever, bm25_ref, query, chunks, index):
    """The pre-H6 retrieve, step for step: restack the embeddings per query and run
    rank_bm25's get_scores once for the hybrid blend and again for the seed ranking."""
    q_emb = retr.embedder.embed([query], kind="query")
    k = min(retr.k, len(chunks))
    all_scores = (np.stack([c.embedding for c in chunks]).astype(np.float32) @ q_emb.T).squeeze()
    if retr.hybrid_lex_w > 0 and retr.bm25_enabled:
        bm = np.clip(np.asarray(bm25_ref.get_scores(_tokenize(query)), dtype=np.float64), 0.0, None)
        if bm.max() > 0:
            dmax = float(all_scores.max())
            dense_n = all_scores / dmax if dmax > 0 else all_scores
            all_scores = (1.0 - retr.hybrid_lex_w) * dense_n + retr.hybrid_lex_w * (bm / bm.max())
    dense_k = min(max(retr.k, retr.uq_k), len(chunks))
    _, dense_idx = index.search(q_emb, dense_k)
    ranked = [[int(i) for i in dense_idx[0] if i >= 0]]
    if retr.bm25_enabled:
        raw = bm25_ref.get_scores(_tokenize(query))
        order = np.argsort(-raw)
        bl = [int(i) for i in order[: retr.bm25_k] if raw[i] > 0.0]
        if bl:
            ranked.append(bl)
    if len(ranked) == 1:
        return ranked[0][:k], all_scores
    return retr._rrf(ranked)[:k], all_scores


def timed(fn, reps):
    for q in QUERIES:                       # warm-up (not counted)
        fn(q)
    out = []
    for _ in range(reps):
        for q in QUERIES:
            t = time.perf_counter()
            fn(q)
            out.append((time.perf_counter() - t) * 1000)
    return float(np.median(out)), float(np.percentile(out, 95))


def main() -> None:
    from rank_bm25 import BM25Okapi
    args = sys.argv[1:]
    reps = 20
    if "--reps" in args:
        i = args.index("--reps")
        reps = int(args[i + 1])
        del args[i:i + 2]
    sizes = [int(a) for a in args if a.isdigit()] or [10_000, 100_000]
    cfg = Config()
    cfg.retrieval.uq_seed_enabled = False
    print(f"reps per query: {reps}, queries: {len(QUERIES)}, fake embedder, "
          f"hybrid_lexical_weight={cfg.retrieval.hybrid_lexical_weight}")
    print("| chunks | old median ms | old p95 ms | new median ms | new p95 ms "
          "| speedup (median) | one-off cache build s |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n in sizes:
        emb, chunks, index = make_corpus(n)
        retr = SeedRetriever(cfg, emb)
        t = time.perf_counter()
        retr.retrieve(QUERIES[0], chunks, index)            # builds caches
        build = time.perf_counter() - t
        ref = BM25Okapi([_tokenize(c.text) for c in chunks])
        for q in QUERIES:                                   # equivalence at this size
            ns, nsc = retr.retrieve(q, chunks, index)
            os_, osc = old_retrieve(retr, ref, q, chunks, index)
            assert ns == os_
            np.testing.assert_array_equal(nsc, osc)
        new_med, new_p95 = timed(lambda q: retr.retrieve(q, chunks, index), reps)
        old_med, old_p95 = timed(lambda q: old_retrieve(retr, ref, q, chunks, index), reps)
        print(f"| {n:,} | {old_med:.1f} | {old_p95:.1f} | {new_med:.1f} | {new_p95:.1f} "
              f"| {old_med / new_med:.1f}x | {build:.1f} |", flush=True)


if __name__ == "__main__":
    main()
