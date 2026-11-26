"""Stage-1 query latency at scale, with the fake embedder (no network, no LLM).

    python benchmarks/bench_query.py 10000 100000

For each corpus size it times SeedRetriever.retrieve (dense + BM25 hybrid) and the
pre-H6 reference path (per-query np.stack of all embeddings + rank_bm25's
per-document Python loop), and asserts both give identical relevance vectors.
"""
from __future__ import annotations

import statistics
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
    index.add(np.stack([c.embedding for c in chunks]).astype(np.float32))
    return emb, chunks, index


def old_scores(retr, query, chunks, q_emb, bm25_ref):
    """Pre-H6 relevance vector: restack embeddings and run rank_bm25 per query."""
    all_scores = (np.stack([c.embedding for c in chunks]).astype(np.float32) @ q_emb.T).squeeze()
    bm = np.clip(np.asarray(bm25_ref.get_scores(_tokenize(query)), dtype=np.float64), 0.0, None)
    w = retr.hybrid_lex_w
    dmax = float(all_scores.max())
    return (1.0 - w) * (all_scores / dmax) + w * (bm / bm.max())


def timed(fn, reps):
    out = []
    for _ in range(reps):
        for q in QUERIES:
            t = time.perf_counter()
            fn(q)
            out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out)


def main() -> None:
    from rank_bm25 import BM25Okapi
    sizes = [int(a) for a in sys.argv[1:] if a.isdigit()] or [10_000, 100_000]
    cfg = Config()
    cfg.retrieval.uq_seed_enabled = False
    print("| chunks | old ms/query | new ms/query | speedup | one-off cache build s |")
    print("|---:|---:|---:|---:|---:|")
    for n in sizes:
        emb, chunks, index = make_corpus(n)
        retr = SeedRetriever(cfg, emb)
        t = time.perf_counter()
        retr.retrieve(QUERIES[0], chunks, index)            # warm: builds caches
        build = time.perf_counter() - t
        ref = BM25Okapi([_tokenize(c.text) for c in chunks])
        for q in QUERIES:                                   # equivalence at this size
            q_emb = emb.embed([q], kind="query")
            _s, new = retr.retrieve(q, chunks, index)
            np.testing.assert_array_equal(new, old_scores(retr, q, chunks, q_emb, ref))
        new_ms = timed(lambda q: retr.retrieve(q, chunks, index), 3)
        old_ms = timed(lambda q: old_scores(retr, q, chunks, emb.embed([q], kind="query"), ref), 1)
        print(f"| {n:,} | {old_ms:.1f} | {new_ms:.1f} | {old_ms / new_ms:.1f}x | {build:.1f} |")


if __name__ == "__main__":
    main()
