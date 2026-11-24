"""Runs one evaluation level and returns an aggregate-only result dict."""
from __future__ import annotations

import datetime as _dt
import random
import resource
import subprocess
import time

import numpy as np

from config import Config
from eval import data, judge, metrics, pins
from eval.systems import (NaiveRAG, SpectRAG, build_full_index, build_offline_index,
                          temp_workdir, tiktoken_encoder)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _seed_everything() -> None:
    random.seed(pins.SEED)
    np.random.seed(pins.SEED)


def smoke_config() -> Config:
    cfg = Config()
    cfg.indexing.n_spectral_components = 4
    cfg.indexing.n_offline_clusters = 3
    cfg.indexing.semantic_threshold = 0.2
    cfg.retrieval.seed_k = cfg.retrieval.uq_seed_k = cfg.retrieval.bm25_seed_k = 4
    cfg.retrieval.n_clusters = 3
    cfg.retrieval.token_budget = 120
    return cfg


def full_config() -> Config:
    """Library defaults, with caches under the eval cache root and models pinned."""
    cfg = Config()
    root = pins.cache_root()
    cfg.ollama.embedding_cache_dir = str(root / "embeddings")
    cfg.indexing.utility_question_cache_dir = str(root / "uq")
    cfg.indexing.concept_cache_dir = str(root / "concepts")
    cfg.anthropic.model = pins.INDEX_LLM_MODEL
    assert pins.GENERATOR_MODEL == pins.INDEX_LLM_MODEL  # one Anthropic config serves both
    if {cfg.generation_backend, cfg.indexing.utility_question_llm_backend,
            cfg.indexing.concept_llm_backend} != {"anthropic"}:
        raise RuntimeError("eval requires the anthropic backend everywhere (no OpenAI)")
    return cfg


def evaluate_dataset(cfg: Config, ds: data.Dataset, *, offline: bool, embedder=None,
                     encoder=None, generator=None, judge_llm=None, meter=None) -> dict:
    """Build twice, query every question with SpectRAG and NaiveRAG, aggregate."""
    _seed_everything()
    builds, times = [], []
    with temp_workdir(ds.name) as tmp:
        for n in range(2):  # second build measures top-10 stability (LLM caches are warm)
            t0 = time.perf_counter()
            if offline:
                builds.append(build_offline_index(cfg, ds, embedder))
            elif n == 0 and meter is not None:
                with meter.watch():
                    builds.append(build_full_index(cfg, ds, _sub(tmp, n)))
            else:
                builds.append(build_full_index(cfg, ds, _sub(tmp, n)))
            times.append(time.perf_counter() - t0)
    peak = _peak_rss_mb()
    encoder = encoder or tiktoken_encoder(cfg)
    spec, spec_b = SpectRAG(cfg, builds[0], encoder), SpectRAG(cfg, builds[1], encoder)
    naive = NaiveRAG(cfg, builds[0], encoder)

    rows = {"spectrag": [], "naiverag": []}
    lat = {"spectrag": [], "naiverag": []}
    stability, wins, invalid = [], [], 0
    for q in ds.questions:
        res = {}
        for name, system in (("spectrag", spec), ("naiverag", naive)):
            t0 = time.perf_counter()
            r = res[name] = system.retrieve(q.question)
            lat[name].append((time.perf_counter() - t0) * 1000)
            row = {"evidence_recall": metrics.evidence_recall(r.ranked_doc_ids, list(q.evidence)),
                   "precision_at_5": metrics.precision_at_k(r.ranked_doc_ids, list(q.evidence)),
                   "mrr": metrics.reciprocal_rank(r.ranked_doc_ids, list(q.evidence))}
            if generator is not None:
                r.answer = generator.generate(q.question, r.chunks)
                if q.answers:
                    row["em"] = metrics.exact_match(r.answer, list(q.answers))
                    row["f1"] = metrics.f1(r.answer, list(q.answers))
            rows[name].append(row)
        stability.append(metrics.topk_overlap(res["spectrag"].top10_ids,
                                              spec_b.retrieve(q.question).top10_ids))
        if generator is not None and judge_llm is not None:
            j = judge.pairwise(judge_llm, q.question, res["spectrag"].answer, res["naiverag"].answer)
            wins.append(j["score"])
            invalid += j["invalid"]

    out = {"n_questions": len(ds.questions), "n_docs": len(ds.docs),
           "n_chunks": len(builds[0].chunks), "systems": {}}
    for name in rows:
        keys = rows[name][0].keys() if rows[name] else []
        agg = {k: round(metrics.mean([r[k] for r in rows[name] if k in r]), 4) for k in keys}
        agg["query_p50_ms"] = round(metrics.percentile(lat[name], 50), 2)
        agg["query_p95_ms"] = round(metrics.percentile(lat[name], 95), 2)
        out["systems"][name] = agg
    s = out["systems"]["spectrag"]
    s["top10_stability"] = round(metrics.mean(stability), 4)
    if wins:
        s["judge_win_rate_vs_naiverag"] = round(metrics.mean(wins), 4)
        s["judge_invalid_verdicts"] = invalid
    out["index"] = {"build_time_s": round(times[0], 3), "rebuild_time_s": round(times[1], 3),
                    "peak_rss_mb": round(peak, 1),
                    "cost": meter.as_dict() if meter is not None else
                    {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}}
    return out


def _sub(tmp: str, n: int):
    from pathlib import Path
    return Path(tmp) / f"build{n}"


def envelope(level: str, cfg: Config, datasets: dict, extra: dict | None = None) -> dict:
    return {"level": level,
            "date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "config_hash": pins.config_hash(cfg), "pins": pins.pins(),
            "config": pins.config_snapshot(cfg), "datasets": datasets, **(extra or {})}
