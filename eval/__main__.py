"""CLI: `python -m eval smoke` | `python -m eval run --dev` | `python -m eval run --release`.

Exit codes: 0 ok, 1 smoke invariant failed, 2 precondition missing (key, data).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from eval import data, pins

RESULTS_DIR = Path(os.environ.get("SPECTRAG_EVAL_OUT", Path(__file__).parent / "results"))


def _table(result: dict) -> str:
    cols = ["evidence_recall", "precision_at_5", "mrr", "em", "f1", "judge_win_rate_vs_naiverag",
            "query_p95_ms", "top10_stability"]
    lines = ["| dataset | system | " + " | ".join(cols) + " | build_s | index_tokens |",
             "|" + " --- |" * (len(cols) + 4)]
    for ds, r in result["datasets"].items():
        cost = r["index"]["cost"]
        for name, m in r["systems"].items():
            vals = [str(m.get(c, "—")) for c in cols]
            lines.append(f"| {ds} | {name} | " + " | ".join(vals) +
                         f" | {r['index']['build_time_s']} | "
                         f"{cost['input_tokens'] + cost['output_tokens']} |")
    return "\n".join(lines)


def _save(result: dict, level: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = RESULTS_DIR / f"{level}-{stamp}-{result['commit'][:9]}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _smoke_invariants(result: dict) -> list[str]:
    bad = []
    for ds, r in result["datasets"].items():
        for name, m in r["systems"].items():
            for k in ("evidence_recall", "precision_at_5", "mrr", "em", "f1",
                      "judge_win_rate_vs_naiverag", "top10_stability"):
                if k in m and not 0.0 <= m[k] <= 1.0:
                    bad.append(f"{ds}/{name}/{k} out of [0,1]: {m[k]}")
        if r["systems"]["spectrag"]["top10_stability"] != 1.0:
            bad.append(f"{ds}: offline build is not deterministic (top-10 stability < 1)")
    if result["llm_calls_attempted"]:
        bad.append(f"smoke attempted {result['llm_calls_attempted']} paid LLM call(s)")
    return bad


def run_smoke() -> dict:
    """The whole smoke run (build, query, generate, judge) happens under a meter that
    counts and blocks every Anthropic SDK call, so "no paid calls" is checked, not assumed."""
    from eval import fakes
    from eval.llm import PaidCallBlocked, UsageMeter
    from eval.runner import envelope, evaluate_dataset, smoke_config
    t0 = time.perf_counter()
    cfg = smoke_config()
    ds = data.load_dataset(pins.SMOKE_DIR.parent, pins.SMOKE_DIR.name)
    meter = UsageMeter()
    datasets, error = {}, None
    try:
        with meter.watch(forbid=True):
            datasets["smoke"] = evaluate_dataset(
                cfg, ds, offline=True, embedder=fakes.HashEmbedder(),
                encoder=fakes.WhitespaceEncoder(), generator=fakes.ExtractiveGenerator(),
                judge_llm=fakes.LexicalJudgeLLM())
    except PaidCallBlocked as e:
        error = str(e)
    return envelope("smoke", cfg, datasets, {
        "offline_fakes": True, "llm_calls_attempted": meter.calls, "error": error,
        "elapsed_s": round(time.perf_counter() - t0, 2)})


def cmd_smoke(args) -> int:
    result = run_smoke()
    if result["datasets"]:
        print(_table(result))
    if args.save:
        print(f"saved {_save(result, 'smoke')}")
    bad = _smoke_invariants(result)
    for b in bad:
        print(f"SMOKE FAIL: {b}", file=sys.stderr)
    print(f"smoke {'FAILED' if bad else 'ok'} in {result['elapsed_s']}s")
    return 1 if bad else 0


def cmd_run(args) -> int:
    from eval.llm import HaikuLLM, MissingKey, UsageMeter, require_key
    level = "release" if args.release else "dev"
    root = pins.test_dir() if args.release else pins.dev_dir()
    env = "SPECTRAG_TEST_DIR" if args.release else "SPECTRAG_DEV_DIR"
    try:
        require_key()
    except MissingKey as e:
        print(f"eval: {e}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"eval: {level} split not found at {root} (set {env})", file=sys.stderr)
        return 2
    wanted = list(pins.DATASETS) if args.release else (args.datasets or list(pins.DATASETS))
    present = data.available(root, wanted)
    missing = sorted(set(pins.DATASETS) - set(present))  # vs. all six, so subsets are visible
    if args.release and missing:
        print(f"eval: release run needs every dataset; missing under {env}: {missing}",
              file=sys.stderr)
        return 2
    if not present:
        print(f"eval: no datasets under {root} (expected <dataset>/corpus.jsonl)", file=sys.stderr)
        return 2

    from rag_system.generation.generator import Generator
    from eval import judge
    from eval.runner import envelope, evaluate_dataset, full_config
    cfg = full_config()
    judge_llm = HaikuLLM(pins.JUDGE_MODEL, max_tokens=pins.JUDGE_MAX_TOKENS)
    index_llm = HaikuLLM(pins.INDEX_LLM_MODEL, max_tokens=64)
    generator = Generator(cfg)
    results, contamination = {}, {}
    for name in present:
        ds = data.load_dataset(root, name, limit=None if args.release else args.limit)
        contamination[name] = judge.contamination(index_llm, ds.questions)
        results[name] = evaluate_dataset(cfg, ds, offline=False, generator=generator,
                                         judge_llm=judge_llm, meter=UsageMeter())
    result = envelope(level, cfg, results, {"contamination": contamination,
                                            "evaluated_datasets": list(present),
                                            "skipped_datasets": missing,
                                            "complete": not missing and not args.limit,
                                            "limit": args.limit})
    print(_table(result))
    print(f"saved {_save(result, level)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="offline run on the bundled fixture (no paid calls)")
    s.add_argument("--save", action="store_true", help="also write the result JSON")
    r = sub.add_parser("run", help="full run with Claude Haiku")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true", help="dev splits ($SPECTRAG_DEV_DIR)")
    g.add_argument("--release", action="store_true", help="frozen test splits (evaluator only)")
    r.add_argument("--datasets", nargs="+", choices=pins.DATASETS)
    r.add_argument("--limit", type=int, default=None, help="questions per dataset (dev only)")
    args = p.parse_args(argv)
    if getattr(args, "release", False) and args.limit:
        p.error("--limit is not allowed with --release")
    if getattr(args, "release", False) and args.datasets:
        p.error("--datasets is not allowed with --release (all six datasets are required)")
    logging.basicConfig(level=logging.WARNING)
    try:
        return cmd_smoke(args) if args.cmd == "smoke" else cmd_run(args)
    except data.DataError as e:
        print(f"eval: data error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
