"""Tests for the evaluation harness (E0a). Offline: fakes only, no paid calls."""
from __future__ import annotations

import json
import time

import pytest

import eval.__main__ as cli
from eval import data, fakes, judge, metrics, pins
from eval.runner import smoke_config


# ---- metrics ----------------------------------------------------------------

def test_retrieval_metrics():
    ranked = ["x", "a", "y", "b", "a"]
    assert metrics.evidence_recall(ranked, ["a", "b", "c"]) == pytest.approx(2 / 3)
    assert metrics.precision_at_k(ranked, ["a", "b"], k=5) == pytest.approx(3 / 5)
    assert metrics.precision_at_k(["a"], ["a"], k=5) == pytest.approx(1 / 5)  # denominator is k
    assert metrics.reciprocal_rank(ranked, ["b"]) == pytest.approx(1 / 4)
    assert metrics.reciprocal_rank(ranked, ["z"]) == 0.0


def test_answer_metrics_use_squad_normalisation():
    assert metrics.exact_match("The Helium.", ["helium"]) == 1.0
    assert metrics.exact_match("hydrogen", ["helium"]) == 0.0
    assert metrics.f1("a neutron star", ["a neutron star or a black hole"]) == pytest.approx(4 / 7)  # P=2/2, R=2/5 after dropping articles
    assert metrics.f1("anything", []) == 0.0
    assert metrics.answer_contains("It is helium, I think", ["Helium"]) == 1.0


def test_stability_and_percentiles():
    assert metrics.topk_overlap(list("abcdefghij"), list("abcdefghiz")) == pytest.approx(0.9)
    assert metrics.topk_overlap([], []) == 1.0
    assert metrics.percentile([1, 2, 3, 4, 100], 50) == 3.0


# ---- judge ------------------------------------------------------------------

class _Fixed:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, user, system=None):
        self.prompts.append((system, user))
        return self.replies.pop(0)


def test_pairwise_judge_swaps_order_and_scores():
    # System wins in first order (A) and second order (B) -> 1.0
    assert judge.pairwise(_Fixed("A", "B"), "q", "s", "n")["score"] == 1.0
    # Position-biased judge always says A -> one win, one loss -> 0.5
    llm = _Fixed("A", "A")
    assert judge.pairwise(llm, "q", "sys", "base")["score"] == 0.5
    assert "Answer A:\nsys" in llm.prompts[0][1] and "Answer A:\nbase" in llm.prompts[1][1]
    assert llm.prompts[0][0] == pins.JUDGE_SYSTEM
    out = judge.pairwise(_Fixed("garbage", "TIE."), "q", "s", "n")
    assert out == {"invalid": 1, "score": 0.5}
    # "Answer B" in sys_first order means the baseline won; it must never count as a
    # system win (it used to be read as "A"). Non-verdicts are INVALID, not guessed.
    out = judge.pairwise(_Fixed("Answer B", "Both"), "q", "s", "n")
    assert out == {"invalid": 2, "score": 0.5}
    assert judge.pairwise(_Fixed("B.", "**A**"), "q", "s", "n") == {"invalid": 0, "score": 0.0}
    assert judge.pairwise(_Fixed("a\n", "tie"), "q", "s", "n") == {"invalid": 0, "score": 0.75}


@pytest.mark.parametrize("reply,verdict", [
    ("A", "A"), ("B.", "B"), (" `TIE` ", "TIE"), ("tie", "TIE"),
    ("Answer B", "INVALID"), ("Both", "INVALID"), ("Baseline is better", "INVALID"),
    ("A and B are equal", "INVALID"), ("TIEBREAK", "INVALID"), ("", "INVALID"), (None, "INVALID"),
])
def test_judge_verdict_parser_is_strict(reply, verdict):
    assert judge._verdict(reply) == verdict


def test_contamination_reports_aggregates_only():
    qs = [data.Question("1", "q1", ("Paris",), ("d",)), data.Question("2", "q2", ("x",), ("d",)),
          data.Question("3", "q3", (), ("d",))]
    out = judge.contamination(_Fixed("Paris", "UNKNOWN"), qs)
    assert out == {"checked": 2, "closed_book_em": 0.5, "closed_book_contains": 0.5}
    assert "q1" not in json.dumps(out)


# ---- pins -------------------------------------------------------------------

def test_pins_are_haiku_and_config_hash_ignores_secrets_and_paths(monkeypatch):
    p = pins.pins()
    assert {p["index_llm_model"], p["generator_model"], p["judge_model"]} == {"claude-haiku-5-5"}
    assert {"judge", "contamination", "generation_system"} <= set(p["prompt_versions"])
    a = smoke_config()
    b = smoke_config()
    b.anthropic.api_key, b.neo4j.password = "sk-secret", "pw"
    b.ollama.embedding_cache_dir = "/elsewhere"
    assert pins.config_hash(a) == pins.config_hash(b)
    assert "sk-secret" not in json.dumps(pins.config_snapshot(b))
    b.retrieval.token_budget += 1
    assert pins.config_hash(a) != pins.config_hash(b)


def test_full_config_is_haiku_everywhere_with_eval_caches(monkeypatch, tmp_path):
    from eval.runner import full_config
    monkeypatch.setenv("SPECTRAG_EVAL_CACHE", str(tmp_path))
    cfg = full_config()
    assert cfg.anthropic.model == "claude-haiku-5-5"
    assert cfg.generation_backend == cfg.indexing.concept_llm_backend == "anthropic"
    assert cfg.indexing.utility_question_llm_backend == "anthropic"
    assert cfg.ollama.embedding_cache_dir.startswith(str(tmp_path))


# ---- data -------------------------------------------------------------------

def _write(tmp_path, corpus, questions):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in corpus))
    (d / "questions.jsonl").write_text("\n".join(json.dumps(r) for r in questions))


def test_loader_rejects_bad_evidence_without_leaking_text(tmp_path):
    _write(tmp_path, [{"doc_id": "d1", "text": "t"}],
           [{"qid": "1", "question": "SECRET QUESTION", "answers": ["SECRET"], "evidence": ["d9"]}])
    with pytest.raises(data.DataError) as e:
        data.load_dataset(tmp_path, "ds")
    assert "SECRET" not in str(e.value) and "questions.jsonl:1" in str(e.value)


def test_loader_sorts_and_limits(tmp_path):
    _write(tmp_path, [{"doc_id": "d1", "text": "t"}],
           [{"qid": q, "question": q, "evidence": ["d1"]} for q in ("b", "a", "c")])
    ds = data.load_dataset(tmp_path, "ds", limit=2)
    assert [q.qid for q in ds.questions] == ["a", "b"] and not ds.has_answers


# ---- smoke run ---------------------------------------------------------------

def _smoke():
    result = cli.run_smoke()  # the same metered path as `python -m eval smoke`
    assert result["llm_calls_attempted"] == 0 and result["error"] is None
    return result["datasets"]["smoke"]


def test_smoke_metrics_are_deterministic_and_sane():
    a, b = _smoke(), _smoke()

    def quality(r):  # everything except wall-clock latencies
        return {k: {m: v for m, v in s.items() if not m.startswith("query_p")}
                for k, s in r["systems"].items()}

    assert quality(a) == quality(b)
    spec, naive = a["systems"]["spectrag"], a["systems"]["naiverag"]
    assert spec["top10_stability"] == 1.0
    for m in (spec, naive):
        assert 0 < m["evidence_recall"] <= 1 and 0 <= m["mrr"] <= 1 and "em" in m and "f1" in m
    assert 0 <= spec["judge_win_rate_vs_naiverag"] <= 1
    # The fixture is easy for lexical retrieval: both systems find most evidence.
    assert naive["evidence_recall"] >= 0.75 and spec["evidence_recall"] >= 0.75


def test_smoke_cli_runs_fast_and_passes(capsys):
    t0 = time.perf_counter()
    assert cli.main(["smoke"]) == 0
    assert time.perf_counter() - t0 < 300
    out = capsys.readouterr().out
    assert "| smoke | spectrag |" in out and "smoke ok" in out
    with capsys.disabled():  # visible with `pytest -s`: the smoke table in CI logs
        print("\n" + out)


@pytest.fixture
def sdk(monkeypatch):
    """The real `anthropic` SDK if installed, else a minimal stand-in with the same
    `anthropic.resources.messages.Messages.create` seam the meter wraps. Either way the
    unwrapped `create` must never be reached in these tests."""
    try:
        import anthropic
        from anthropic.resources.messages import Messages
    except ImportError:
        import sys
        import types

        class Messages:
            def create(self, *a, **kw):
                raise AssertionError("network call reached the (stand-in) SDK")

        class Anthropic:
            def __init__(self, api_key=None):
                self.messages = Messages()

        anthropic = types.ModuleType("anthropic")
        resources = types.ModuleType("anthropic.resources")
        messages = types.ModuleType("anthropic.resources.messages")
        messages.Messages, anthropic.Anthropic = Messages, Anthropic
        anthropic.resources, resources.messages = resources, messages
        for name, mod in (("anthropic", anthropic), ("anthropic.resources", resources),
                          ("anthropic.resources.messages", messages)):
            monkeypatch.setitem(sys.modules, name, mod)
    return anthropic, Messages


def _paid_call(anthropic):
    anthropic.Anthropic(api_key="dummy-never-sent").messages.create(
        model=pins.JUDGE_MODEL, max_tokens=1, messages=[{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("swallowed", [False, True])
def test_smoke_fails_when_a_paid_call_is_attempted(monkeypatch, capsys, sdk, swallowed):
    """The no-paid-calls invariant must actually fire, even if the calling code
    swallows the exception (as library fallbacks may)."""
    anthropic, Messages = sdk
    original = Messages.create

    def generate(self, question, chunks):
        try:
            _paid_call(anthropic)
        except Exception:
            if not swallowed:
                raise
        return ""

    monkeypatch.setattr(fakes.ExtractiveGenerator, "generate", generate)
    assert cli.main(["smoke"]) == 1
    err = capsys.readouterr().err
    assert "SMOKE FAIL" in err and "paid LLM call" in err
    assert Messages.create is original  # the meter restores the SDK


def test_usage_meter_counts_calls_and_tokens(monkeypatch, sdk):
    from types import SimpleNamespace
    from eval.llm import UsageMeter
    anthropic, Messages = sdk
    fake_resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=3))
    monkeypatch.setattr(Messages, "create", lambda self_, *a, **kw: fake_resp)
    meter = UsageMeter()
    with meter.watch():
        _paid_call(anthropic)
        _paid_call(anthropic)
    assert meter.as_dict() == {"llm_calls": 2, "input_tokens": 14, "output_tokens": 6}
    _paid_call(anthropic)  # outside watch(): not counted
    assert meter.calls == 2


# ---- run preconditions fail clearly -------------------------------------------

def test_run_without_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("SPECTRAG_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "agent-billing-key-must-not-be-used")
    assert cli.main(["run", "--dev"]) == 2
    assert "SPECTRAG_ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_release_without_test_dir_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SPECTRAG_ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("SPECTRAG_TEST_DIR", str(tmp_path / "absent"))
    assert cli.main(["run", "--release"]) == 2
    assert "SPECTRAG_TEST_DIR" in capsys.readouterr().err


def test_release_requires_every_dataset(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SPECTRAG_ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("SPECTRAG_TEST_DIR", str(tmp_path))
    (tmp_path / "hotpotqa").mkdir()
    (tmp_path / "hotpotqa" / "corpus.jsonl").write_text("")
    assert cli.main(["run", "--release"]) == 2
    assert "missing" in capsys.readouterr().err


def test_release_rejects_limit():
    with pytest.raises(SystemExit):
        cli.main(["run", "--release", "--limit", "5"])


def test_release_rejects_dataset_subset(monkeypatch, tmp_path, capsys):
    """A one-dataset run must not be able to pass as a complete release run."""
    monkeypatch.setenv("SPECTRAG_ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("SPECTRAG_TEST_DIR", str(tmp_path))
    (tmp_path / "hotpotqa").mkdir()
    (tmp_path / "hotpotqa" / "corpus.jsonl").write_text("")
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--release", "--datasets", "hotpotqa"])
    assert e.value.code != 0
    assert "--datasets is not allowed with --release" in capsys.readouterr().err


def test_dev_with_no_datasets_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SPECTRAG_ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("SPECTRAG_DEV_DIR", str(tmp_path))
    assert cli.main(["run", "--dev"]) == 2
    assert "no datasets" in capsys.readouterr().err
