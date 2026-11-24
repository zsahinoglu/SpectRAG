"""Pinned models, prompts, seeds and paths. Every result file embeds `pins()`."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

HARNESS_VERSION = "0.1.0"
SEED = 1234

# One hosted model for every role (CLAUDE.md): index-time LLM, generator, judge.
INDEX_LLM_MODEL = "claude-haiku-5-5"
GENERATOR_MODEL = "claude-haiku-5-5"
JUDGE_MODEL = "claude-haiku-5-5"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 8

DATASETS = ("multihoprag", "ultradomain", "musique", "legal", "hotpotqa", "2wikimultihopqa")

JUDGE_SYSTEM = (
    "You are an impartial judge comparing two answers to the same question. "
    "Judge comprehensiveness, correctness with respect to the question, and directness. "
    "Ignore answer length unless it hurts clarity. Reply with exactly one token: A, B, or TIE."
)
JUDGE_USER = "Question:\n{question}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\nWhich answer is better? Reply A, B, or TIE."

CONTAMINATION_USER = (
    "Answer the question from memory in as few words as possible. "
    "If you do not know, reply UNKNOWN.\n\nQuestion: {question}"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _library_prompt_versions() -> dict:
    import prompts  # library prompts: hashed, never edited here
    names = ("GENERATION_SYSTEM", "GENERATION_CONTEXT_BLOCK", "GENERATION_PROMPT",
             "CONCEPT_EXTRACTION_PROMPT", "UTILITY_QUESTION_PROMPT", "LLM_NER_PROMPT")
    return {n.lower(): _sha(getattr(prompts, n)) for n in names if hasattr(prompts, n)}


PROMPT_VERSIONS = {
    "judge": _sha(JUDGE_SYSTEM + "\x00" + JUDGE_USER),
    "contamination": _sha(CONTAMINATION_USER),
    **_library_prompt_versions(),
}

SMOKE_DIR = Path(__file__).parent / "fixtures" / "smoke"


def dev_dir() -> Path:
    return Path(os.environ.get("SPECTRAG_DEV_DIR", "~/spectrag-data/eval-dev")).expanduser()


def test_dir() -> Path:
    return Path(os.environ.get("SPECTRAG_TEST_DIR", "~/spectrag-private/eval-test")).expanduser()


_SECRET_FIELDS = {"api_key", "password"}


def _scrub(obj):
    if isinstance(obj, dict):
        return {k: ("<redacted>" if k in _SECRET_FIELDS else _scrub(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def config_snapshot(cfg) -> dict:
    """The library Config as JSON, secrets redacted."""
    return _scrub(json.loads(json.dumps(dataclasses.asdict(cfg), default=str)))


def _drop_env(obj):
    """Remove machine-specific values (paths, caches, endpoints) before hashing."""
    if isinstance(obj, dict):
        return {k: _drop_env(v) for k, v in obj.items()
                if not (k.endswith("cache_dir") or k in ("store_path", "base_url"))}
    return obj


def config_hash(cfg) -> str:
    """Hash of the redacted config without paths/endpoints: stable across machines."""
    return _sha(json.dumps(_drop_env(config_snapshot(cfg)), sort_keys=True))


def cache_root() -> Path:
    return Path(os.environ.get("SPECTRAG_EVAL_CACHE", "~/.cache/spectrag-eval")).expanduser()


def pins() -> dict:
    return {
        "harness_version": HARNESS_VERSION,
        "seed": SEED,
        "index_llm_model": INDEX_LLM_MODEL,
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_temperature": JUDGE_TEMPERATURE,
        "prompt_versions": dict(PROMPT_VERSIONS),
    }
