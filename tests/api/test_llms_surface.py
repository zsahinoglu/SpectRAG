"""ARCH2b: pins the public surface of `spectrag.llms` (ADR 0005).

Changing anything asserted here is an API change and needs an ADR.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import spectrag.llms as llms
from spectrag.llms import AnthropicLLM, LLMUnavailableError, NoLLM, OllamaLLM

REPO_ROOT = Path(__file__).resolve().parents[2]
KW = "KEYWORD_ONLY"


def _params(fn, skip=1):
    sig = inspect.signature(fn)
    return [(p.name, p.kind.name, p.default) for p in list(sig.parameters.values())[skip:]]


def test_llms_all_is_pinned():
    assert sorted(llms.__all__) == ["AnthropicLLM", "LLMUnavailableError", "NoLLM", "OllamaLLM"]
    assert issubclass(LLMUnavailableError, RuntimeError)


def test_anthropic_constructor_is_pinned():
    assert _params(AnthropicLLM.__init__) == [
        ("model", "POSITIONAL_OR_KEYWORD", "claude-haiku-5-5"),
        ("api_key", KW, None),
        ("base_url", KW, None),
        ("timeout", KW, 120.0),
        ("max_retries", KW, 3),
        ("client", KW, None),
    ]


def test_ollama_constructor_is_pinned():
    assert _params(OllamaLLM.__init__) == [
        ("model", "POSITIONAL_OR_KEYWORD", "llama3.1:8b"),
        ("host", KW, "http://localhost:11434"),
        ("timeout", KW, 120.0),
        ("max_retries", KW, 3),
        ("client", KW, None),
    ]


def test_no_llm_takes_no_arguments():
    assert _params(NoLLM.__init__) in ([], [("args", "VAR_POSITIONAL", inspect.Parameter.empty),
                                           ("kwargs", "VAR_KEYWORD", inspect.Parameter.empty)])
    NoLLM()


def test_method_signatures_match_protocol():
    from spectrag.protocols import IndexLLM

    for cls in (AnthropicLLM, OllamaLLM, NoLLM):
        for name in ("complete", "complete_json"):
            assert _params(getattr(cls, name)) == _params(getattr(IndexLLM, name)), (cls, name)


def test_importing_llms_imports_no_backend():
    code = (
        "import sys, spectrag.llms\n"
        "heavy = {'config', 'spectrag.settings', 'ollama', 'openai', 'anthropic', "
        "'sentence_transformers', 'torch', 'faiss', 'rag_system'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT,
    ).stdout.strip()
    assert out == "[]"
