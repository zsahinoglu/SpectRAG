"""ARCH2a: pins the public surface of `spectrag.embedders` (ADR 0004).

Changing anything asserted here is an API change and needs an ADR.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import spectrag.embedders as embedders
from spectrag.embedders import OllamaEmbedder, SentenceTransformerEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]


def _params(cls):
    sig = inspect.signature(cls.__init__)
    return [(p.name, p.kind.name, p.default) for p in list(sig.parameters.values())[1:]]


def test_embedders_all_is_pinned():
    assert sorted(embedders.__all__) == ["OllamaEmbedder", "SentenceTransformerEmbedder"]


def test_ollama_constructor_is_pinned():
    kw = "KEYWORD_ONLY"
    assert _params(OllamaEmbedder) == [
        ("model", "POSITIONAL_OR_KEYWORD", "nomic-embed-text"),
        ("host", kw, "http://localhost:11434"),
        ("timeout", kw, 120.0),
        ("batch_size", kw, 32),
        ("dimension", kw, None),
        ("query_prefix", kw, None),
        ("document_prefix", kw, None),
        ("client", kw, None),
    ]


def test_sentence_transformer_constructor_is_pinned():
    kw = "KEYWORD_ONLY"
    assert _params(SentenceTransformerEmbedder) == [
        ("model", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("revision", kw, None),
        ("device", kw, None),
        ("batch_size", kw, 32),
        ("query_prefix", kw, ""),
        ("document_prefix", kw, ""),
        ("client", kw, None),
    ]


def test_importing_embedders_imports_no_backend():
    code = (
        "import sys, spectrag.embedders\n"
        "heavy = {'config', 'spectrag.settings', 'ollama', 'openai', 'anthropic', "
        "'sentence_transformers', 'torch', 'faiss', 'rag_system'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT,
    ).stdout.strip()
    assert out == "[]"
