"""ARCH1: pins the public surface of the `spectrag` package skeleton.

Changing anything asserted here is an API change and needs an ADR.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import spectrag

REPO_ROOT = Path(__file__).resolve().parents[2]

SUBPACKAGES = [
    "ingest",
    "index",
    "retrieve",
    "stores",
    "embedders",
    "llms",
    "rerankers",
    "formats",
]


def test_top_level_all_is_pinned():
    assert sorted(spectrag.__all__) == sorted([
        "__version__",
        "Chunk",
        "QueryResult",
        "Embedder",
        "IndexLLM",
        "VectorStore",
        "GraphStore",
        "Reranker",
    ])
    for name in spectrag.__all__:
        assert hasattr(spectrag, name), name


def test_version_is_a_string():
    assert isinstance(spectrag.__version__, str) and spectrag.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_importable(name):
    mod = importlib.import_module(f"spectrag.{name}")
    assert mod.__doc__


def test_import_spectrag_is_lightweight():
    """`import spectrag` must not pull in settings/.env loading or optional backends."""
    code = (
        "import sys, spectrag\n"
        "heavy = {'config', 'spectrag.settings', 'neo4j', 'ollama', 'openai', "
        "'anthropic', 'faiss', 'torch', 'networkx', 'rag_system'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        cwd=REPO_ROOT,
    ).stdout.strip()
    assert out == "[]"


def test_settings_reexports_config_dataclasses():
    import config
    from spectrag import settings

    assert sorted(settings.__all__) == sorted([
        "Config",
        "EdgeWeights",
        "IndexingConfig",
        "LanguageConfig",
        "Neo4jConfig",
        "NERConfig",
        "OllamaConfig",
        "OpenAIConfig",
        "RetrievalConfig",
        "ServerConfig",
    ])
    for name in settings.__all__:
        assert getattr(settings, name) is getattr(config, name)
    assert isinstance(settings.Config(), config.Config)
