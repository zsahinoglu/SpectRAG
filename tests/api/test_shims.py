"""ARCH1: old import paths keep working after the move to `spectrag` (ADR 0003)."""
from __future__ import annotations

import warnings


def test_rag_system_models_is_a_reexport():
    import rag_system.models as old
    import spectrag.models as new

    assert old.Chunk is new.Chunk
    assert old.QueryResult is new.QueryResult
    assert sorted(old.__all__) == ["Chunk", "QueryResult"]


def test_chunk_identity_across_paths():
    from rag_system.models import Chunk as OldChunk
    from spectrag import Chunk

    c = OldChunk("c1", "d1", "text", 0, [], [])
    assert isinstance(c, Chunk)
    assert c == Chunk("c1", "d2", "other", 3, [], [])  # equality by chunk_id
    assert hash(c) == hash("c1")


def test_shim_is_silent_until_internal_callers_migrate():
    import importlib
    import sys

    sys.modules.pop("rag_system.models", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        importlib.import_module("rag_system.models")
