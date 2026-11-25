"""Old import paths keep working after the move to `spectrag` (ADR 0003, ADR 0006)."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def _import_old():
    sys.modules.pop("rag_system.models", None)
    return importlib.import_module("rag_system.models")


def test_rag_system_models_is_a_reexport():
    import spectrag.models as new

    with pytest.warns(DeprecationWarning):
        old = _import_old()
    assert old.Chunk is new.Chunk
    assert old.QueryResult is new.QueryResult
    assert sorted(old.__all__) == ["Chunk", "QueryResult"]


def test_chunk_identity_across_paths():
    from spectrag import Chunk

    with pytest.warns(DeprecationWarning):
        OldChunk = _import_old().Chunk
    c = OldChunk("c1", "d1", "text", 0, [], [])
    assert isinstance(c, Chunk)
    assert c == Chunk("c1", "d2", "other", 3, [], [])  # equality by chunk_id
    assert hash(c) == hash("c1")


def test_shim_warns_and_names_the_new_path():
    """ARCH2c migrated every internal caller, so the shim now warns (ADR 0003 rule 3)."""
    with pytest.warns(DeprecationWarning, match=r"spectrag\.models"):
        _import_old()


def test_library_code_does_not_import_the_shim():
    """Importing the pipelines, the server services and the migration tool must
    not touch the deprecated module."""
    code = (
        "import sys\n"
        "import rag_system.indexing.pipeline, rag_system.query.pipeline\n"
        "import rag_system.store.index_store, rag_server.services, spectrag.formats.migrate\n"
        "assert 'rag_system.models' not in sys.modules, 'deprecated shim imported'\n"
    )
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
