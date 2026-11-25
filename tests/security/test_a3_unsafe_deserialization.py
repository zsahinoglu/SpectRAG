"""R1 / A3 — unsafe deserialization in index load paths. Must be made to pass by H3.

Threat: an index directory is data. Users download shared indexes, restore
backups, point the server at a session folder, or share an embedding cache.
Today three load paths call ``pickle.load`` on files from those places, so
anyone who can write one of those files runs code as the SpectRAG process:

* ``IndexStore.load``               -> ``<store>/chunks.pkl``
* ``InMemoryGraphStore.load``       -> ``sessions/<id>/graph.pkl``
* ``OllamaEmbedder.embed_chunks``   -> ``<embedding_cache_dir>/<model>.pkl``

The behavioural tests plant a pickle whose only effect is creating a sentinel
file, then go through the public entry points (including the server's
``ActiveIndex.ensure_loaded``). They pass as soon as the payload no longer runs,
however H3 achieves that (new format, refusing legacy files, ...). The static
test pins the hard rule "no pickle in any load path" for the whole code base.

Tests marked ``H3`` fail today (strict xfail). H3 makes them pass and removes
the marker; an unmarked test here is a control that must keep passing.
"""
from __future__ import annotations

import ast
import contextlib
import pickle
from pathlib import Path

import faiss
import numpy as np
import pytest

H3 = pytest.mark.xfail(strict=True, reason="R1/A3 open: H3 (safe on-disk format) must make "
                                           "this pass, then drop this marker")

REPO = Path(__file__).resolve().parents[2]


class _Payload:
    """Unpickling this creates ``path``. Stands in for arbitrary code execution."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)

    def __reduce__(self):
        return (open, (self.path, "w"))


def _plant(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(obj, protocol=5))


def _write_faiss(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = faiss.IndexFlatIP(4)
    idx.add(np.eye(4, dtype=np.float32))
    faiss.write_index(idx, str(path))


def test_control_payload_fires_when_unpickled(tmp_path):
    """Sanity: the payload really does execute under plain pickle.load."""
    sentinel = tmp_path / "pwned"
    _plant(tmp_path / "x.pkl", [_Payload(sentinel)])
    with open(tmp_path / "x.pkl", "rb") as fh:
        pickle.load(fh).pop().close()
    assert sentinel.exists()


# ---- behavioural: planted files on the real entry points ---------------------------

@H3
def test_index_store_load_does_not_execute_planted_chunks_file(tmp_path):
    from rag_system.store.index_store import IndexStore

    sentinel = tmp_path / "pwned-chunks"
    store = tmp_path / "index"
    _plant(store / "chunks.pkl", [_Payload(sentinel)])
    _write_faiss(store / "faiss.index")
    with contextlib.suppress(Exception):
        IndexStore(store).load()
    assert not sentinel.exists(), "IndexStore.load executed code from chunks.pkl"


@H3
def test_graph_store_load_does_not_execute_planted_graph_file(tmp_path):
    from rag_system.store.memory_graph import InMemoryGraphStore

    sentinel = tmp_path / "pwned-graph"
    _plant(tmp_path / "graph.pkl", _Payload(sentinel))
    with contextlib.suppress(Exception):
        InMemoryGraphStore().load(tmp_path / "graph.pkl")
    assert not sentinel.exists(), "InMemoryGraphStore.load executed code from graph.pkl"


@H3
def test_server_session_load_does_not_execute_planted_files(tmp_path):
    """The path a running server takes when a user opens a session."""
    from config import Config
    from rag_server.runtime import ServerPaths
    from rag_server.services import ActiveIndex

    paths = ServerPaths(tmp_path / "store")
    sid = "abc123"
    sdir = paths.session_dir(sid)
    s_chunks, s_graph = tmp_path / "pwned-session-chunks", tmp_path / "pwned-session-graph"
    _plant(sdir / ServerPaths.CHUNKS_FILE, [_Payload(s_chunks)])
    _plant(paths.graph_file(sid), _Payload(s_graph))
    _write_faiss(sdir / "faiss.index")
    with contextlib.suppress(Exception):
        ActiveIndex(Config(), paths).ensure_loaded(sid)
    fired = [p.name for p in (s_chunks, s_graph) if p.exists()]
    assert not fired, f"opening session {sid!r} executed planted payloads: {fired}"


class _NetworkCalled(BaseException):
    """Raised by the network stubs. A BaseException, so tenacity's
    ``retry_if_exception_type(Exception)`` neither retries nor sleeps on it."""


@H3
def test_embedding_cache_does_not_execute_planted_cache_file(tmp_path, monkeypatch):
    """A shared/restored embedding cache is loaded on the next indexing run.

    Today the planted cache holds a valid vector for the chunk, so the chunk is
    a cache hit. Once H3 stops loading legacy ``.pkl`` caches, it is a miss and
    ``embed_chunks`` falls through to Ollama. The network side is stubbed to
    raise ``_NetworkCalled``, so the test stays offline (no request, no retry
    backoff) both before and after H3. The miss is expected and suppressed;
    only the sentinel decides the outcome."""
    import ollama

    from config import OllamaConfig
    from rag_system.indexing.embedder import OllamaEmbedder, _text_hash
    from tests.security._fixtures import chunk

    def _no_network(*_a, **_k):
        raise _NetworkCalled("unit test tried to reach Ollama")

    monkeypatch.setattr(OllamaEmbedder, "_embed_batch", _no_network)
    monkeypatch.setattr(ollama.Client, "embed", _no_network)  # backstop if _embed_batch is renamed

    cfg = OllamaConfig()
    cfg.embedding_cache_dir = str(tmp_path / "emb")
    c = chunk("c-0", "d", "some chunk text")
    sentinel = tmp_path / "pwned-cache"
    cache_file = Path(cfg.embedding_cache_dir) / f"{cfg.embedding_model.replace('/', '_')}.pkl"
    _plant(cache_file, {_text_hash(c.text): np.ones(4, dtype=np.float32),
                        "zz-payload": _Payload(sentinel)})
    with contextlib.suppress(Exception, _NetworkCalled):
        OllamaEmbedder(cfg).embed_chunks([c])
    assert not sentinel.exists(), "embedding cache load executed code from the .pkl file"


# ---- static: no unsafe loader anywhere in shipped code -----------------------------

_SCAN_EXCLUDE = {"tests", ".claude", ".git", "node_modules", "frontend", "landing", "assets",
                 "build", "dist", ".venv", "venv"}

# Module-level calls that deserialize arbitrary objects.
_ALWAYS_UNSAFE = {
    ("pickle", "load"), ("pickle", "loads"), ("pickle", "Unpickler"),
    ("cPickle", "load"), ("cPickle", "loads"), ("_pickle", "load"), ("_pickle", "loads"),
    ("dill", "load"), ("dill", "loads"), ("cloudpickle", "load"), ("cloudpickle", "loads"),
    ("joblib", "load"), ("marshal", "load"), ("marshal", "loads"), ("shelve", "open"),
    ("yaml", "unsafe_load"), ("yaml", "full_load"), ("yaml", "unsafe_load_all"),
    ("pd", "read_pickle"), ("pandas", "read_pickle"),
}
_SAFE_YAML_LOADERS = {"SafeLoader", "CSafeLoader", "BaseLoader", "CBaseLoader"}

# In library packages, merely importing a pickle-family module is a finding.
# ``from pickle import load``, ``import pickle as p`` and ``getattr(pickle, "load")``
# all get past the call matcher above, but every one of them needs the import.
_LIBRARY_PACKAGES = {"spectrag", "rag_system", "rag_server"}
_PICKLE_FAMILY = {"pickle", "cPickle", "_pickle", "dill", "cloudpickle", "marshal", "shelve"}


def _kw(call: ast.Call, name: str):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _is_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _unsafe_calls(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id in {"eval", "exec"}:
            found.append((node.lineno, f"{f.id}()"))
            continue
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)):
            continue
        mod, attr = f.value.id, f.attr
        if (mod, attr) in _ALWAYS_UNSAFE:
            found.append((node.lineno, f"{mod}.{attr}"))
        elif mod == "torch" and attr == "load" and not _is_true(_kw(node, "weights_only")):
            found.append((node.lineno, "torch.load without weights_only=True"))
        elif mod in {"np", "numpy"} and attr == "load" and _is_true(_kw(node, "allow_pickle")):
            found.append((node.lineno, "np.load(allow_pickle=True)"))
        elif mod == "yaml" and attr in {"load", "load_all"}:
            loader = _kw(node, "Loader") or (node.args[1] if len(node.args) > 1 else None)
            name = getattr(loader, "attr", getattr(loader, "id", None))
            if name not in _SAFE_YAML_LOADERS:
                found.append((node.lineno, f"yaml.{attr} without SafeLoader"))
    return found


def _pickle_imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module]
        else:
            continue
        found += [(node.lineno, f"import {m}") for m in mods
                  if m.split(".")[0] in _PICKLE_FAMILY]
    return found


def _in_library(rel: Path) -> bool:
    return bool(rel.parts) and rel.parts[0] in _LIBRARY_PACKAGES


# H3's one-way migration tool must read legacy pickles. It is exempt only by its exact
# repo-relative POSIX path, added to this set in H3's own submission (so Lens reviews
# it). It must never be imported by a load path; the behavioural tests above check
# that. Empty until H3 creates the module.
_MIGRATION_ALLOWLIST: frozenset[str] = frozenset()


def _shipped_python_files() -> list[Path]:
    out = []
    for p in REPO.rglob("*.py"):
        rel = p.relative_to(REPO)
        if _SCAN_EXCLUDE & set(rel.parts):
            continue
        if rel.as_posix() in _MIGRATION_ALLOWLIST:
            continue
        out.append(p)
    return sorted(out)


def test_control_scanner_detects_each_pattern():
    src = (
        "import pickle, yaml, torch, numpy as np\n"
        "pickle.load(f)\n"
        "yaml.load(s)\n"
        "yaml.load(s, Loader=yaml.SafeLoader)\n"
        "torch.load(p)\n"
        "torch.load(p, weights_only=True)\n"
        "np.load(p, allow_pickle=True)\n"
        "np.load(p)\n"
        "eval(x)\n"
    )
    kinds = [k for _, k in _unsafe_calls(ast.parse(src))]
    assert kinds == ["pickle.load", "yaml.load without SafeLoader",
                     "torch.load without weights_only=True", "np.load(allow_pickle=True)",
                     "eval()"], kinds
    assert len(_shipped_python_files()) > 20, "scan found too few files; check REPO path"


def test_control_scanner_detects_evasive_pickle_imports():
    src = (
        "from pickle import load\n"
        "import pickle as p\n"
        "import dill\n"
        "from cloudpickle import loads as l2\n"
        "import os, marshal\n"
        "from . import pickle\n"          # relative import of a local module: not flagged
        "import pickletools_like\n"       # different top-level name: not flagged
        "load(f)\n"
    )
    kinds = [k for _, k in _pickle_imports(ast.parse(src))]
    assert kinds == ["import pickle", "import pickle", "import dill", "import cloudpickle",
                     "import marshal"], kinds
    assert _in_library(Path("rag_system/store/index_store.py"))
    assert not _in_library(Path("scripts/tool.py"))
    for entry in _MIGRATION_ALLOWLIST:  # exact relative paths only, no globs
        assert not entry.startswith("/") and ".." not in entry and "*" not in entry, entry
        assert (REPO / entry).is_file(), f"allowlisted migration module missing: {entry}"


@H3
def test_no_unsafe_deserialization_in_shipped_code():
    hits = []
    for p in _shipped_python_files():
        rel = p.relative_to(REPO)
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        found = _unsafe_calls(tree)
        if _in_library(rel):
            found += _pickle_imports(tree)
        hits += [f"{rel}:{line}: {what}" for line, what in sorted(found)]
    assert not hits, "unsafe deserialization in shipped code:\n  " + "\n  ".join(hits)
