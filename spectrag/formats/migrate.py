"""One-way migration of legacy pickle indexes to the safe format (H3).

    python -m spectrag.formats.migrate <index-dir-or-store-dir> [...]

This is the ONLY module that reads pickle, and nothing on a load path imports it.
Pickle can execute code, so even here it is read through a restricted unpickler that
resolves only the handful of classes a legacy index contains (``Chunk``, NumPy array
reconstruction, NetworkX graph containers) and refuses every other global. Still, only
migrate files you created yourself or fully trust. Original ``.pkl`` files are left in
place (delete them once you have checked the result); the embedding cache is not
migrated, since it is rebuilt on demand.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import logging
import pickle
import sys
from pathlib import Path

import networkx as nx

from spectrag.formats.files import MANIFEST_FILE, write_chunks, write_graph
from spectrag.models import Chunk

logger = logging.getLogger(__name__)

_ALLOWED = {
    ("spectrag.models", "Chunk"), ("rag_system.models", "Chunk"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    ("collections", "OrderedDict"), ("builtins", "set"), ("builtins", "frozenset"),
}
# Exact NetworkX classes a legacy graph pickle can contain: the Graph itself plus the
# views that ``cached_property`` leaves in its ``__dict__``. No prefix rules.
_ALLOWED |= {("networkx.classes.graph", "Graph")} | {
    ("networkx.classes.reportviews", n)
    for n in ("NodeView", "EdgeView", "DegreeView", "NodeDataView", "EdgeDataView",
              "OutEdgeView", "OutEdgeDataView")}
_ALLOWED |= {("networkx.classes.coreviews", n) for n in ("AdjacencyView", "AtlasView")}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # Protocol >= 4 resolves dotted names by nested getattr; never allow them.
        if "." not in name and (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"legacy file references forbidden global {module}.{name}")


def _load_legacy(path: Path):
    return _RestrictedUnpickler(io.BytesIO(path.read_bytes())).load()


def _to_chunk(obj) -> Chunk:
    if not isinstance(obj, Chunk):
        obj_cls = type(obj).__name__
        raise pickle.UnpicklingError(f"expected Chunk, found {obj_cls}")
    # Old pickles restore __dict__ verbatim, so fields added since are missing.
    kw = {}
    for f in dataclasses.fields(Chunk):
        if hasattr(obj, f.name):
            kw[f.name] = getattr(obj, f.name)
        elif f.default is not dataclasses.MISSING:
            kw[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            kw[f.name] = f.default_factory()
        else:
            raise pickle.UnpicklingError(f"legacy chunk lacks required field {f.name!r}")
    return Chunk(**kw)


def migrate_index_dir(directory: Path | str) -> list[str]:
    """Convert ``chunks.pkl`` / ``graph.pkl`` in ``directory``. Returns what was converted."""
    d = Path(directory)
    done: list[str] = []
    legacy_chunks, legacy_graph = d / "chunks.pkl", d / "graph.pkl"
    if legacy_chunks.is_file() and not (d / MANIFEST_FILE).exists():
        if not (d / "faiss.index").is_file():
            raise FileNotFoundError(f"{d}/faiss.index missing; cannot migrate {legacy_chunks}")
        raw = _load_legacy(legacy_chunks)
        if not isinstance(raw, list):
            raise pickle.UnpicklingError("chunks.pkl does not contain a list")
        write_chunks(d, [_to_chunk(c) for c in raw], faiss_file="faiss.index",
                     migrated_from="chunks.pkl")
        done.append("chunks.pkl")
    if legacy_graph.is_file() and not (d / "graph" / MANIFEST_FILE).exists():
        g = _load_legacy(legacy_graph)
        if not isinstance(g, nx.Graph):
            raise pickle.UnpicklingError("graph.pkl does not contain a graph")
        write_graph(d / "graph", g)
        done.append("graph.pkl")
    return done


def migrate(path: Path | str) -> dict[str, list[str]]:
    """Migrate ``path`` itself, or every ``sessions/<id>`` under a server store dir."""
    p = Path(path)
    targets = [p] + (sorted(s for s in (p / "sessions").iterdir() if s.is_dir())
                     if (p / "sessions").is_dir() else [])
    return {str(t): done for t in targets if (done := migrate_index_dir(t))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args(argv)
    rc = 0
    for p in args.paths:
        try:
            result = migrate(p)
        except Exception as exc:  # noqa: BLE001 — report and continue with other paths
            print(f"FAILED {p}: {exc}", file=sys.stderr)
            rc = 1
            continue
        for where, done in result.items():
            print(f"migrated {where}: {', '.join(done)}")
        if not result:
            print(f"nothing to migrate in {p}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
