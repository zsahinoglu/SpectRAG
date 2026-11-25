"""H5: isolated chunks get zero spectral credit; ARPACK retries then raises; residuals logged."""
from __future__ import annotations

import json
import logging

import networkx as nx
import numpy as np
import pytest
import scipy.sparse.linalg as spla

from config import IndexingConfig
from rag_system.indexing import spectral as spectral_mod
from rag_system.indexing.spectral import SpectralComputer, SpectralConvergenceError
from spectrag.formats import read_manifest, write_chunks
from spectrag.models import Chunk


def _chunks(n):
    return [Chunk(chunk_id=f"c{i}", doc_id="d", text="t", position=i,
                  section_path=[], references=[]) for i in range(n)]


def _graph(n_connected, n_isolated):
    G = nx.Graph()
    ids = [f"c{i}" for i in range(n_connected + n_isolated)]
    G.add_nodes_from(ids)
    # two cliques joined by a bridge, so the spectrum is non-degenerate
    half = n_connected // 2
    for grp in (range(half), range(half, n_connected)):
        for a in grp:
            for b in grp:
                if a < b:
                    G.add_edge(ids[a], ids[b], weight=1.0)
    G.add_edge(ids[half - 1], ids[half], weight=0.1)
    return G


def _computer(k=3):
    return SpectralComputer(IndexingConfig(n_spectral_components=k))


def test_isolated_chunks_get_zero_coords():
    chunks = _chunks(14)
    sc = _computer()
    res = sc.compute(_graph(12, 2), chunks)
    assert np.all(res[12:] == 0.0)
    assert np.all(chunks[13].spectral_coords == 0.0)
    assert np.any(res[:12] != 0.0, axis=1).all()
    assert sc.diagnostics["n_isolated"] == 2


def test_residual_norms_recorded_and_small():
    sc = _computer()
    sc.compute(_graph(12, 0), _chunks(12))
    d = sc.diagnostics
    assert len(d["residual_norms"]) == len(d["eigenvalues"]) == 4
    assert d["max_residual_norm"] < 1e-6
    json.dumps(d)  # manifest-serialisable


def test_residuals_logged(caplog):
    with caplog.at_level(logging.INFO, logger=spectral_mod.logger.name):
        _computer().compute(_graph(12, 0), _chunks(12))
    assert any("residual norm" in r.message for r in caplog.records)


def test_arpack_retries_then_succeeds(monkeypatch):
    real = spla.eigsh
    calls = []

    def flaky(*a, **kw):
        calls.append(kw)
        if len(calls) < 3:
            raise spla.ArpackNoConvergence("no conv", np.array([]), np.zeros((12, 0)))
        return real(*a, **kw)

    monkeypatch.setattr(spectral_mod.spla, "eigsh", flaky)
    res = _computer().compute(_graph(12, 0), _chunks(12))
    assert len(calls) == 3 and np.isfinite(res).all()


def test_arpack_raises_after_retries(monkeypatch):
    calls = []

    def never(*a, **kw):
        calls.append(1)
        raise spla.ArpackNoConvergence("no conv", np.array([0.0]), np.zeros((12, 1)))

    monkeypatch.setattr(spectral_mod.spla, "eigsh", never)
    with pytest.raises(SpectralConvergenceError):
        _computer().compute(_graph(12, 0), _chunks(12))
    assert len(calls) > 1


def test_diagnostics_written_to_manifest(tmp_path):
    sc = _computer()
    chunks = _chunks(12)
    sc.compute(_graph(12, 0), chunks)
    write_chunks(tmp_path, chunks, spectral=sc.diagnostics)
    m = read_manifest(tmp_path)
    assert m["spectral"]["residual_norms"] == sc.diagnostics["residual_norms"]
