"""Coverage tests for weight calibration and multiplex spectral (H1b)."""
import numpy as np
import pytest

from config import EdgeWeights, IndexingConfig
from rag_system.indexing.edges.base import RawEdge
from rag_system.indexing.spectral import SpectralComputer
from rag_system.indexing.weight_calibrator import (
    calibrate_edge_weights,
    fit_edge_weights,
)
from spectrag.models import Chunk

N = 40
_NAMES = ("semantic", "section", "entity", "adjacency", "utility_question", "citation")


def _signals(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    adj = [RawEdge(i, i + 1, 1.0) for i in range(N - 1)]
    sec = [RawEdge(i, i + 2, 1.0) for i in range(N - 2)]
    sem = [RawEdge(i, i + 1, float(rng.random()) + 1.0) for i in range(N - 1)]
    sem += [RawEdge(i, i + 5, float(rng.random())) for i in range(N - 5)]
    ent = [RawEdge(i, i + 3, float(rng.random())) for i in range(N - 3)]
    return {"adjacency": adj, "section": sec, "semantic": sem, "entity": ent}


def _chunks(n=N):
    return [Chunk(chunk_id=f"c{i}", doc_id="d", text=f"t{i}", position=i,
                  section_path=[], references=[]) for i in range(n)]


def _total(w):
    return sum(getattr(w, n) for n in _NAMES)


def test_calibrate_sums_to_one_and_fixes_adjacency():
    pri = EdgeWeights()
    w = calibrate_edge_weights(_signals(), pri)
    assert _total(w) == pytest.approx(1.0)
    assert w.adjacency == pri.adjacency
    assert w.semantic > 0


def test_calibrate_falls_back_to_priors_when_uninformative():
    pri = EdgeWeights()
    assert calibrate_edge_weights({}, pri) is pri


def test_fit_sums_to_one_and_keeps_label_priors():
    pri = EdgeWeights()
    w = fit_edge_weights(_signals(), pri, n_chunks=N, min_positives=5)
    assert _total(w) == pytest.approx(1.0)
    assert w.adjacency == pri.adjacency and w.section == pri.section


def test_fit_falls_back_with_too_few_positives():
    pri = EdgeWeights()
    assert fit_edge_weights(_signals(), pri, n_chunks=N, min_positives=10_000) is pri
    assert fit_edge_weights(_signals(), pri, n_chunks=3) is pri


def test_fit_falls_back_when_no_negatives_available():
    # Complete graph on 5 nodes: every pair is positive, so no negatives exist.
    edges = [RawEdge(i, j, 1.0) for i in range(5) for j in range(i + 1, 5)]
    pri = EdgeWeights()
    assert fit_edge_weights({"adjacency": edges}, pri, n_chunks=5, min_positives=1) is pri


def test_multiplex_arpack_and_isolated_interpolation():
    # Tail chunks have no edges, exercising the isolated-chunk interpolation.
    sig = {k: [e for e in v if e.j < N - 5] for k, v in _signals().items()}
    chunks = _chunks()
    out = SpectralComputer(IndexingConfig()).compute_multiplex(sig, EdgeWeights(), chunks)
    assert out.shape[0] == N and np.isfinite(out).all()
    assert chunks[0].spectral_coords is not None


def test_multiplex_without_layers_raises():
    with pytest.raises(ValueError):
        SpectralComputer(IndexingConfig()).compute_multiplex({}, EdgeWeights(), _chunks(5))


def test_multiplex_lobpcg_path_and_tiny_fallback():
    cfg = IndexingConfig()
    cfg.spectral_large_graph_threshold = 10
    out = SpectralComputer(cfg).compute_multiplex(_signals(), EdgeWeights(), _chunks())
    assert out.shape[0] == N and np.isfinite(out).all()
    tiny = SpectralComputer(IndexingConfig()).compute_multiplex(
        {"semantic": [RawEdge(0, 1, 1.0)]}, EdgeWeights(), _chunks(2))
    assert tiny.shape == (2, 1)
