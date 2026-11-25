"""Behavioral conformance suite for `spectrag.Embedder` (ADR 0002, ADR 0004).

`isinstance(x, Embedder)` only checks member names. `check_embedder` checks the
contract every implementation must keep, so the pipelines can treat embedders
as interchangeable. Call it on any implementation (with a fake backend in unit
tests); it raises `AssertionError` with a message on the first violation.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectrag.protocols import Embedder

TEXTS = ["Spectral graphs cluster chunks.", "Diffusion spreads relevance.", "Ünïcödé ✓ text"]


def check_embedder(emb) -> None:
    assert isinstance(emb, Embedder), "missing Embedder members"
    assert isinstance(emb.model_id, str) and ":" in emb.model_id, (
        "model_id must be '<provider>:<model>' (ADR 0004)"
    )
    dim = emb.dimension
    assert isinstance(dim, int) and dim > 0

    for kind in ("document", "query"):
        out = emb.embed(TEXTS, kind=kind)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32, f"dtype {out.dtype} for kind={kind}"
        assert out.shape == (len(TEXTS), dim), f"shape {out.shape} for kind={kind}"
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    # default kind is "document"
    np.testing.assert_allclose(emb.embed(TEXTS), emb.embed(TEXTS, kind="document"), atol=1e-6)
    # input order is preserved and the result does not depend on batch composition
    single = np.vstack([emb.embed([t]) for t in TEXTS])
    np.testing.assert_allclose(emb.embed(TEXTS), single, atol=1e-5)
    np.testing.assert_allclose(emb.embed(TEXTS[::-1]), single[::-1], atol=1e-5)
    # any sequence is accepted, not only lists
    np.testing.assert_allclose(emb.embed(tuple(TEXTS)), single, atol=1e-5)

    empty = emb.embed([])
    assert empty.shape == (0, dim) and empty.dtype == np.float32

    with pytest.raises(TypeError):
        emb.embed("a single string is a common bug")
    with pytest.raises(ValueError):
        emb.embed(TEXTS, kind="passage")
    assert emb.dimension == dim, "dimension must be stable"
