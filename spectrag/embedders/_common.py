"""Helpers shared by the `Embedder` implementations (ADR 0004). Not public API."""
from __future__ import annotations

from typing import Sequence

import numpy as np

KINDS = ("document", "query")


def check_inputs(texts: Sequence[str], kind: str) -> list[str]:
    """Validate `embed()` arguments the same way in every implementation.

    A bare string is rejected: iterating it would silently embed one character
    per row.
    """
    if isinstance(texts, (str, bytes)):
        raise TypeError("embed() takes a sequence of strings, not a single string")
    if kind not in KINDS:
        raise ValueError(f"kind must be 'document' or 'query', got {kind!r}")
    texts = list(texts)
    for t in texts:
        if not isinstance(t, str):
            raise TypeError(f"embed() inputs must be str, got {type(t).__name__}")
    return texts


def to_unit_rows(vectors, n_expected: int) -> np.ndarray:
    """Return `vectors` as a float32 (n, d) array with L2-normalized rows.

    All-zero rows stay zero (they have no direction). Raises `RuntimeError` on a
    malformed backend response, so a short or ragged reply can never map the
    wrong vector onto a text.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != n_expected:
        raise RuntimeError(
            f"backend returned embeddings of shape {arr.shape} for {n_expected} inputs"
        )
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (arr / norms).astype(np.float32, copy=False)
