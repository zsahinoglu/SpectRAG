"""Deprecated location: use `spectrag.models` (ADR 0003).

Re-export shim. Internal callers were migrated in ARCH2c, so importing this
module now emits a `DeprecationWarning`. Removed no earlier than v1.0.
"""
import warnings

from spectrag.models import Chunk, QueryResult

warnings.warn(
    "rag_system.models is deprecated; import Chunk and QueryResult from spectrag.models "
    "(or spectrag) instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Chunk", "QueryResult"]
