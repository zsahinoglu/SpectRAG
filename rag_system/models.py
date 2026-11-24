"""Deprecated location: use `spectrag.models` (ADR 0003).

Re-export shim. It stays silent while code inside this repository still
imports from here; it starts emitting `DeprecationWarning` once internal
callers are migrated (ARCH2). Indexes pickled before ARCH1 reference
`rag_system.models.Chunk` and keep loading through this module.
"""
from spectrag.models import Chunk, QueryResult

__all__ = ["Chunk", "QueryResult"]
