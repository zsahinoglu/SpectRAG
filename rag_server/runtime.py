"""On-disk layout for the server, as a class.

Every session owns its documents and index under ``<store_dir>/sessions/<id>/``,
so reopening a session (or restarting the server) restores everything it had:

    sessions.json                     session metadata (name, messages, …)
    sessions/<id>/docs/               the session's uploaded source files
    sessions/<id>/manifest.json       safe index format (see spectrag.formats)
    sessions/<id>/chunks.parquet, embeddings.npy, spectral.npy, faiss.index
    sessions/<id>/graph/              that session's chunk graph (own manifest)

Legacy ``chunks.pkl`` / ``graph.pkl`` are never loaded; see spectrag.formats.migrate.
"""
from __future__ import annotations

from pathlib import Path

from spectrag.formats import MANIFEST_FILE


class ServerPaths:
    """Resolves every on-disk path the server uses, from a single store dir."""

    GRAPH_FILE = "graph"            # a directory
    CHUNKS_FILE = MANIFEST_FILE     # an index exists once its manifest is written

    def __init__(self, store_dir: Path | str) -> None:
        self.store_dir = Path(store_dir)
        self.sessions_dir = self.store_dir / "sessions"
        self.sessions_file = self.store_dir / "sessions.json"

    def session_dir(self, sid: str) -> Path:
        return self.sessions_dir / sid

    def docs_dir(self, sid: str) -> Path:
        return self.session_dir(sid) / "docs"

    def graph_file(self, sid: str) -> Path:
        return self.session_dir(sid) / self.GRAPH_FILE

    def has_index(self, sid: str) -> bool:
        """True once a session has been indexed (its chunk store exists on disk)."""
        return (self.session_dir(sid) / self.CHUNKS_FILE).exists()
