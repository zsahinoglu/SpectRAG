"""Configuration for SpectRAG.

New code imports settings from here. During Phase 1 the dataclasses are still
defined in the top-level `config` module (builders are actively changing their
defaults); this module re-exports them so the `spectrag.settings` import path is
stable now and the definitions can move here later without touching callers
(ADR 0003). Presets (I4) will also live under `spectrag.settings`.

Note: importing this module imports `config`, which loads a `.env` file next to
it. That side effect is removed when the definitions move here.
"""
from __future__ import annotations

from config import (
    Config,
    EdgeWeights,
    IndexingConfig,
    LanguageConfig,
    Neo4jConfig,
    NERConfig,
    OllamaConfig,
    RetrievalConfig,
    ServerConfig,
)

__all__ = [
    "Config",
    "EdgeWeights",
    "IndexingConfig",
    "LanguageConfig",
    "Neo4jConfig",
    "NERConfig",
    "OllamaConfig",
    "RetrievalConfig",
    "ServerConfig",
]
