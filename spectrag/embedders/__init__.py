"""`Embedder` implementations (ADR 0002, ADR 0004).

    OllamaEmbedder               local Ollama server       extra: spectrag[ollama]
    SentenceTransformerEmbedder  in-process model          extra: spectrag[rerank]

Importing this package does not import any backend; a missing extra fails when
the embedder is constructed, with the `pip install` line to fix it. The OpenAI
embedder is tracked separately (ARCH2d).
"""
from __future__ import annotations

from spectrag.embedders.ollama import OllamaEmbedder
from spectrag.embedders.sentence_transformer import SentenceTransformerEmbedder

__all__ = ["OllamaEmbedder", "SentenceTransformerEmbedder"]
