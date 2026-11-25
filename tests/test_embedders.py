"""ARCH2a: embedder implementations conform to `spectrag.Embedder`.

No network and no models: Ollama and sentence-transformers are replaced by
deterministic fake backends injected through `client=`.
"""
from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from tenacity import wait_none

from spectrag import Embedder
from spectrag.embedders import OllamaEmbedder, SentenceTransformerEmbedder
from tests.conformance.embedder import check_embedder
from tests.fakes import FakeEmbedder


def _raw_vec(text: str, dim: int) -> list[float]:
    """Deterministic, unnormalized, text-dependent vector."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return [float(b) - 100.0 for b in (h * (dim // len(h) + 1))[:dim]]


class FakeOllamaClient:
    def __init__(self, dim: int = 8, short_replies: int = 0) -> None:
        self.dim = dim
        self.calls: list[tuple[str, list[str]]] = []
        self.short_replies = short_replies  # first N calls return one vector too few

    def embed(self, model, input):
        self.calls.append((model, list(input)))
        vecs = [_raw_vec(t, self.dim) for t in input]
        if self.short_replies:
            self.short_replies -= 1
            vecs = vecs[:-1]
        return SimpleNamespace(embeddings=vecs)


class FakeSentenceTransformer:
    def __init__(self, dim: int | None = 6) -> None:
        self.dim = dim
        self.calls: list[dict] = []

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        arr = np.array([_raw_vec(t, self.dim or 5) for t in texts], dtype=np.float64)
        if kwargs.get("normalize_embeddings"):
            arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(OllamaEmbedder._embed_batch.retry, "wait", wait_none())


# --- conformance -------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: OllamaEmbedder(client=FakeOllamaClient(), batch_size=2),
    lambda: OllamaEmbedder("mxbai-embed-large", client=FakeOllamaClient(dim=16)),
    lambda: SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", client=FakeSentenceTransformer()),
    lambda: SentenceTransformerEmbedder("m", client=FakeSentenceTransformer(dim=None)),
    lambda: FakeEmbedder(dim=32),
], ids=["ollama-nomic", "ollama-other", "st", "st-unknown-dim", "fake"])
def test_conformance(make):
    check_embedder(make())


def test_legacy_rag_system_embedder_conforms():
    from config import OllamaConfig
    from rag_system.indexing.embedder import OllamaEmbedder as Legacy

    emb = Legacy(OllamaConfig(embedding_dim=8))
    emb._client = FakeOllamaClient(dim=8)
    assert isinstance(emb, OllamaEmbedder) and isinstance(emb, Embedder)
    assert emb.model_id == "ollama:nomic-embed-text"
    check_embedder(emb)


# --- Ollama specifics ----------------------------------------------------------

def test_ollama_nomic_prefixes_and_batching():
    client = FakeOllamaClient()
    emb = OllamaEmbedder("nomic-embed-text", client=client, batch_size=2)
    emb.embed(["a", "b", "c"], kind="document")
    emb.embed(["q"], kind="query")
    assert client.calls == [
        ("nomic-embed-text", ["search_document: a", "search_document: b"]),
        ("nomic-embed-text", ["search_document: c"]),
        ("nomic-embed-text", ["search_query: q"]),
    ]


def test_ollama_prefixes_overridable_and_absent_for_other_models():
    client = FakeOllamaClient()
    OllamaEmbedder("mxbai-embed-large", client=client).embed(["a"], kind="query")
    OllamaEmbedder("nomic-embed-text", client=client, query_prefix="").embed(["b"], kind="query")
    OllamaEmbedder("x", client=client, document_prefix="passage: ").embed(["c"])
    assert [c[1] for c in client.calls] == [["a"], ["b"], ["passage: c"]]


def test_ollama_matches_previous_normalization():
    client = FakeOllamaClient(dim=8)
    out = OllamaEmbedder("plain", client=client).embed(["x", "y"])
    raw = np.array([_raw_vec("x", 8), _raw_vec("y", 8)], dtype=np.float32)
    np.testing.assert_allclose(out, raw / np.linalg.norm(raw, axis=1, keepdims=True), atol=1e-6)


def test_ollama_short_reply_is_retried_then_succeeds():
    client = FakeOllamaClient(short_replies=2)
    out = OllamaEmbedder(client=client).embed(["a", "b"])
    assert out.shape == (2, 8) and len(client.calls) == 3


def test_ollama_short_reply_fails_after_retries():
    client = FakeOllamaClient(short_replies=5)
    with pytest.raises(RuntimeError, match="for 2 inputs"):
        OllamaEmbedder(client=client).embed(["a", "b"])
    assert len(client.calls) == 3


def test_ollama_dimension_mismatch_warns_and_uses_model(caplog):
    emb = OllamaEmbedder(client=FakeOllamaClient(dim=8), dimension=768)
    with caplog.at_level(logging.WARNING):
        emb.embed(["a"])
    assert emb.dimension == 8
    assert "differs from the expected 768" in caplog.text


def test_ollama_dimension_drift_is_an_error():
    client = FakeOllamaClient(dim=8)
    emb = OllamaEmbedder(client=client)
    emb.embed(["a"])
    client.dim = 4
    with pytest.raises(RuntimeError, match="changed mid-run"):
        emb.embed(["b"])


def test_ollama_dimension_probe_is_cached():
    client = FakeOllamaClient(dim=8)
    emb = OllamaEmbedder(client=client)
    assert emb.dimension == 8 and emb.dimension == 8
    assert len(client.calls) == 1


def test_ollama_accepts_dict_responses():
    class DictClient:
        def embed(self, model, input):
            return {"embeddings": [[1.0, 0.0], [0.0, 2.0]]}

    out = OllamaEmbedder(client=DictClient()).embed(["a", "b"])
    np.testing.assert_allclose(out, [[1, 0], [0, 1]])


def test_ollama_rejects_bad_batch_size():
    with pytest.raises(ValueError):
        OllamaEmbedder(client=FakeOllamaClient(), batch_size=0)


# --- sentence-transformers specifics -----------------------------------------

def test_st_model_id_pins_revision():
    fake = FakeSentenceTransformer()
    assert SentenceTransformerEmbedder("org/m", client=fake).model_id == "sentence-transformers:org/m"
    assert (SentenceTransformerEmbedder("org/m", revision="abc123", client=fake).model_id
            == "sentence-transformers:org/m@abc123")


def test_st_prefixes_and_encode_arguments():
    fake = FakeSentenceTransformer()
    emb = SentenceTransformerEmbedder("m", client=fake, batch_size=7,
                                      query_prefix="query: ", document_prefix="passage: ")
    emb.embed(["a"], kind="query")
    emb.embed(["b"])
    assert [c["texts"] for c in fake.calls] == [["query: a"], ["passage: b"]]
    assert fake.calls[0]["batch_size"] == 7
    assert fake.calls[0]["normalize_embeddings"] is True


def test_st_dimension_from_model_needs_no_encode():
    fake = FakeSentenceTransformer(dim=6)
    emb = SentenceTransformerEmbedder("m", client=fake)
    assert emb.dimension == 6
    assert emb.embed([]).shape == (0, 6)
    assert fake.calls == []
