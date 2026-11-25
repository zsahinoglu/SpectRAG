"""ARCH2c: pipelines take Embedder and IndexLLM by injection (ADR 0006). Offline."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from config import Config
from rag_system import components
from rag_system.generation.generator import Generator
from rag_system.indexing import chunker as chunker_mod
from rag_system.indexing.edges.entity import EntityEdgeExtractor
from rag_system.indexing.embedder import build_faiss_index, embed_chunks
from rag_system.indexing.pipeline import IndexingPipeline
from rag_system.query.pipeline import QueryPipeline
from rag_system.store.memory_graph import InMemoryGraphStore
from spectrag.llms import AnthropicLLM, LLMUnavailableError, NoLLM, OllamaLLM
from spectrag.models import Chunk
from tests.fakes import FakeEmbedder
from tests.golden_corpus import QUESTIONS, build_index, offline_encoder
from tests.test_golden_retrieval import GOLDEN

LLM_CALL_SITES = [
    "rag_system/generation/generator.py",
    "rag_system/indexing/edges/llm_concept.py",
    "rag_system/indexing/edges/utility_question.py",
    "rag_system/indexing/edges/entity.py",
    "rag_system/indexing/pipeline.py",
    "rag_system/query/pipeline.py",
]
ROOT = Path(__file__).resolve().parent.parent


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    def embed(self, texts, kind="document"):
        self.texts += list(texts)
        return super().embed(texts, kind)


class RecordingLLM:
    """A minimal third-party IndexLLM: not a spectrag class, conforms structurally."""
    model_id = "test:recorder"

    def __init__(self, reply="What is it?"):
        self.reply, self.calls = reply, []

    def complete(self, prompt, *, system=None, max_tokens=1024, temperature=0.0):
        self.calls.append(("complete", prompt))
        return self.reply

    def complete_json(self, prompt, schema, *, system=None, max_tokens=1024, temperature=0.0):
        self.calls.append(("json", prompt))
        return {"concepts": ["shared idea"]}


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"c{i}", doc_id="d", text=text, position=i,
                 section_path=[], references=[])


# ---- factory: the only place backend strings are resolved --------------------------

def test_factory_none_backend():
    assert isinstance(components.build_llm(Config(), "none"), NoLLM)


def test_factory_anthropic_backend():
    pytest.importorskip("anthropic")
    cfg = Config()
    cfg.anthropic.api_key = "k"
    llm = components.build_llm(cfg, "anthropic")
    assert isinstance(llm, AnthropicLLM) and llm.model_id == "anthropic:claude-haiku-5-5"


def test_factory_ollama_backend():
    pytest.importorskip("ollama")
    llm = components.build_llm(Config(), "ollama")
    assert isinstance(llm, OllamaLLM) and llm.model_id == "ollama:llama3.1:8b"


@pytest.mark.parametrize("name", ["openai", "gpt", ""])
def test_factory_rejects_openai_and_unknown(name):
    with pytest.raises(ValueError, match="OpenAI backend was removed" if name == "openai"
                       else "Unknown LLM backend"):
        components.build_llm(Config(), name)


def test_factory_missing_anthropic_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("SPECTRAG_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = Config()
    cfg.anthropic.api_key = ""
    with pytest.raises(ValueError, match="SPECTRAG_ANTHROPIC_API_KEY"):
        components.build_llm(cfg, "anthropic")


def test_generation_sampling_follows_backend():
    cfg = Config()
    assert components.generation_sampling(cfg) == (cfg.anthropic.temperature,
                                                   cfg.anthropic.max_tokens)
    cfg.generation_backend = "ollama"
    assert components.generation_sampling(cfg) == (cfg.ollama.llm_temperature,
                                                   cfg.ollama.llm_max_tokens)


def test_no_backend_branches_or_openai_left_at_call_sites():
    for rel in LLM_CALL_SITES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "openai" not in src.lower(), rel
        assert "_backend ==" not in src and "import ollama" not in src, rel
        assert "rag_system.llm import" not in src, rel


def test_config_has_no_openai_section():
    import dataclasses
    assert "openai" not in {f.name for f in dataclasses.fields(Config)}


# ---- call sites take components -----------------------------------------------------

def test_pipeline_constructors_take_components():
    for cls, params in [(IndexingPipeline, ("embedder", "llm")),
                        (QueryPipeline, ("embedder", "llm"))]:
        sig = inspect.signature(cls.__init__)
        for p in params:
            assert sig.parameters[p].kind is inspect.Parameter.KEYWORD_ONLY
            assert sig.parameters[p].default is None
    assert list(inspect.signature(Generator.__init__).parameters) == ["self", "cfg", "llm"]


def test_query_pipeline_with_injected_components_matches_golden():
    """No Ollama and no API key needed once both components are injected; retrieval
    output is identical to the pinned golden results."""
    cfg, emb, chunks, fidx, store = build_index()
    p = QueryPipeline(cfg, embedder=emb, llm=NoLLM())
    assert p.embedder is emb and p.seed_retriever.embedder is emb
    p.selector._enc = offline_encoder()
    golden = json.loads(GOLDEN.read_text())
    for q in QUESTIONS[:2]:
        res = p.query(q, chunks, fidx, store)
        assert res.answer == ""                      # NoLLM → retrieval only
        assert [c.chunk_id for c in res.chunks] == golden[q]["selected"]


def test_generator_uses_injected_llm_and_skips_with_nollm():
    llm = RecordingLLM(reply="  An answer. ")
    assert Generator(Config(), llm).generate("What is it?", [_chunk(0, "Text.")]) == "An answer."
    assert llm.calls[0][0] == "complete" and "Text." in llm.calls[0][1]
    assert Generator(Config(), NoLLM()).generate("What is it?", [_chunk(0, "Text.")]) == ""


def test_entity_edges_with_injected_llm(tmp_path):
    cfg = Config()
    cfg.indexing.concept_cache_dir = str(tmp_path / "c")
    llm = RecordingLLM()
    edges = EntityEdgeExtractor(cfg.indexing, cfg.ner, llm=llm).extract(
        [_chunk(0, "a"), _chunk(1, "b")])
    assert [(e.i, e.j) for e in edges] == [(0, 1)]
    assert {kind for kind, _ in llm.calls} == {"json"}
    assert (tmp_path / "c" / "test_recorder.json").exists()


def test_entity_edges_skip_with_nollm_and_require_an_llm(tmp_path):
    cfg = Config()
    cfg.indexing.concept_cache_dir = str(tmp_path / "c")
    assert EntityEdgeExtractor(cfg.indexing, cfg.ner, llm=NoLLM()).extract(
        [_chunk(0, "a"), _chunk(1, "b")]) == []
    with pytest.raises(TypeError, match="llm="):
        EntityEdgeExtractor(cfg.indexing, cfg.ner)


def test_nollm_reaching_a_call_site_fails_loudly(tmp_path):
    from rag_system.indexing.edges.llm_concept import LLMConceptExtractor
    cfg = Config()
    cfg.indexing.concept_cache_dir = str(tmp_path / "c")
    with pytest.raises(LLMUnavailableError):
        LLMConceptExtractor(NoLLM(), cfg.indexing).extract("text")


def test_utility_questions_sequential_only_for_ollama(tmp_path):
    from rag_system.indexing.edges.utility_question import UtilityQuestionEdgeExtractor
    cfg = Config()
    cfg.indexing.utility_question_cache_dir = str(tmp_path / "u")

    class _Gen:
        def generate(self, **kw):
            raise AssertionError("not called")

    assert UtilityQuestionEdgeExtractor(cfg.indexing, RecordingLLM(), None)._workers == \
        cfg.indexing.llm_parallel_workers
    ollama = OllamaLLM(client=_Gen())
    assert UtilityQuestionEdgeExtractor(cfg.indexing, ollama, None)._workers == 1


# ---- embedding helpers work with any Embedder ---------------------------------------

def test_embed_chunks_caches_by_model_id(tmp_path):
    emb = CountingEmbedder()
    chunks = [_chunk(0, "red star"), _chunk(1, "blue planet")]
    embed_chunks(emb, chunks, str(tmp_path))
    assert (tmp_path / "fake_hashed-bow-64.npz").exists()
    again = [_chunk(0, "red star"), _chunk(1, "blue planet"), _chunk(2, "new text")]
    embed_chunks(emb, again, str(tmp_path))
    assert emb.texts == ["red star", "blue planet", "new text"]   # only the new one re-embedded
    np.testing.assert_array_equal(again[0].embedding, chunks[0].embedding)
    idx = build_faiss_index(again, emb.dimension)
    assert idx.ntotal == 3 and idx.d == emb.dimension


def test_embed_chunks_reads_pre_arch2c_cache(tmp_path):
    from rag_system.indexing.embedder import _EmbeddingCache, _text_hash
    old = _EmbeddingCache(str(tmp_path), "hashed-bow-64")   # legacy name: bare model
    vec = np.ones(64, dtype=np.float32) / 8.0
    old.put(_text_hash("cached text"), vec)
    old.flush()
    emb = CountingEmbedder()
    c = _chunk(0, "cached text")
    embed_chunks(emb, [c], str(tmp_path))
    assert emb.texts == [] and np.array_equal(c.embedding, vec)


def test_build_faiss_index_rejects_unembedded():
    with pytest.raises(ValueError, match="not embedded"):
        build_faiss_index([_chunk(0, "x")], 4)
    assert build_faiss_index([], 4).d == 4


# ---- full indexing pass with injected components ------------------------------------

class _WordEncoder:
    def encode(self, text, **kw):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def test_indexing_pipeline_end_to_end_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(chunker_mod, "_get_encoder", lambda name: _WordEncoder())
    docs = []
    for name, topic in [("stars", "Stars fuse hydrogen into helium in the stellar core."),
                        ("bread", "Bread dough rises because yeast ferments the sugar.")]:
        p = tmp_path / f"{name}.md"
        p.write_text(f"# {name}\n\n" + "\n\n".join(
            f"## Part {i}\n\n{topic} Section {i} adds detail number {i}." for i in range(4)))
        docs.append(p)
    cfg = Config()
    cfg.store_path = tmp_path / "index"
    cfg.ollama.embedding_cache_dir = str(tmp_path / "emb")
    cfg.indexing.concept_cache_dir = str(tmp_path / "c")
    cfg.indexing.utility_question_cache_dir = str(tmp_path / "u")
    emb, llm = CountingEmbedder(), RecordingLLM()
    store = InMemoryGraphStore()
    pipe = IndexingPipeline(cfg, store, embedder=emb, llm=llm)
    assert pipe.embedder is emb
    pipe.index(docs)
    assert emb.texts                                           # injected embedder used
    assert {k for k, _ in llm.calls} == {"json", "complete"}   # concepts + questions
    assert (tmp_path / "emb" / "fake_hashed-bow-64.npz").exists()

    # NoLLM: both LLM-derived signals are skipped, indexing still completes.
    cfg.store_path = tmp_path / "index2"
    IndexingPipeline(cfg, InMemoryGraphStore(), embedder=CountingEmbedder(),
                     llm=NoLLM()).index(docs)
