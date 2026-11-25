"""C1: Claude Haiku backend — all LLM traffic goes through a mocked client."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from config import AnthropicConfig, Config, IndexingConfig
from rag_system.generation.generator import Generator
from rag_system.indexing.edges.llm_concept import LLMConceptExtractor
from rag_system.indexing.edges.utility_question import UtilityQuestionEdgeExtractor
from rag_system.llm import make_anthropic_client, parse_concepts_json
from spectrag.llms import AnthropicLLM
from spectrag.models import Chunk


class FakeAnthropic:
    """Mimics anthropic.Anthropic().messages.create; replies are queued."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=r)])


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


@pytest.fixture
def icfg(tmp_path):
    return IndexingConfig(concept_cache_dir=str(tmp_path / "c"),
                          utility_question_cache_dir=str(tmp_path / "u"))


def test_defaults_are_anthropic_and_pinned():
    cfg = Config()
    assert IndexingConfig().utility_question_llm_backend == "anthropic"
    assert IndexingConfig().concept_llm_backend == "anthropic"
    assert cfg.generation_backend == "anthropic"
    assert cfg.anthropic.model == "claude-haiku-5-5"


def test_key_precedence(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fallback")
    monkeypatch.delenv("SPECTRAG_ANTHROPIC_API_KEY", raising=False)
    assert AnthropicConfig().api_key == "fallback"
    monkeypatch.setenv("SPECTRAG_ANTHROPIC_API_KEY", "primary")
    assert AnthropicConfig().api_key == "primary"


def test_missing_key_is_clear_error(monkeypatch):
    monkeypatch.delenv("SPECTRAG_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SPECTRAG_ANTHROPIC_API_KEY"):
        make_anthropic_client(AnthropicConfig())


def test_parse_concepts_schema():
    assert parse_concepts_json('{"concepts": ["B ", "a", "a"]}') == ["a", "b"]
    assert parse_concepts_json('```json\n{"concepts": ["x"]}\n```') == ["x"]
    assert parse_concepts_json('Sure: {"concepts": ["x"]}') == ["x"]
    for bad in ["nope", '{"concepts": "x"}', '{"concepts": [1]}', "[]", '{"other": []}']:
        with pytest.raises(ValueError):
            parse_concepts_json(bad)


def test_concepts_cache_revalidation_and_no_failure_caching(icfg):
    # An invalid reply is asked again (ADR 0005: the caller retries validation failures).
    fake = FakeAnthropic(["not json", '{"concepts": ["Graph"]}'])
    ex = LLMConceptExtractor(AnthropicLLM(client=fake), icfg)
    assert ex.extract("some text") == ["graph"]
    assert len(fake.calls) == 2
    assert fake.calls[0]["model"] == "claude-haiku-5-5"
    assert ex.extract("some text") == ["graph"]      # in-memory cache hit
    assert len(fake.calls) == 2
    ex.close()
    assert ex._cache_file.name == "anthropic_claude-haiku-5-5.json"
    # disk cache survives a new instance
    fake2 = FakeAnthropic([])
    ex2 = LLMConceptExtractor(AnthropicLLM(client=fake2), icfg)
    assert ex2.extract("some text") == ["graph"] and not fake2.calls

    # persistently invalid → [] after llm_max_retries asks, NOT cached
    bad = FakeAnthropic(["x"] * icfg.llm_max_retries)
    ex3 = LLMConceptExtractor(AnthropicLLM(client=bad), icfg)
    assert ex3.extract("other text") == []
    assert len(bad.calls) == icfg.llm_max_retries
    assert "other text" not in str(ex3._cache)

    # transport errors are retried inside the IndexLLM (SDK), not again here
    down = FakeAnthropic([RuntimeError("529")])
    assert LLMConceptExtractor(AnthropicLLM(client=down), icfg).extract("third") == []
    assert len(down.calls) == 1


def test_concepts_read_pre_arch2c_cache_file(icfg, tmp_path):
    import hashlib
    key = hashlib.sha256(b"old text").hexdigest()[:16]
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "claude-haiku-5-5.json").write_text(json.dumps({key: ["kept"]}))
    fake = FakeAnthropic([])
    assert LLMConceptExtractor(AnthropicLLM(client=fake), icfg).extract("old text") == ["kept"]
    assert not fake.calls


def test_utility_questions_anthropic(icfg):
    fake = FakeAnthropic(["1. What is A?\n2. Why B?\n3. extra"])
    uq = UtilityQuestionEdgeExtractor(icfg, AnthropicLLM(client=fake), None)
    qs = uq._generate_questions("chunk text")
    assert qs == ["What is A?", "Why B?"][: icfg.utility_questions_per_chunk]
    assert len(fake.calls) == 1
    assert uq._generate_questions("chunk text") == qs   # cache hit, no new call
    assert len(fake.calls) == 1
    # Same file name as before ARCH2c, so existing caches stay valid.
    assert uq._cache_file.name == f"anthropic_claude-haiku-5-5_{icfg.utility_questions_per_chunk}q.json"

    down = FakeAnthropic([RuntimeError("500"), "What is C?"])
    uq2 = UtilityQuestionEdgeExtractor(icfg, AnthropicLLM(client=down), None)
    assert uq2._generate_questions("other chunk") == []             # failure → no questions
    assert uq2._generate_questions("other chunk") == ["What is C?"]  # ...and not cached
    assert len(down.calls) == 2


def test_generator_anthropic(monkeypatch):
    cfg = Config()
    fake = FakeAnthropic(["  The answer.  "])
    gen = Generator(cfg, AnthropicLLM(client=fake))
    chunk = Chunk(chunk_id="c1", doc_id="d", text="Paris is the capital of France.",
                  position=0, section_path=[], references=[],
                  metadata={"source": "geo.txt"})
    assert gen.generate("What is the capital of France?", [chunk]) == "The answer."
    call = fake.calls[0]
    assert call["model"] == "claude-haiku-5-5" and "system" in call
    assert "Paris is the capital" in call["messages"][0]["content"]
