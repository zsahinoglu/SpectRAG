"""ARCH2b: IndexLLM implementations conform to `spectrag.IndexLLM`.

No network and no API keys: Anthropic and Ollama are replaced by fake clients
injected through `client=`.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from tenacity import wait_none

from spectrag import IndexLLM
from spectrag.llms import AnthropicLLM, LLMUnavailableError, NoLLM, OllamaLLM
from spectrag.llms._common import parse_json_reply, validate
from tests.conformance.llm import check_llm, check_no_llm


class FakeAnthropicClient:
    """Mimics `anthropic.Anthropic().messages.create`; replies are queued."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=r)])


class FakeOllamaClient:
    """Mimics `ollama.Client().generate`; replies are queued."""

    def __init__(self, replies=(), as_dict=False):
        self.replies = list(replies)
        self.as_dict = as_dict
        self.calls: list[dict] = []

    def generate(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"response": r} if self.as_dict else SimpleNamespace(response=r)


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(OllamaLLM, "_wait", wait_none())


@pytest.fixture
def no_env_keys(monkeypatch):
    monkeypatch.delenv("SPECTRAG_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# --- conformance -------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda r: AnthropicLLM(client=FakeAnthropicClient(r)),
    lambda r: OllamaLLM(client=FakeOllamaClient(r)),
    lambda r: OllamaLLM("qwen2.5:7b", client=FakeOllamaClient(r, as_dict=True)),
], ids=["anthropic", "ollama", "ollama-dict"])
def test_conformance(make):
    check_llm(make)


def test_no_llm_conformance():
    check_no_llm(NoLLM())
    assert isinstance(NoLLM(), IndexLLM)
    assert issubclass(LLMUnavailableError, RuntimeError)


def test_model_ids():
    assert AnthropicLLM(client=FakeAnthropicClient()).model_id == "anthropic:claude-haiku-5-5"
    assert OllamaLLM(client=FakeOllamaClient()).model_id == "ollama:llama3.1:8b"
    assert NoLLM().model_id == "none:none"


# --- Anthropic specifics ---------------------------------------------------------

def test_anthropic_request_shape():
    fake = FakeAnthropicClient(["a", "b"])
    llm = AnthropicLLM("claude-x", client=fake)
    llm.complete("hi")
    llm.complete("hi", system="sys", max_tokens=7, temperature=0.5)
    assert fake.calls[0] == {
        "model": "claude-x", "max_tokens": 1024, "temperature": 0.0,
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert fake.calls[1]["system"] == "sys"
    assert fake.calls[1]["max_tokens"] == 7 and fake.calls[1]["temperature"] == 0.5


def test_anthropic_joins_text_blocks_only():
    fake = FakeAnthropicClient()
    fake.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[
        SimpleNamespace(type="thinking", thinking="hmm"),
        SimpleNamespace(type="text", text="Hel"),
        SimpleNamespace(type="text", text="lo"),
    ]))
    assert AnthropicLLM(client=fake).complete("p") == "Hello"


def test_anthropic_backend_errors_propagate():
    fake = FakeAnthropicClient([RuntimeError("overloaded")])
    with pytest.raises(RuntimeError, match="overloaded"):
        AnthropicLLM(client=fake).complete("p")
    assert len(fake.calls) == 1, "retries are the SDK's job, not repeated here"


def test_anthropic_missing_key_fails_at_construction(no_env_keys):
    with pytest.raises(ValueError, match="SPECTRAG_ANTHROPIC_API_KEY"):
        AnthropicLLM()


class _RecordingAnthropicModule:
    def __init__(self):
        self.kwargs = None

    def Anthropic(self, **kw):  # noqa: N802 - mimics the SDK class
        self.kwargs = kw
        return FakeAnthropicClient()


@pytest.mark.parametrize("env, arg, expected", [
    ({"SPECTRAG_ANTHROPIC_API_KEY": "spec", "ANTHROPIC_API_KEY": "plain"}, None, "spec"),
    ({"ANTHROPIC_API_KEY": "plain"}, None, "plain"),
    ({"SPECTRAG_ANTHROPIC_API_KEY": "spec"}, "explicit", "explicit"),
])
def test_anthropic_key_lookup_order(monkeypatch, no_env_keys, env, arg, expected):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mod = _RecordingAnthropicModule()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    AnthropicLLM(api_key=arg, timeout=9.0, max_retries=4, base_url="http://proxy")
    assert mod.kwargs == {
        "api_key": expected, "timeout": 9.0, "max_retries": 4, "base_url": "http://proxy",
    }


# --- Ollama specifics ----------------------------------------------------------

def test_ollama_request_shape():
    fake = FakeOllamaClient(["x", '{"a": 1}'])
    llm = OllamaLLM("m", client=fake)
    llm.complete("p", max_tokens=9, temperature=0.2)
    llm.complete_json("q", {"type": "object"}, system="json")
    assert fake.calls[0] == {
        "model": "m", "prompt": "p", "options": {"temperature": 0.2, "num_predict": 9},
    }
    assert fake.calls[1]["format"] == "json" and fake.calls[1]["system"] == "json"
    assert "format" not in fake.calls[0] and "system" not in fake.calls[0]


def test_ollama_retries_transport_errors_then_succeeds():
    # None -> a reply without response text, which is retried like a transport error
    fake = FakeOllamaClient([ConnectionError("down"), None, "ok"])
    assert OllamaLLM(client=fake).complete("p") == "ok"
    assert len(fake.calls) == 3


def test_ollama_gives_up_after_max_retries():
    fake = FakeOllamaClient([ConnectionError("1"), ConnectionError("2"), "never"])
    with pytest.raises(ConnectionError, match="2"):
        OllamaLLM(client=fake, max_retries=2).complete("p")
    assert len(fake.calls) == 2


def test_ollama_invalid_json_is_not_retried():
    fake = FakeOllamaClient(["nope", '{"a": 1}'])
    with pytest.raises(ValueError):
        OllamaLLM(client=fake).complete_json("p", {"type": "object"})
    assert len(fake.calls) == 1, "validation failures are the caller's decision"


def test_ollama_rejects_zero_attempts():
    with pytest.raises(ValueError):
        OllamaLLM(client=FakeOllamaClient(), max_retries=0)


# --- NoLLM -------------------------------------------------------------------------

def test_no_llm_message_names_alternatives():
    with pytest.raises(LLMUnavailableError, match="AnthropicLLM or OllamaLLM"):
        NoLLM().complete("p")


# --- JSON helpers --------------------------------------------------------------------

def test_parse_json_reply_skips_unbalanced_brace_in_prose():
    assert parse_json_reply('Using {braces} loosely: {"a": [1]}') == {"a": [1]}


@pytest.mark.parametrize("value, schema", [
    ("x", {"enum": ["a", "b"]}),
    ("x", {"const": "y"}),
    ("", {"type": "string", "minLength": 1}),
    ("long", {"type": "string", "maxLength": 3}),
    (5, {"type": "number", "maximum": 4}),
    ([1, 1], {"type": "array", "uniqueItems": True}),
    ([], {"type": "array", "minItems": 1}),
    ({"k": "v"}, {"type": "object", "additionalProperties": {"type": "integer"}}),
    (None, {"type": ["string", "integer"]}),
])
def test_validate_rejects(value, schema):
    with pytest.raises(ValueError):
        validate(value, schema)


@pytest.mark.parametrize("value, schema", [
    (None, {"type": ["string", "null"]}),
    (3.0, {"type": "integer"}),
    ({"k": 2}, {"type": "object", "additionalProperties": {"type": "integer"}}),
    ("a", {"type": "string", "title": "annotations are ignored", "description": "x"}),
])
def test_validate_accepts(value, schema):
    validate(value, schema)
