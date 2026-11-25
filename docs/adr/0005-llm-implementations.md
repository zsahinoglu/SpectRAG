# ADR 0005: IndexLLM implementations

- Status: accepted (ARCH2b); constructor signatures provisional until the v1.0 freeze (ARCH3)
- Date: 2026-11-25

## Context

ADR 0002 defined the `IndexLLM` Protocol (`model_id`, `complete`, `complete_json`). Today every
LLM call site (`llm_concept.py`, `utility_question.py`, `generator.py`) builds its own client
from config objects and branches on a backend string (`anthropic`/`openai`/`ollama`), with its
own retry loop and its own output parsing. ARCH2 asks for Anthropic (default), Ollama and "no
LLM" implementations of the Protocol; ARCH2b delivers them and a conformance suite, and ARCH2c
wires them into the pipelines. The human dropped the OpenAI backends (ARCH2d).

## Decision

1. **`spectrag.llms` exports `AnthropicLLM`, `OllamaLLM`, `NoLLM` and `LLMUnavailableError`.**
   One module per backend; backend packages are imported inside the constructor, so
   `import spectrag.llms` imports no backend (pinned by a test).
2. **Same construction rules as embedders (ADR 0004):** the model name is the only positional
   argument, everything else is keyword-only, no `config`/`spectrag.settings` import, and
   `client=` injects a ready backend object (tests pass fakes).
3. **`model_id` is `"<provider>:<model>"`**: `anthropic:claude-haiku-5-5`, `ollama:llama3.1:8b`,
   `none:none`. Index-time caches and manifests key on it (hard rule 6).
4. **API key lookup for `AnthropicLLM`**: `api_key=`, else `SPECTRAG_ANTHROPIC_API_KEY`, else
   `ANTHROPIC_API_KEY`. The key is always passed to the SDK explicitly. A missing key fails at
   construction with a message naming the variable, not on the first call.
5. **Retries are transport-only and live in one place per backend.** Anthropic: the SDK's own
   `max_retries` (rate limits, 5xx, connection errors), not repeated by us. Ollama: tenacity,
   `max_retries` attempts with exponential backoff, also on a reply without text.
   A reply that fails JSON parsing or validation is *not* retried inside the component:
   `complete_json` raises `ValueError` and the caller decides (the pipelines already retry or
   skip a chunk). This keeps a single call's cost bounded and predictable.
6. **`complete` returns the reply text unchanged** (Anthropic: concatenated `text` blocks only).
   **`complete_json` parses and validates**: it accepts bare JSON, a fenced block, or one JSON
   value inside prose, then validates against the schema; callers never see unvalidated output.
   Ollama additionally requests `format="json"`; Anthropic relies on the prompt.
7. **A small JSON Schema subset, no new dependency.** Supported: `type` (incl. lists), `enum`,
   `const`, `properties`, `required`, `additionalProperties`, `items`, `minItems`, `maxItems`,
   `uniqueItems`, `minLength`, `maxLength`, `minimum`, `maximum`; annotations (`title`,
   `description`, ...) are ignored. Any other keyword raises `ValueError` *before* the model is
   called, so a constraint is never silently skipped. `true` is not an integer. If H7/I2 need
   more, either extend the subset or adopt `jsonschema` with its own ADR.
8. **Argument checks are shared** (`spectrag/llms/_common.py`, private `LLMBase`): `prompt` and
   `system` must be `str` (`TypeError`), `max_tokens` a positive int and `temperature` a finite
   number ≥ 0 (`ValueError`). Backend-specific upper limits are left to the backend.
9. **`NoLLM` is an explicit component, not `None`.** It conforms to the Protocol, validates
   arguments and then raises `LLMUnavailableError` (a `RuntimeError`). Pipelines (ARCH2c) test
   `isinstance(llm, NoLLM)` up front and skip LLM concepts, utility questions and answer
   generation; a missed check fails loudly instead of producing empty edge signals.
10. **Conformance suite**: `tests/conformance/llm.py` — `check_llm(make)` takes a factory that
    builds the implementation over scripted replies and checks signatures, argument and schema
    errors, pass-through text, JSON extraction and rejection of every invalid reply;
    `check_no_llm` covers `NoLLM`. Every implementation runs it.

## Consequences

- Nothing in the pipelines changes in ARCH2b; retrieval output is unchanged. ARCH2c replaces
  the backend branches and `rag_system/llm.py` helpers with injected `IndexLLM`s and removes
  the OpenAI code paths.
- Default sampling stays deterministic (`temperature=0.0`); the legacy Ollama default of 0.1
  becomes an explicit argument at the call site when ARCH2c migrates it.
- Prompt versions are not part of `model_id`; the pipelines keep pinning them with the cache
  key and manifest (H3).
- `tests/api/test_llms_surface.py` pins the exports, constructor signatures, and that the
  method signatures equal the Protocol's.
