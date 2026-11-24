# Evaluation harness (`eval/`)

Owner: evaluator. Backlog: E0 (E0a = harness, E0b = paper reproduction).

## Levels

| Command | Data | LLM calls | Who |
| --- | --- | --- | --- |
| `python -m eval smoke` | bundled fixture `eval/fixtures/smoke/` | none (fake embedder, extractive generator, lexical fake judge) | anyone, CI |
| `python -m eval run --dev [--datasets ...] [--limit N]` | `$SPECTRAG_DEV_DIR` (default `~/spectrag-data/eval-dev/`) | Claude Haiku `claude-haiku-5-5` | anyone may request; evaluator runs |
| `python -m eval run --release` | `$SPECTRAG_TEST_DIR` (default `~/spectrag-private/eval-test/`), all six datasets required, no `--limit`, no `--datasets` | Claude Haiku `claude-haiku-5-5` | evaluator only |

`run` needs `SPECTRAG_ANTHROPIC_API_KEY`; the harness deliberately does **not** fall back to
`ANTHROPIC_API_KEY`. Missing key, missing split directory or (for `--release`) a missing
dataset exits with code 2 and a one-line message. Result files record `evaluated_datasets`,
`skipped_datasets` (relative to all six) and `complete` (all six, no `--limit`).

Smoke exits 1 if an invariant fails: a metric outside [0, 1], an offline build that is not
deterministic, or any attempted LLM call. The whole smoke run executes under a meter that wraps
the Anthropic SDK's `Messages.create`, counts every attempt and raises instead of sending it, so
the result's `llm_calls_attempted` is measured, not assumed.

Full runs also need Ollama (`nomic-embed-text`) for embeddings. Embeddings, utility questions
and concepts are cached under `$SPECTRAG_EVAL_CACHE` (default `~/.cache/spectrag-eval/`).
Result JSON goes to `eval/results/` (git-ignored) or `$SPECTRAG_EVAL_OUT`.

## Data layout

`<split-root>/<dataset>/corpus.jsonl` — `{"doc_id", "text", "title"?}`;
`<split-root>/<dataset>/questions.jsonl` — `{"qid", "question", "answers": [..], "evidence": [doc_id, ..]}`.
Dataset names: `multihoprag`, `ultradomain`, `musique`, `legal`, `hotpotqa`, `2wikimultihopqa`.
Questions are sorted by `qid` so results do not depend on file order. Loader errors name the file
and line only, never question or answer text.

## Metrics (per dataset; SpectRAG and NaiveRAG)

- **evidence_recall**: share of gold evidence documents present in the token-budgeted context
  (`retrieval.token_budget`). NaiveRAG = dense top-k packed into the same budget.
- **precision_at_5**: share of the top 5 selected chunks (by final score) from an evidence document; denominator 5.
- **mrr**: reciprocal rank of the first evidence chunk.
- **em / f1**: SQuAD-normalised, only on questions with gold answers.
- **judge_win_rate_vs_naiverag**: pairwise judge, both answer orders, win 1 / tie or invalid 0.5 / loss 0.
  The verdict must be exactly `A`, `B` or `TIE` (case and surrounding punctuation ignored); any
  other reply is counted in `judge_invalid_verdicts`, never guessed.
- **index cost**: Anthropic calls and input/output tokens during the **first build only** (counted by
  wrapping the SDK's `Messages.create` in-process). The second build hits warm caches and is not
  metered. Generator and judge tokens are **not** included in index cost and are not reported yet.
- **build_time_s**, **peak_rss_mb** (process peak RSS), **query_p50_ms / query_p95_ms** (retrieval only, no generation).
- **top10_stability**: mean top-10 overlap between two builds. The second build reuses the warm
  utility-question and concept caches, so this measures non-LLM nondeterminism (e.g. the
  eigensolver), **not** variance of the index-time LLM.

## Contamination check

Every `run` first asks the index-time LLM each question with no context (pinned prompt
`pins.CONTAMINATION_USER`) and records `closed_book_em` and `closed_book_contains` per dataset in
the result file. A benchmark does not count until this has been recorded.

## Pinning

`eval/pins.py` holds models, judge/contamination prompts, seed and harness version. Every result
file stores `pins` (including SHA-256 prefixes of the library's generation and index-time prompts),
the redacted library config and `config_hash` (excludes secrets, paths and endpoints).
