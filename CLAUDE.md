# SpectRAG — agent team rules

This repository is the **local working copy** of SpectRAG. **There is no GitHub.** `origin` is a
local bare repository (`<spectrag-home>/origin.git`); issues, pull requests, labels and review
comments are replaced by the local board, which you use only through `scripts/board.sh`.
A team of Claude Code agents turns the research code into a production library, following the
"SpectRAG — Production Library Plan". The backlog of record is `docs/agents/BACKLOG.md`; release
gates are in `docs/agents/RELEASES.md`.

## Current phase

**Phase 1 — production fixes and prep** (releases v0.2 → v0.3 → v0.4 → v1.0).
Phase 2 items (I1, I3, T1–T4, T6, T7) are proposal-only until v1.0 ships: write documents under
`docs/theory/`, never library code for them.

## Roles and lanes

| Role (agent file) | May edit | Must never edit |
| --- | --- | --- |
| architect | `spectrag/` structure, `docs/` (not `docs/agents/`) | `eval/`, `.claude/`, `scripts/` |
| builder | files needed by its one backlog item, plus tests | `eval/`, `docs/agents/`, `.claude/`, `scripts/` |
| theory | `docs/theory/`, `experiments/` | library code in Phase 1 (except T5), `eval/` |
| evaluator | `eval/`, `docs/evaluation/` | library code |
| tuner | `spectrag/settings*` presets, `experiments/` | `eval/`, test splits |
| red-team | new failing tests under `tests/security/`, board findings | fixes to library code |
| release-manager | `CHANGELOG.md`, `pyproject.toml` version, `docs/reports/`, `docs/agents/` | library code |
| pr-reviewer (Lens) | nothing (board verdicts and notes only) | everything |

Hooks in `.claude/hooks/` enforce these lanes; if a hook blocks you, you are outside your lane.
Stop and report instead of working around it.

## Hard rules

1. **One backlog item per branch and per submission.** Branch name: `<role>/<item-id>-<short-slug>`,
   for example `builder/h3-safe-format`. Commit messages start with the item ID: `H3: ...`.
   Claim with `scripts/board.sh claim <ID>`; submit with `scripts/board.sh submit <ID>`, piping
   in the filled-in `docs/agents/SUBMISSION_TEMPLATE.md`. Submitting pushes your branch.
2. **Never touch `main`, never force-push or rebase a submitted branch, never merge.** Lens
   reviews each submission; `scripts/merge-gate.sh` (human-only) runs the tests and merges.
   To pick up newer `main`, `git merge origin/main` into your branch.
3. **Every code change ships with tests.** Run `python -m pytest -q` before submitting.
   Once `eval/` exists, also run `python -m eval smoke`.
4. **Fill in the submission template completely**, including "Eval impact". A submission that
   changes retrieval output without an eval report will be blocked by review.
5. **Never read or search the frozen test split** (`~/spectrag-private/`). Only the evaluator
   may, and only through `python -m eval`.
6. **No `pickle` in any load path.** Pin model and prompt versions in index manifests.
7. **Stay inside your item's scope.** If you find a separate problem, file it:
   `scripts/board.sh file "<title>" <<< "<details>"`, and continue with your item.
8. **When blocked or unsure, stop.** `scripts/board.sh ask <ID> "<the decision you need>"`.
   Do not guess on public API names, on-disk format choices, or anything listed as an open
   decision (D1–D5) in the plan.
9. Keep submissions small: aim for under ~400 changed lines excluding tests and fixtures.
   Split larger items into sub-items (`H4a`, `H4b`) and say so on the board.
10. Never touch the board's files, the origin repository, or the merge token directly; hooks
    block it. `scripts/board.sh` is the only interface.

## Environment

- Python 3.11. Install with `pip install -r requirements.txt` until `pyproject.toml` exists,
  then `pip install -e ".[dev]"`.
- Ollama must be running locally for embedding-dependent tests; unit and golden tests must use
  the deterministic fake embedder and need no network.
- **Hosted LLM for SpectRAG itself: Claude Haiku (`claude-haiku-5-5`), not OpenAI.** It is used
  for index-time concepts and utility questions, answer generation, and the evaluation judge.
  The key lives in `SPECTRAG_ANTHROPIC_API_KEY` (never `ANTHROPIC_API_KEY`, which Claude Code
  itself would use for billing your agent sessions). SpectRAG code reads
  `SPECTRAG_ANTHROPIC_API_KEY` first and falls back to `ANTHROPIC_API_KEY`.
- Builders have no SpectRAG LLM key on purpose; their tests mock the LLM client. Anything that
  needs real hosted-LLM calls belongs to the evaluator.
- OpenAI is not used anywhere in the default configuration. Do not add OpenAI calls.

## Useful commands

- Tests: `python -m pytest -q`
- Evaluation (after v0.2): `python -m eval smoke`, `python -m eval run --dev`,
  `python -m eval run --release` (evaluator only)
- Board: `scripts/board.sh list [state...]`, `scripts/board.sh show <ID>`,
  `scripts/board.sh deps-met <ID>`, `scripts/board.sh note <ID> <<< "..."`
- A submission's diff: `git fetch origin && git diff origin/main...origin/<branch>`
