---
name: evaluate
description: Score SpectRAG with the evaluation harness via the evaluator agent - smoke or dev run for a board item or branch, or the full release run on the frozen test split.
argument-hint: "[ITEM-ID | branch | --release]"
disable-model-invocation: true
---

Run an evaluation for **$ARGUMENTS** by delegating to the **evaluator** subagent.

- A board item ID → check out that item's branch (`scripts/board.sh get <ID> branch`, then
  `origin/<branch>`) in the evaluator's worktree, run `python -m eval run --dev`, compare
  against the latest `main` row in `docs/evaluation/scoreboard.md`, and post the delta table
  with `scripts/board.sh note <ID>`. Run `scripts/board.sh block <ID> "<why>"` if any dataset
  drops more than 1 point or p95 latency / index cost rises more than 10%.
- A branch name → same as above, but print the table instead of posting it.
- `--release` → run `python -m eval run --release` on `origin/main`, append the result to the
  scoreboard on a branch `evaluator/scoreboard-<date>`, and submit it as board item
  `SCORE-<date>` (create it with `scripts/board.sh add SCORE-<date> "Scoreboard <date>" evaluator`).
- No argument → run `python -m eval smoke` on the current checkout and print the result.

If `eval/` does not exist yet, the evaluator's task is backlog item **E0**: build the harness
first (see `docs/agents/BACKLOG.md`).

Report back the evaluator's summary table and any blocking regressions.
