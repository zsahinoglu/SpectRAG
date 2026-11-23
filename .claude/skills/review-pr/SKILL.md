---
name: review-pr
description: Independently review one SpectRAG submission on the local board and record PASS or BLOCK. Run by Lens's window for every submission; runnable locally.
argument-hint: "[ITEM-ID]"
allowed-tools: Bash(scripts/board.sh *) Bash(git fetch *) Bash(git diff *) Bash(git log *) Bash(git show *) Read Grep Glob
---

Review the submission for board item **$ARGUMENTS**. You did not write it. Your review is the
only per-change review it gets before the merge gate merges it, so check correctness carefully.

## Gather

1. `scripts/board.sh show $ARGUMENTS` — title, "Done when", branch, sha, submission text, log.
2. `git fetch origin`, then `git diff origin/main...origin/<branch>` and
   `git log --oneline origin/main..origin/<branch>`.
3. Read surrounding code for every changed function, not just the diff
   (`git show origin/<branch>:<path>`).

## Check, in this order

1. **Scope**: the submission implements the item named and nothing unrelated.
2. **Done when**: the backlog row's "Done when" is met, or the submission says clearly which
   part is left for a sub-item.
3. **Tests**: new behavior has tests that would fail without the change. No existing test was
   deleted, skipped or loosened without a stated reason.
4. **Lanes**: files changed are inside the author role's lane (see `CLAUDE.md`). Any change
   under `eval/`, `docs/agents/`, `.claude/` or `scripts/` by a builder or architect is an
   automatic block.
5. **Safety**: no `pickle.load`/`torch.load` without `weights_only`/`eval`/`yaml.load` in load
   paths; no secrets or keys; no network calls in unit tests.
6. **Behavior**: if retrieval output or scores can change, "Eval impact" contains an
   evaluator report (or the log has the evaluator's delta table). Missing report is a block.
7. **Correctness**: logic errors, unhandled edge cases (empty corpus, isolated nodes, tiny
   graphs, non-convergence), wrong math.

## Decide and record

Record exactly one verdict:

```
scripts/board.sh verdict $ARGUMENTS PASS|BLOCK <<'MD'
Item: <ITEM-ID> — Done when: met / partly met / not met
1. [BLOCKING|NIT] path/to/file.py:123 — problem — concrete fix
...
MD
```

Block only for real problems. Style preferences are NITs and never block.
