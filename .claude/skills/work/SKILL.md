---
name: work
description: Implement one SpectRAG backlog item end to end (branch, code, tests, submission) using the builder agent. Use when asked to work on H1–H8, I2, I4, I5 or a sub-item such as H4a.
argument-hint: "[ITEM-ID]"
disable-model-invocation: true
---

Implement backlog item **$ARGUMENTS** for SpectRAG.

1. `scripts/board.sh show $ARGUMENTS`. If the item does not exist, stop and say so. If its
   owner role is not `builder`, stop and name the right command (`/theory`, `/evaluate`,
   `/redteam`, `/gate`, or the tuner agent).
2. Check its dependencies: `scripts/board.sh deps-met $ARGUMENTS`. If it prints unmerged
   items, stop and report them.
3. Delegate the whole item to the **builder** subagent, passing this task verbatim plus the
   backlog row: "Implement backlog item $ARGUMENTS per CLAUDE.md and the builder procedure.
   Finish with scripts/board.sh submit $ARGUMENTS."
4. When the builder returns, print: the branch, a three-line summary, test results, and any
   follow-up item it proposed.
