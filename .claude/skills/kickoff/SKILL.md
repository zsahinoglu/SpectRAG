---
name: kickoff
description: Plan a SpectRAG release - order its backlog items by dependency, group them into parallel batches with disjoint files, make sure each is on the local board, and print the team assignments.
argument-hint: "[release, e.g. v0.2]"
disable-model-invocation: true
---

Kick off release **$ARGUMENTS**.

1. Delegate to the **architect** subagent with this task: "Read docs/agents/RELEASES.md and
   docs/agents/BACKLOG.md. For release $ARGUMENTS: list its items; order them by the
   'Depends on' column; group them into batches whose items touch disjoint files (inspect the
   code to decide); split any item likely to exceed ~400 changed lines into sub-items. Make
   sure every item and sub-item is on the board (`scripts/board.sh list`; create missing
   sub-items with `scripts/board.sh add`; never duplicate). Return the batches as a table:
   batch, item, owner role, files touched, board state."
2. Print the architect's table, then the `docs/agents/TEAM.conf` lines that assign this
   release's items to named agents (auto for batch 1 and for items that should start as soon
   as their dependencies merge, hold otherwise). The human pastes them into TEAM.conf and
   runs `scripts/launch-team.sh`. Also print the headless alternative:
   `scripts/run-batch.sh <ITEM-ID> <ITEM-ID> ...` for batch 1.
3. List any open human decisions (`scripts/board.sh list needs-human`) that block the release.
