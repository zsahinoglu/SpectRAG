---
name: gate
description: Prepare a SpectRAG release gate - full evaluation, red-team sweep, gate report, changelog and version bump - for human approval.
argument-hint: "[version, e.g. v0.2]"
disable-model-invocation: true
---

Prepare the release gate for **$ARGUMENTS**.

1. Check `docs/agents/RELEASES.md` for the gate criteria of $ARGUMENTS and confirm every item
   in scope is `merged` or `deferred` on the board (`scripts/board.sh list`). If items remain
   open, list them and stop.
2. Delegate to the **evaluator**: "Run `python -m eval run --release` on origin/main and add
   the result to the scoreboard (submit as SCORE-<date>)."
3. Delegate to the **red-team**: "Run a full sweep (`all`) against origin/main and report open
   severe findings."
4. Delegate to the **release-manager**: "Write docs/reports/gate-$ARGUMENTS.md in your gate
   report format, update CHANGELOG.md, bump the version, and submit it as
   REL-$ARGUMENTS with --release."
5. Print the gate report's criteria table and its yes/no decisions for the human. Remind the
   human: to approve, run `scripts/merge-gate.sh --release REL-$ARGUMENTS` once Lens has
   passed it, then `scripts/release-local.sh $ARGUMENTS`.
