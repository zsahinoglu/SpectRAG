---
name: digest
description: Write the weekly SpectRAG agent-team digest (merged work, scores, blocked items, decisions needed, spend) in one screen.
disable-model-invocation: true
---

Delegate to the **release-manager** subagent: "Write the weekly digest for the last 7 days
to docs/reports/digest-<today>.md, following your digest format. Use
`git log --since='7 days ago' --oneline origin/main`, `scripts/board.sh list` (all states),
`scripts/board.sh list needs-human blocked`, the scoreboard in docs/evaluation/scoreboard.md,
and cost data in logs/*.json if present (field total_cost_usd). Submit it as DIGEST-<today>."

Then print the digest itself here, followed by a numbered list of the decisions the human
needs to make, each answerable yes/no or with one choice, with the command to answer it
(`scripts/board.sh answer <ID> "..."`).
