---
name: redteam
description: Run the SpectRAG red team against a target area (deserialization, injection, poisoning, contamination) or a submission on the board.
argument-hint: "[area | ITEM-ID]"
disable-model-invocation: true
---

Delegate to the **red-team** subagent with target **$ARGUMENTS**.

- A board item ID → review that submission adversarially (`scripts/board.sh show <ID>`,
  `git diff origin/main...origin/<branch>`); post findings with `scripts/board.sh note <ID>`;
  run `scripts/board.sh block <ID> "<reason>"` only for real, severe problems.
- An area (`deserialization`, `injection`, `poisoning`, `contamination`, `all`) → write
  failing tests under `tests/security/` or `tests/redteam/` that demonstrate each problem,
  file one finding per problem (`scripts/board.sh file "<title>" --tag redteam`), submit each
  test branch on its finding, and name the backlog item that must make each test pass.

Print a list of findings with severity and board IDs.
