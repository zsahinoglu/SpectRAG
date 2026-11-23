---
name: theory
description: Produce a derivation-backed theory proposal (T1–T7) for SpectRAG via the theory agent, then have the red team attack it.
argument-hint: "[T-ID]"
disable-model-invocation: true
---

Work on theory item **$ARGUMENTS**.

1. Read its row in `docs/agents/BACKLOG.md`. In Phase 1, only T5 may touch library code;
   every other T-item produces a proposal document and optional prototype under
   `experiments/`.
2. Delegate to the **theory** subagent: "Write the proposal for $ARGUMENTS following your
   proposal format and submit it with scripts/board.sh submit $ARGUMENTS."
3. When it returns, delegate to the **red-team** subagent: "Attack the derivation submitted
   for $ARGUMENTS. Post the weakest assumption and a counterexample or contradicting
   reference with scripts/board.sh note $ARGUMENTS. Run scripts/board.sh block only for a
   fatal flaw."
4. Print the branch, the proposal's predicted effect, and the red team's main objection.
