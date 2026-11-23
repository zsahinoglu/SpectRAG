#!/usr/bin/env bash
# Run one backlog item headlessly with the right agent, in its own git worktree.
#
# Usage:  scripts/run-item.sh <ITEM-ID> [role] [budget-usd]
#   role defaults to the Owner column of docs/agents/BACKLOG.md
#   budget defaults to $SPECTRAG_ITEM_BUDGET or 20 (USD cap for this run)
#
# Output: logs/<ITEM>-<timestamp>.json (Claude's result, session id and cost).
# The agent ends with `scripts/board.sh submit <ITEM>` or `scripts/board.sh ask <ITEM> ...`.
#   SPECTRAG_AGENT_NAME names the agent on the board (default: the role).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ITEM="${1:?usage: run-item.sh <ITEM-ID> [role] [budget-usd]}"
ROLE="${2:-}"
BUDGET="${3:-${SPECTRAG_ITEM_BUDGET:-20}}"

if [ -z "$ROLE" ]; then
  ROLE="$(awk -F'|' -v id="$ITEM" '{gsub(/ /,"",$2); gsub(/ /,"",$5)} $2==id {print $5; exit}' docs/agents/BACKLOG.md)"
fi
if [ -z "$ROLE" ]; then   # sub-items and findings exist only on the board
  ROLE="$(scripts/board.sh get "$ITEM" owner 2>/dev/null || true)"
fi
if [ "$ROLE" = "human" ]; then
  echo "Item $ITEM is owned by a human (scripts/board.sh show $ITEM); agents do not run it." >&2
  exit 1
fi
if [ -z "$ROLE" ]; then
  echo "Item $ITEM not found in docs/agents/BACKLOG.md and no role given." >&2
  exit 1
fi

# Only the evaluator gets SpectRAG's hosted-LLM key (SPECTRAG_ANTHROPIC_API_KEY) and the
# test-split location. The key name differs from ANTHROPIC_API_KEY on purpose: Claude Code
# would otherwise use it to bill the agent session itself.
if [ "$ROLE" = "evaluator" ] && [ -f "$HOME/.config/spectrag/eval.env" ]; then
  set -a; . "$HOME/.config/spectrag/eval.env"; set +a
else
  unset SPECTRAG_ANTHROPIC_API_KEY OPENAI_API_KEY SPECTRAG_TEST_DIR || true
fi

export SPECTRAG_AGENT_NAME="${SPECTRAG_AGENT_NAME:-$ROLE}"
git fetch -q origin main
STAMP="$(date +%Y%m%d-%H%M%S)"
SLUG="$(echo "$ITEM" | tr '[:upper:]' '[:lower:]')-$STAMP"
mkdir -p logs

PROMPT="You are $SPECTRAG_AGENT_NAME, the SpectRAG team's $ROLE, working on backlog item $ITEM
(see docs/agents/BACKLOG.md and scripts/board.sh show $ITEM). Follow CLAUDE.md and your role's
procedure exactly. Finish with scripts/board.sh submit $ITEM (submission text on stdin), or, if
you are blocked on a human decision, scripts/board.sh ask $ITEM \"<question>\" and stop."

echo "==> $ITEM as $ROLE (budget \$$BUDGET) — log: logs/$ITEM-$STAMP.json"
claude -p "$PROMPT" \
  --agent "$ROLE" \
  --worktree "$SLUG" \
  --permission-mode acceptEdits \
  --permission-prompts none \
  --max-budget-usd "$BUDGET" \
  --output-format json \
  > "logs/$ITEM-$STAMP.json"

python3 - "logs/$ITEM-$STAMP.json" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print("Could not parse result:", e); sys.exit(0)
print("result:", (r.get("result") or "")[:800])
print("cost_usd:", r.get("total_cost_usd"), "| session:", r.get("session_id"))
PY
