#!/usr/bin/env bash
# Run several backlog items in parallel, each in its own worktree.
#
# Usage:  scripts/run-batch.sh <ITEM-ID> [<ITEM-ID> ...]
#   MAX_PARALLEL (default 3) limits concurrent agents.
#   SPECTRAG_ITEM_BUDGET (default 20) caps USD per item.
#
# Only batch items that touch disjoint files (the /kickoff plan groups them that way).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
mkdir -p logs

for ITEM in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 10; done
  echo "==> starting $ITEM"
  scripts/run-item.sh "$ITEM" > "logs/$ITEM-console.txt" 2>&1 &
  sleep 5   # stagger worktree creation
done
wait
echo "==> batch finished. Summaries:"
for ITEM in "$@"; do echo "--- $ITEM"; tail -n 3 "logs/$ITEM-console.txt"; done
echo "Board:"; scripts/board.sh list review approved blocked needs-human
