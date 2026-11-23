#!/usr/bin/env bash
# Local replacement for CI, the review gate and auto-merge. HUMAN-ONLY (agents are blocked
# from running it). Lens's window runs it after each review pass.
#
# For every board item in state "approved" (release items only with --release ID):
#   1. the reviewer's PASS must be for the branch's current head commit;
#   2. squash-merge the branch onto origin/main in a scratch worktree;
#   3. run python -m pytest -q, and python -m eval smoke once eval/ exists;
#   4. push main to the local origin with the merge token, and mark the item merged.
# Anything that fails goes to state "blocked" with the reason in the item's log.
#
# Usage: scripts/merge-gate.sh [--release ID]
#   SPECTRAG_PYTHON   interpreter for tests (default: .venv/bin/python, then python3)
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
B=scripts/board.sh
TOKEN_FILE="$HOME/.config/spectrag/merge-token"
[ -r "$TOKEN_FILE" ] || { echo "No merge token at $TOKEN_FILE (run setup-local.sh)." >&2; exit 1; }
PY="${SPECTRAG_PYTHON:-$ROOT/.venv/bin/python}"; [ -x "$PY" ] || PY="$(command -v python3)"
export SPECTRAG_AGENT_NAME=merge-gate
mkdir -p logs

ITEMS="$($B ids approved)"
RELEASE=""
if [ "${1:-}" = "--release" ]; then RELEASE="${2:?--release needs an item ID}"; ITEMS="$RELEASE"; fi

for ID in $ITEMS; do
  [ "$($B get "$ID" state)" = approved ] || { echo "$ID is not approved; skipping."; continue; }
  if [ "$($B get "$ID" kind)" = release ] && [ "$ID" != "$RELEASE" ]; then
    echo "$ID is a release item: merge it yourself with scripts/merge-gate.sh --release $ID"; continue
  fi
  BR="$($B get "$ID" branch)"; SHA="$($B get "$ID" sha)"; V="$($B get "$ID" verdict)"
  git fetch -q origin
  HEAD_SHA="$(git rev-parse -q --verify "origin/$BR" || true)"
  if [ "$V" != "PASS $HEAD_SHA" ] || [ "$SHA" != "$HEAD_SHA" ]; then
    $B block "$ID" "branch $BR changed after review (or review missing); resubmit for review"; continue
  fi
  WT="$(mktemp -d)/gate"; OUT="logs/gate-$ID-$(date +%Y%m%d-%H%M%S).txt"
  git worktree add -q --detach "$WT" origin/main
  ok=1
  if ! git -C "$WT" merge -q --squash "origin/$BR" > "$OUT" 2>&1; then
    ok=0; reason="conflicts with main: merge origin/main into $BR, fix, resubmit"
  else
    if ! git -C "$WT" commit -q -m "$ID: $($B get "$ID" title)" -m "Squash of $BR @ ${SHA:0:9}; reviewed PASS." >> "$OUT" 2>&1; then
      ok=0; reason="nothing to merge: the branch has no changes against main"
    elif [ -d "$WT/tests" ]; then
      (cd "$WT" && "$PY" -m pytest -q) >> "$OUT" 2>&1; rc=$?
      [ $rc -eq 0 ] || [ $rc -eq 5 ] || { ok=0; reason="tests failed (see $OUT)"; }
    fi
    if [ $ok -eq 1 ] && [ -d "$WT/eval" ]; then
      (cd "$WT" && "$PY" -m eval smoke) >> "$OUT" 2>&1 || { ok=0; reason="eval smoke failed (see $OUT)"; }
    fi
  fi
  if [ $ok -eq 1 ]; then
    if SPECTRAG_MERGE_TOKEN="$(cat "$TOKEN_FILE")" git -C "$WT" push -q origin HEAD:main >> "$OUT" 2>&1; then
      MERGED="$(git -C "$WT" rev-parse HEAD)"
      $B note "$ID" "Merged to main as ${MERGED:0:9}. Gate log: $OUT"
      echo merged > "${SPECTRAG_BOARD:-$(dirname "$(git remote get-url origin)")/board}/items/$ID/state"
      echo "MERGED  $ID  ${MERGED:0:9}"
    else
      ok=0; reason="push to main rejected (main moved? it will retry next round)"
      $B note "$ID" "$reason"; echo "RETRY   $ID"
    fi
  else
    $B note "$ID" "$(tail -n 30 "$OUT")"
    $B block "$ID" "$reason"; echo "BLOCKED $ID: $reason"
  fi
  git worktree remove --force "$WT"; rm -rf "$(dirname "$WT")"
done

# Keep the human's checkout current when it is clean and on main.
if [ "$(git branch --show-current)" = main ] && [ -z "$(git status --porcelain -uno)" ]; then
  git merge -q --ff-only origin/main 2>/dev/null || true
fi
