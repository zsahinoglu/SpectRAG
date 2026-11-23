#!/usr/bin/env bash
# Open the SpectRAG agent team in tmux: a "control" window plus one window per named agent
# from docs/agents/TEAM.conf. Each agent window waits for its item's dependencies to merge,
# then runs Claude Code with that agent's role in its own git worktree.
#
# Usage:  scripts/launch-team.sh [team-file]
#   SPECTRAG_TEAM_MODE=watch|auto   watch (default): interactive sessions you can follow and
#                                   answer; auto: headless run-item.sh with a USD budget
#   SPECTRAG_TMUX_SESSION           tmux session name (default: spectrag)
#   SKIP_GUARD_CHECK=1              start even if scripts/check-guards.sh has not passed
#
# Navigate: Ctrl-b w (window list) · Ctrl-b n / p (next / previous) · Ctrl-b d (detach).
# Re-attach later with: tmux attach -t spectrag
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [ "${1:-}" = "--control" ]; then   # the control window's loop
  while :; do
    clear; echo "SpectRAG control · $(date '+%Y-%m-%d %H:%M')   (refresh 60s, Ctrl-C for a shell)"
    echo; echo "YOUR DECISIONS (answer: scripts/board.sh answer <ID> \"...\"):"
    scripts/board.sh list needs-human || true
    echo; echo "In flight:"; scripts/board.sh list in-progress review approved blocked || true
    echo; echo "Recently merged to main:"; git log --oneline -5 origin/main 2>/dev/null || true
    git fetch -q origin 2>/dev/null || true
    sleep 60
  done
fi

TEAM="${1:-docs/agents/TEAM.conf}"
SESSION="${SPECTRAG_TMUX_SESSION:-spectrag}"

for bin in tmux claude git; do
  command -v "$bin" >/dev/null || { echo "Missing '$bin' on PATH." >&2; exit 1; }
done
[ -f "$TEAM" ] || { echo "No team file at $TEAM." >&2; exit 1; }
scripts/board.sh list >/dev/null 2>&1 || { echo "No board yet. Run scripts/setup-local.sh first." >&2; exit 1; }
if [ ! -f logs/guards-passed ] && [ "${SKIP_GUARD_CHECK:-}" != "1" ]; then
  echo "Run scripts/check-guards.sh first (it must report failed: 0)." >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Team already running. Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
mkdir -p logs

tmux new-session -d -s "$SESSION" -n control -c "$ROOT" \
  "scripts/launch-team.sh --control; exec \${SHELL:-bash}"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
tmux set-option -t "$SESSION" allow-rename off >/dev/null

idx=0
while read -r NAME ROLE ITEM START _REST; do
  case "${NAME:-}" in ""|\#*) continue ;; esac
  START="${START:-hold}"
  tmux new-window -d -t "$SESSION:" -n "$NAME" -c "$ROOT" \
    -e "SPECTRAG_TEAM_MODE=${SPECTRAG_TEAM_MODE:-watch}" \
    "scripts/agent-window.sh '$NAME' '$ROLE' '$ITEM' '$START' $idx"
  sleep 0.2
  printf '  %-8s %-16s %-6s %s\n' "$NAME" "$ROLE" "$ITEM" "$START"
  if [ "$START" = "auto" ]; then idx=$((idx + 1)); fi
done < "$TEAM"

echo "Team started in tmux session '$SESSION'."
if [ -n "${TMUX:-}" ]; then
  tmux switch-client -t "$SESSION"
else
  exec tmux attach -t "$SESSION"
fi
