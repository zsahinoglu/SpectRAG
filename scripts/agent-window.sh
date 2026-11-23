#!/usr/bin/env bash
# Runs inside one tmux window: one named agent, one backlog item.
# Started by scripts/launch-team.sh; you can also run it by hand in any terminal.
#
# Usage: scripts/agent-window.sh <NAME> <ROLE> <ITEM|-> [auto|hold] [stagger-index]
#
#   SPECTRAG_TEAM_MODE=watch (default)  interactive Claude Code session; you watch it and
#                                       answer any permission prompt in this window
#   SPECTRAG_TEAM_MODE=auto             headless via scripts/run-item.sh, with its USD budget
#   SPECTRAG_ITEM_BUDGET                USD cap per item in auto mode (default 20)
#   SPECTRAG_REVIEW_BUDGET              USD cap per review for Lens (default 5)
#   SPECTRAG_AUTOMERGE=0                Lens reviews but leaves merging to you
#   SPECTRAG_POLL_SECONDS               dependency re-check interval (default 120)
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

NAME="${1:?usage: agent-window.sh NAME ROLE ITEM [auto|hold] [index]}"
ROLE="${2:?role}"
ITEM="${3:?item or -}"
START="${4:-hold}"
IDX="${5:-0}"
MODE="${SPECTRAG_TEAM_MODE:-watch}"
BUDGET="${SPECTRAG_ITEM_BUDGET:-20}"
POLL="${SPECTRAG_POLL_SECONDS:-120}"
B=scripts/board.sh
export SPECTRAG_AGENT_NAME="$NAME"

printf '\033]2;%s\007' "$NAME"   # terminal title
stay_open() { echo; echo "[$NAME] $1"; exec "${SHELL:-bash}"; }

banner() {
  clear
  echo "=================================================================="
  echo "  $NAME  ·  $ROLE"
  if [ "$ITEM" != "-" ]; then
    echo "  Item:      $ITEM — $($B get "$ITEM" title)   (release $($B get "$ITEM" release))"
    echo "  State:     $($B get "$ITEM" state)     Needs: $($B get "$ITEM" deps)"
    echo "  Done when: $($B get "$ITEM" done | cut -c1-200)"
  fi
  echo "  Mode:      $MODE   Start: $START"
  echo "=================================================================="
}

isolate_keys() {  # same rule as run-item.sh: only the evaluator gets SpectRAG's LLM key
  if [ "$ROLE" = "evaluator" ] && [ -f "$HOME/.config/spectrag/eval.env" ]; then
    set -a; . "$HOME/.config/spectrag/eval.env"; set +a
  else
    unset SPECTRAG_ANTHROPIC_API_KEY OPENAI_API_KEY SPECTRAG_TEST_DIR || true
  fi
}

run_agent() {
  sleep $((IDX * 5))   # stagger worktree creation, as run-batch.sh does
  if [ "$MODE" = "auto" ] && [ "$ITEM" != "-" ]; then
    scripts/run-item.sh "$ITEM" "$ROLE" "$BUDGET"; return
  fi
  isolate_keys
  git fetch -q origin main 2>/dev/null || true
  if [ "$ITEM" = "-" ]; then
    claude --agent "$ROLE" --permission-mode acceptEdits; return
  fi
  local slug prompt
  slug="$(echo "$NAME-$ITEM" | tr '[:upper:]' '[:lower:]')-$(date +%Y%m%d-%H%M%S)"
  prompt="You are $NAME, the SpectRAG team's $ROLE, working on backlog item $ITEM
(see docs/agents/BACKLOG.md and scripts/board.sh show $ITEM). Follow CLAUDE.md and your role's
procedure exactly. Finish with scripts/board.sh submit $ITEM (submission text on stdin), or, if
you are blocked on a human decision, scripts/board.sh ask $ITEM \"<question>\" and stop."
  claude --agent "$ROLE" --worktree "$slug" --permission-mode acceptEdits "$prompt"
}

# Lens (pr-reviewer): reviews every submission in state "review", then runs the merge gate.
review_loop() {
  isolate_keys
  mkdir -p logs
  while :; do
    banner
    for id in $($B ids review); do
      echo "[$(date +%H:%M)] reviewing $id ($($B get "$id" branch))"
      git fetch -q origin
      claude -p "You are Lens. Review the submission for board item $id. Follow .claude/skills/review-pr/SKILL.md exactly." \
        --agent pr-reviewer --permission-mode acceptEdits --permission-prompts none \
        --max-budget-usd "${SPECTRAG_REVIEW_BUDGET:-5}" --output-format json \
        > "logs/review-$id-$(date +%Y%m%d-%H%M%S).json" 2>&1
      echo "    → $($B get "$id" state)  $($B get "$id" verdict | cut -c1-14)"
      if [ "$($B get "$id" state)" = review ]; then
        $B block "$id" "review run ended without a verdict; see logs/review-$id-*.json"
      fi
    done
    if [ "${SPECTRAG_AUTOMERGE:-1}" = "1" ]; then scripts/merge-gate.sh; fi
    echo; $B list review approved blocked
    echo; echo "Next pass in 60s. Ctrl-C for a shell."
    sleep 60
  done
}

banner
[ -f ".claude/agents/$ROLE.md" ] || stay_open "No agent file .claude/agents/$ROLE.md."

if [ "$ROLE" = "pr-reviewer" ]; then
  trap 'stay_open "Review loop stopped."' INT
  review_loop
fi

if [ "$ITEM" = "-" ]; then
  read -r -p "Press Enter to open a $ROLE session for $NAME (Ctrl-C to skip) " _ || true
  run_agent
  stay_open "Session ended."
fi

OWNER="$($B get "$ITEM" owner)"
[ -n "$OWNER" ] || stay_open "$ITEM is not on the board (scripts/board.sh list)."
[ "$OWNER" = "human" ] && stay_open "$ITEM is owned by a human; agents do not run it."
if [ "$OWNER" != "$ROLE" ]; then echo "Note: backlog owner of $ITEM is '$OWNER'; running as '$ROLE'."; fi
case "$($B get "$ITEM" state)" in
  merged|deferred) stay_open "$ITEM is already $($B get "$ITEM" state)." ;;
esac

while :; do
  if UNMET="$($B deps-met "$ITEM")"; then
    if [ "$START" = "hold" ]; then
      read -r -p "Dependencies met. Press Enter to start $NAME on $ITEM (Ctrl-C to skip) " _ || true
    else
      echo "Dependencies met. Starting $NAME on $ITEM."
    fi
    run_agent
    stay_open "$NAME finished $ITEM → $($B get "$ITEM" state). Details: scripts/board.sh show $ITEM"
  fi
  echo "[$(date +%H:%M)] waiting for merged: $UNMET"
  ans=""
  read -r -t "$POLL" -p "  Enter = re-check now · 'force' = start anyway: " ans; rc=$?
  if [ $rc -eq 1 ]; then sleep "$POLL"; elif [ $rc -gt 128 ]; then echo; fi   # EOF: no terminal
  if [ "$ans" = "force" ]; then
    run_agent
    stay_open "$NAME finished $ITEM (started with unmet dependencies: $UNMET)."
  fi
done
