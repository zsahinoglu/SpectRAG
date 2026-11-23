#!/usr/bin/env bash
# Remove agent worktrees whose board item is merged or deferred. Headless runs (claude -p --worktree)
# do not clean up after themselves, so run this weekly.
#
# Usage: scripts/cleanup-worktrees.sh [--dry-run]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
DRY="${1:-}"
BOARD="${SPECTRAG_BOARD:-$(dirname "$(git remote get-url origin)")/board}"

git worktree list --porcelain | awk '/^worktree /{wt=$2} /^branch /{print wt" "$2}' |
while read -r WT REF; do
  case "$WT" in *"/.claude/worktrees/"*) ;; *) continue ;; esac
  BRANCH="${REF#refs/heads/}"
  # Agents usually rename the worktree branch with `git switch -c`; check the branch checked out.
  CUR="$(git -C "$WT" branch --show-current 2>/dev/null || echo "$BRANCH")"
  STATE=NONE
  for d in "$BOARD"/items/*/; do
    [ "$(cat "$d/branch" 2>/dev/null)" = "$CUR" ] && STATE="$(cat "$d/state")" && break
  done
  if [ "$STATE" = "merged" ] || [ "$STATE" = "deferred" ]; then
    echo "remove  $WT  ($CUR: $STATE)"
    if [ "$DRY" != "--dry-run" ]; then
      git worktree unlock "$WT" 2>/dev/null || true
      git worktree remove --force "$WT"
      git branch -D "$CUR" >/dev/null 2>&1 || true
    fi
  else
    echo "keep    $WT  ($CUR: $STATE)"
  fi
done
git worktree prune
