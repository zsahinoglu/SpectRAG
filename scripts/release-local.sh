#!/usr/bin/env bash
# After a release gate is approved and the release item is merged, tag the release and
# export a clean copy (agent-team tooling removed) to <spectrag-home>/releases/. HUMAN-ONLY.
#
# Usage: scripts/release-local.sh <version>      e.g. scripts/release-local.sh v0.2
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
V="${1:?usage: release-local.sh <version>}"
HOME_DIR="$(dirname "$(git remote get-url origin)")"
TOKEN_FILE="$HOME/.config/spectrag/merge-token"

git fetch -q origin
git cat-file -e "origin/main:docs/reports/gate-$V.md" 2>/dev/null \
  || { echo "No docs/reports/gate-$V.md on main. Run /gate $V and merge its release item first." >&2; exit 1; }

git tag -a "$V" -m "SpectRAG $V" origin/main
SPECTRAG_MERGE_TOKEN="$(tr -d '\n' < "$TOKEN_FILE")" git push -q origin "$V"

OUT="$HOME_DIR/releases/spectrag-$V"
mkdir -p "$HOME_DIR/releases"
git archive --format=tar.gz --prefix="spectrag-$V/" -o "$OUT.tar.gz" "$V" -- . \
  ':(exclude).claude' ':(exclude)CLAUDE.md' ':(exclude)SETUP-CHECKLIST.md' \
  ':(exclude)docs/agents' ':(exclude)docs/reports' \
  ':(exclude)scripts/board.sh' ':(exclude)scripts/merge-gate.sh' ':(exclude)scripts/setup-local.sh' ':(exclude)scripts/release-local.sh' ':(exclude)scripts/run-item.sh' ':(exclude)scripts/run-batch.sh' ':(exclude)scripts/check-guards.sh' ':(exclude)scripts/cleanup-worktrees.sh' ':(exclude)scripts/launch-team.sh' ':(exclude)scripts/agent-window.sh'
git show "$V:docs/reports/gate-$V.md" > "$OUT-gate-report.md"
echo "Tagged $V. Release: $OUT.tar.gz   Gate report: $OUT-gate-report.md"
echo "Share these with your co-authors however you like (email, shared drive, PyPI upload)."
