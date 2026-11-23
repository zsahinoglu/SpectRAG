#!/usr/bin/env bash
# One-time setup of a fully local SpectRAG workspace for the agent team. No GitHub.
#
# Usage:  scripts/setup-local.sh <spectrag-source-folder> [<spectrag-home>]
#   source: your local SpectRAG folder (a git repo keeps its history; a plain folder works too)
#   home:   where the workspace goes (default: ~/spectrag)
# Run from the root of the unpacked starter kit. Requires git, tar and python3.
#
# Creates:
#   <home>/origin.git     local bare repo that plays GitHub's role; main is protected
#   <home>/spectrag-dev   your working copy, with this kit committed on top
#   <home>/board          task board (replaces issues, PRs, labels, review comments)
#   <home>/releases       exported releases
#   ~/.config/spectrag/merge-token   secret that only scripts/merge-gate.sh uses
set -euo pipefail
SRC="$(cd "${1:?usage: setup-local.sh <spectrag-source-folder> [<home>]}" && pwd)"
HOME_DIR="${2:-$HOME/spectrag}"
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ORIGIN="$HOME_DIR/origin.git"; WORK="$HOME_DIR/spectrag-dev"
[ -e "$ORIGIN" ] && { echo "$ORIGIN already exists; refusing to overwrite." >&2; exit 1; }
mkdir -p "$HOME_DIR/releases"

echo "==> 1/5 Local origin from $SRC"
if git -C "$SRC" rev-parse --git-dir >/dev/null 2>&1 && [ "$(git -C "$SRC" rev-parse --show-toplevel)" = "$SRC" ]; then
  [ -z "$(git -C "$SRC" status --porcelain)" ] || echo "   note: uncommitted changes in $SRC are NOT copied."
  git clone -q --bare --no-local "$SRC" "$ORIGIN"
  CUR="$(git -C "$ORIGIN" symbolic-ref --short HEAD)"
  if [ "$CUR" != main ]; then git -C "$ORIGIN" branch -m "$CUR" main; git -C "$ORIGIN" symbolic-ref HEAD refs/heads/main; fi
else
  TMP="$(mktemp -d)"; (cd "$SRC" && tar cf - --exclude ./.git .) | (cd "$TMP" && tar xf -)
  git -C "$TMP" init -q -b main; git -C "$TMP" add -A
  git -C "$TMP" -c user.name="${GIT_AUTHOR_NAME:-spectrag}" -c user.email="${GIT_AUTHOR_EMAIL:-spectrag@localhost}" commit -q -m "Import SpectRAG"
  git clone -q --bare "$TMP" "$ORIGIN"; rm -rf "$TMP"
fi
git -C "$ORIGIN" remote remove origin 2>/dev/null || true

echo "==> 2/5 Protecting main (non-fast-forward and deletes refused; main/tags need the merge token)"
git -C "$ORIGIN" config receive.denyNonFastForwards true
git -C "$ORIGIN" config receive.denyDeletes true
mkdir -p "$HOME/.config/spectrag"; chmod 700 "$HOME/.config/spectrag"
TOKEN_FILE="$HOME/.config/spectrag/merge-token"
[ -s "$TOKEN_FILE" ] || python3 -c 'import secrets;print(secrets.token_hex(32))' > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"
hash256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi; }
tr -d '\n' < "$TOKEN_FILE" | hash256 | cut -d' ' -f1 > "$ORIGIN/merge-token.sha256"
cat > "$ORIGIN/hooks/pre-receive" <<'HOOK'
#!/bin/sh
# Only scripts/merge-gate.sh and scripts/release-local.sh (which hold the token) may update
# main or tags. Agents push their own branches freely.
want=$(cat "$GIT_DIR/merge-token.sha256" 2>/dev/null)
if command -v sha256sum >/dev/null 2>&1; then h() { sha256sum; }; else h() { shasum -a 256; }; fi
have=$(printf %s "${SPECTRAG_MERGE_TOKEN:-}" | h | cut -d' ' -f1)
while read -r old new ref; do
  case "$ref" in
    refs/heads/main|refs/heads/master|refs/tags/*)
      if [ -z "$want" ] || [ "$have" != "$want" ]; then
        echo "main is protected: only scripts/merge-gate.sh may update $ref." >&2; exit 1
      fi ;;
  esac
done
exit 0
HOOK
chmod +x "$ORIGIN/hooks/pre-receive"

echo "==> 3/5 Working copy with the starter kit"
git clone -q "$ORIGIN" "$WORK"; cd "$WORK"
(cd "$KIT_DIR" && tar cf - --exclude ./.git --exclude ./README-KIT.md --exclude ./logs .) | tar xf -
chmod +x .claude/hooks/*.py scripts/*.sh
cat >> .gitignore <<'GI'

# Agent team
.claude/worktrees/
.claude/settings.local.json
logs/
.venv/
GI
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-spectrag}" -c user.email="${GIT_AUTHOR_EMAIL:-spectrag@localhost}" \
  commit -q -m "Add agent team starter kit (local mode)"
SPECTRAG_MERGE_TOKEN="$(tr -d '\n' < "$TOKEN_FILE")" git push -q origin main

echo "==> 4/5 Task board"
scripts/board.sh init | tail -1

echo "==> 5/5 Python environment"
PYBIN="$(command -v python3.11 || command -v python3)"
"$PYBIN" -m venv .venv
if [ -f requirements.txt ]; then .venv/bin/pip install -q -r requirements.txt pytest || echo "   pip install failed; fix and re-run: .venv/bin/pip install -r requirements.txt pytest"; fi

cat <<MSG

Done.
  Working copy: $WORK
  Board:        $HOME_DIR/board        (scripts/board.sh list)
  Releases:     $HOME_DIR/releases

Next (see SETUP-CHECKLIST.md, section C):
  - ~/.config/spectrag/eval.env with SPECTRAG_ANTHROPIC_API_KEY=... and SPECTRAG_TEST_DIR=...
  - frozen test splits in ~/spectrag-private/eval-test/
  - cd $WORK && source .venv/bin/activate && claude   (trust the folder once)
  - scripts/check-guards.sh   then   scripts/launch-team.sh
MSG
