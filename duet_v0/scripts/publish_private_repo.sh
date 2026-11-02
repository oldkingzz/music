#!/usr/bin/env bash
# Push this repo's committed history to a PRIVATE team remote. Prepared 2026-10-16; not run yet.
#
# Before you run it:
#   1. Create an EMPTY private repository on the web (GitHub: New repository -> Private, no README,
#      no .gitignore, no licence). Add the 3 teammates as collaborators there.
#   2. Commit what you want to share. Only commits are pushed; uncommitted and git-ignored files stay here.
#
# Usage:
#   REMOTE_URL=git@github.com:<owner>/duet.git scripts/publish_private_repo.sh            # asks before pushing
#   REMOTE_URL=https://github.com/<owner>/duet.git DRY_RUN=1 scripts/publish_private_repo.sh   # checks only
#
# Options (environment variables):
#   REMOTE_URL   required. SSH or HTTPS URL of the empty private repo. URLs with embedded credentials are refused.
#   REMOTE_NAME  git remote name to add (default: team). An existing remote with another URL is refused.
#   BRANCH       remote branch to create (default: main).
#   DRY_RUN=1    print everything and run the checks, but do not add the remote or push.
#
# What it does, in order: refuse without REMOTE_URL; print every file that will be published with its size;
# print the history size; run scripts/check_repo_health.py (stop on any ERROR); run gitleaks if it is on PATH
# or in ~/work/tools (stop on any finding); ask you to type "push"; add the remote; git push -u (no --force,
# no tags).
set -euo pipefail

die() { echo "publish: $*" >&2; exit 1; }

[ -n "${REMOTE_URL:-}" ] || die "REMOTE_URL is not set. Example: REMOTE_URL=git@github.com:<owner>/duet.git $0"
case "$REMOTE_URL" in
  *://*:*@*|*://*@*:*) die "REMOTE_URL contains credentials. Use SSH or a credential helper instead." ;;
  git@*:*|ssh://*|https://*) ;;
  *) die "REMOTE_URL must be an SSH (git@host:owner/repo.git) or https:// URL." ;;
esac
REMOTE_NAME="${REMOTE_NAME:-team}"
BRANCH="${BRANCH:-main}"

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
cd "$ROOT"
HEAD_SHA="$(git rev-parse --short HEAD)"

echo "== Files that will be published (tracked at HEAD $HEAD_SHA), with sizes in bytes"
git ls-tree -r -l HEAD | awk -F'\t' '{ split($1, m, " "); printf "%12s  %s\n", m[4], $2 }'
echo
git ls-tree -r -l HEAD | awk '{ s += $4; n++ } END { printf "== %d files, %.1f MiB at HEAD\n", n, s / 1048576 }'
echo "== History: $(git rev-list --count HEAD) commits on this branch"
git count-objects -vH | grep -E "^(size-pack|size):" | sed 's/^/   /'
echo "   (the push sends every object reachable from HEAD, including old versions of files)"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo
  echo "NOTE: you have uncommitted changes. They will NOT be pushed:"
  git status --short | head -20
fi

echo
echo "== Repo health check"
python3 scripts/check_repo_health.py --quiet || die "health check reported errors (see above); fix them first"

echo
echo "== Secret scan"
GL="$(command -v gitleaks || true)"
[ -n "$GL" ] || GL="$(ls "$HOME"/work/tools/gitleaks/*/bin/gitleaks 2>/dev/null | tail -1 || true)"
if [ -n "$GL" ]; then
  "$GL" git --no-banner --redact --config scripts/ci/gitleaks.toml . || die "gitleaks found possible secrets"
else
  echo "gitleaks not found; skipped. (Install: see docs/COLLABORATION.md, onboarding step 6.)"
fi

if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  existing="$(git remote get-url "$REMOTE_NAME")"
  [ "$existing" = "$REMOTE_URL" ] || die "remote '$REMOTE_NAME' already points to $existing. Set REMOTE_NAME to another name."
  ADD_REMOTE=0
else
  ADD_REMOTE=1
fi

echo
echo "== Commands that will run"
[ "$ADD_REMOTE" = 1 ] && echo "   git remote add $REMOTE_NAME $REMOTE_URL"
echo "   git push -u $REMOTE_NAME HEAD:$BRANCH"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo
  echo "DRY_RUN=1: stopping here. Nothing was added or pushed."
  exit 0
fi

echo
echo "Is the remote repository PRIVATE, and is it the empty repo you just created?"
printf 'Type "push" to continue: '
read -r answer < /dev/tty
[ "$answer" = "push" ] || die "not confirmed; nothing pushed"

[ "$ADD_REMOTE" = 1 ] && git remote add "$REMOTE_NAME" "$REMOTE_URL"
git push -u "$REMOTE_NAME" "HEAD:$BRANCH"
echo
echo "Pushed $HEAD_SHA to $REMOTE_URL ($BRANCH). Next: open the repo page, confirm it says Private,"
echo "and check that the first CI run starts under the Actions tab."
