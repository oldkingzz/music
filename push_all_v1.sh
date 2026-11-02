#!/bin/bash
# Uploads everything in this folder to github.com/oldkingzz/music in several commits and pushes, so that no single
# push comes near GitHub's 2 GB limit. Rerun it any time to upload what is new or changed: batches with nothing new
# are skipped, and a push that failed halfway is simply redone.
# Left out: Python environments and caches (.venv, __pycache__, *.pyc), .DS_Store and git's own .git folder.
# Files are added with `git add -f`, because duet_v0/.gitignore (the team's rules) hides recordings and run outputs
# (*.wav, outputs/, live/runs/, data/raw/, data/external/) from a plain `git add`.
#
#   bash push_all_v1.sh          # commit and push
#   bash push_all_v1.sh --plan   # only print the batches (runs no git command)
set -euo pipefail
cd "$(dirname "$0")"
REMOTE=https://github.com/oldkingzz/music.git
LIVE=duet_v0/runs_mini/live/
LIMIT=$((450 * 1024 * 1024))  # bytes of live runs per batch; a batch can pass it by one folder

# size <tab> path of every file to upload, sorted by path
list() {
  find . \( -name .git -o -name .venv -o -name __pycache__ \) -prune -o -type f ! -name '*.pyc' ! -name .DS_Store -print0 \
    | xargs -0 stat -f $'%z\t%N' | sed $'s|\t\\./|\t|' | LC_ALL=C sort -t $'\t' -k2,2
}
# puts the batch number in front: 1 is everything outside the live runs, 2 and up are the live runs, about LIMIT
# bytes each, with the files of one folder always in the same batch
batch() {
  awk -F'\t' -v OFS='\t' -v lim="$LIMIT" -v pre="$LIVE" '
    index($2, pre) != 1 { print 1, $1, $2; next }
    { d = $2; sub(/\/[^\/]*$/, "", d)
      if (!(d in g)) { if (b < 2) b = 2; else if (s[b] >= lim) b++; g[d] = b }
      s[g[d]] += $1; print g[d], $1, $2 }'
}
PLAN=$(list | batch)
N=$(printf '%s\n' "$PLAN" | awk -F'\t' '$1 > n { n = $1 } END { print n }')
printf '%s\n' "$PLAN" | awk -F'\t' '
  { n[$1]++; s[$1] += $2; t += $2; if ($2 >= 100 * 1048576) big = big "\n  " $3 }
  END { for (i = 1; i in n; i++) printf "batch %d: %5d files, %5.0f MB\n", i, n[i], s[i] / 1048576
        printf "total:   %5d files, %5.0f MB\n", NR, t / 1048576
        if (big != "") { print "GitHub rejects files of 100 MB or more:" big; exit 1 } }'
[ "${1:-}" = --plan ] && exit 0

[ -d .git ] || git init -q -b main
git config http.postBuffer 524288000  # send each push as one HTTP request; large chunked pushes can fail
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REMOTE"
git add -u  # changes to and deletions of files already uploaded
for i in $(seq 1 "$N"); do
  if [ "$i" = 1 ]; then msg="Upload code, model weights, data, configs and archives"
  else msg="Upload live run recordings and logs (part $((i - 1)) of $((N - 1)))"; fi
  printf '%s\n' "$PLAN" | awk -F'\t' -v i="$i" '$1 == i { print $3 }' \
    | git --literal-pathspecs add -f --pathspec-from-file=-
  if git diff --cached --quiet; then echo "batch $i: nothing new"
  else git commit -q -m "$msg" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; fi
  echo "pushing batch $i of $N ..."
  git push -u origin main
done
echo "done: https://github.com/oldkingzz/music"
