#!/usr/bin/env python3
"""Size audit of the git repo: the raw numbers behind docs/REPO_AUDIT.md. Read-only, standard library only.

    python3 scripts/repo_audit.py              # Markdown report to stdout
    python3 scripts/repo_audit.py --ignored    # also size the git-ignored paths on disk (slower: du)
    python3 scripts/repo_audit.py --json out.json

It reports: tracked files at HEAD (total, by top-level folder, by extension, largest), the git object
store (git count-objects), every blob in history (git rev-list --objects --all + cat-file),
blobs that are in history but no longer at HEAD, binary files changed in more than one commit
(churn), tracked files that .gitignore would ignore, and file names with spaces or CRLF line ends.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys

MIB = 1024 * 1024
BINARY_EXT = {'wav', 'png', 'pdf', 'npz', 'glb', 'stl', 'svg', 'jpg', 'jpeg', 'mp4', 'mov', 'ttf', 'zip',
              'kicad_pcb', 'dsn', 'ses', 'mid', 'jsonl', 'dxf'}


def git(*args: str, inp: bytes | None = None) -> bytes:
    return subprocess.run(['git', *args], check=True, capture_output=True, input=inp).stdout


def ext(path: str) -> str:
    b = os.path.basename(path)
    return b.rsplit('.', 1)[1].lower() if '.' in b else '(none)'


def mb(n: int) -> str:
    return f'{n / MIB:.2f}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--ignored', action='store_true')
    ap.add_argument('--json')
    args = ap.parse_args()
    os.chdir(git('rev-parse', '--show-toplevel').decode().strip())

    head = git('rev-parse', '--short', 'HEAD').decode().strip()
    # tracked files at HEAD: "<mode> blob <sha> <size>\t<path>"
    tree = []
    for rec in git('ls-tree', '-r', '-l', '-z', 'HEAD').decode('utf-8', 'surrogateescape').split('\0'):
        if not rec:
            continue
        meta, path = rec.split('\t', 1)
        _, typ, sha, size = meta.split()
        if typ == 'blob':
            tree.append((path, sha, int(size)))
    head_shas = {sha for _, sha, _ in tree}
    total = sum(s for *_, s in tree)

    by_dir = collections.Counter()
    n_dir = collections.Counter()
    by_ext = collections.Counter()
    n_ext = collections.Counter()
    for p, _, s in tree:
        top = p.split('/', 1)[0] if '/' in p else '(root files)'
        by_dir[top] += s
        n_dir[top] += 1
        by_ext[ext(p)] += s
        n_ext[ext(p)] += 1

    count_objects = git('count-objects', '-vH').decode().strip().splitlines()

    # every object reachable from any ref
    objs = git('rev-list', '--objects', '--all').decode('utf-8', 'surrogateescape').splitlines()
    batch_in = '\n'.join(line.split(' ', 1)[0] for line in objs).encode() + b'\n'
    names = {}
    for line in objs:
        sha, _, name = line.partition(' ')
        names.setdefault(sha, name)
    blobs = []
    for line in git('cat-file', '--batch-check=%(objecttype) %(objectname) %(objectsize) %(objectsize:disk)',
                    inp=batch_in).decode().splitlines():
        typ, sha, size, disk = line.split()
        if typ == 'blob':
            blobs.append((names.get(sha, ''), sha, int(size), int(disk)))
    hist_raw = sum(b[2] for b in blobs)
    hist_disk = sum(b[3] for b in blobs)
    gone = [b for b in blobs if b[1] not in head_shas]

    # churn: how many commits touched each binary path
    log = git('log', '--all', '--format=', '--name-only', '-z').decode('utf-8', 'surrogateescape')
    touches = collections.Counter(p.strip('\n') for p in log.split('\0') if p.strip('\n'))
    churn = sorted(((n, p) for p, n in touches.items() if n >= 2 and ext(p) in BINARY_EXT), reverse=True)
    n_commits = int(git('rev-list', '--all', '--count').decode())

    ign = [p for p in git('ls-files', '-z', '-c', '-i', '--exclude-standard').decode('utf-8', 'surrogateescape')
           .split('\0') if p]
    sizes = {p: s for p, _, s in tree}
    ign_total = sum(sizes.get(p, 0) for p in ign)
    spaces = [p for p, *_ in tree if ' ' in p]
    eol = git('ls-files', '--eol').decode('utf-8', 'surrogateescape').splitlines()
    crlf = [line.split('\t', 1)[1] for line in eol if line.startswith('i/crlf')]

    out = []
    w = out.append
    w(f'## Raw numbers at HEAD {head} ({n_commits} commits)\n')
    w(f'- Tracked files: **{len(tree)}**, **{mb(total)} MiB** in total.')
    w(f'- All blobs in history: {len(blobs)}, {mb(hist_raw)} MiB uncompressed, {mb(hist_disk)} MiB in the pack.')
    w(f'- Blobs in history that are not at HEAD (old versions): {len(gone)}, {mb(sum(b[2] for b in gone))} MiB '
      f'uncompressed, {mb(sum(b[3] for b in gone))} MiB in the pack.')
    w(f'- Tracked files that `.gitignore` would ignore: {len(ign)}, {mb(ign_total)} MiB.')
    w(f'- File names with spaces: {len(spaces)}. Files committed with CRLF line ends: {len(crlf)}.\n')
    w('`git count-objects -vH`:\n')
    w('```\n' + '\n'.join(count_objects) + '\n```\n')
    w('### By top-level folder\n')
    w('| folder | files | MiB |\n|---|---:|---:|')
    for d, s in by_dir.most_common():
        w(f'| `{d}` | {n_dir[d]} | {mb(s)} |')
    w('\n### By extension (top 15)\n')
    w('| ext | files | MiB |\n|---|---:|---:|')
    for e, s in by_ext.most_common(15):
        w(f'| `{e}` | {n_ext[e]} | {mb(s)} |')
    w(f'\n### Largest tracked files (top {args.top})\n')
    w('| MiB | path | ignored by .gitignore? |\n|---:|---|---|')
    igs = set(ign)
    for p, _, s in sorted(tree, key=lambda t: -t[2])[:args.top]:
        w(f'| {mb(s)} | `{p}` | {"yes" if p in igs else ""} |')
    w(f'\n### Largest blobs in history that are no longer at HEAD (top 10)\n')
    w('| MiB raw | MiB packed | path |\n|---:|---:|---|')
    for name, _, s, d in sorted(gone, key=lambda b: -b[2])[:10]:
        w(f'| {mb(s)} | {mb(d)} | `{name}` |')
    w('\n### Binary paths changed in 2 or more commits (churn)\n')
    w('| commits | path |\n|---:|---|')
    for n, p in churn[:25]:
        w(f'| {n} | `{p}` |')
    if args.ignored:
        w('\n### Git-ignored paths on disk (du)\n')
        w('| MiB | path |\n|---:|---|')
        st = git('status', '--ignored', '--porcelain', '-z').decode('utf-8', 'surrogateescape').split('\0')
        rows = []
        for rec in st:
            if rec.startswith('!! '):
                p = rec[3:]
                du = subprocess.run(['du', '-sk', p], capture_output=True, text=True).stdout.split()
                if du:
                    rows.append((int(du[0]) * 1024, p))
        for s, p in sorted(rows, reverse=True)[:25]:
            w(f'| {mb(s)} | `{p}` |')
        w(f'\nTotal git-ignored on disk: {mb(sum(s for s, _ in rows))} MiB in {len(rows)} paths.')
    print('\n'.join(out))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(dict(head=head, commits=n_commits, tracked_files=len(tree), tracked_bytes=total,
                           history_blobs=len(blobs), history_raw_bytes=hist_raw, history_pack_bytes=hist_disk,
                           gone_blobs=len(gone), tracked_but_ignored=ign, names_with_spaces=spaces, crlf=crlf,
                           by_dir=dict(by_dir), by_ext=dict(by_ext), churn=churn), f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
