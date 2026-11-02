#!/usr/bin/env python3
"""Repo health check for CI and for scripts/publish_private_repo.sh. Standard library only.

Checks (on the files git tracks, or on the files that would be committed with --staged):
  1. File size. ERROR above --max-mb (default 25 MiB; GitHub's browser-upload limit is 25 MiB,
     its push warning is 50 MiB and its hard block 100 MiB). WARN above --warn-mb (default 5 MiB).
     Git LFS pointer files are measured as pointers (a few hundred bytes).
  2. Total tracked size. WARN above --warn-total-mb (default 250), ERROR above --max-total-mb
     (default 1024; GitHub recommends repositories stay under 1 GB).
  3. Never-commit paths (licence or privacy): dataset audio, dataset folders, caches, secrets.
     These are ERRORs. The list is NEVER below and is mirrored in docs/COLLABORATION.md.
  4. Relative links in tracked Markdown files must point at a file or folder that exists in the
     repo. External (http, mailto) links are not fetched. Broken links are WARN by default and
     ERROR with --strict-links. Paths written in backticks (`dir/file.ext`) are also checked, as WARN
     only, because some of them deliberately name files outside the repo.
  5. Tracked files that the .gitignore rules would ignore (WARN: they were added before the rule
     or force-added, and test runs keep rewriting some of them).

Exit status: 0 if no ERROR, 1 otherwise. Use --json to get a machine-readable report.
Run from anywhere inside the repo:  python3 scripts/check_repo_health.py
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys

MIB = 1024 * 1024

# Paths that must never be committed. fnmatch patterns against the repo-relative path.
NEVER = [
    ('data/external/*', 'downloaded datasets (re-downloadable, 2.5 GB, dataset licences)'),
    ('eval/cache/*', 'evaluation cache (750 MB, contains per-take dataset features)'),
    ('eval/fresh_data/*', 'held-out dataset audio (DCS, CC BY-NC 4.0)'),
    ('data/raw/*', 'our own singer recordings (consent covers the team, not the public)'),
    ('*tebe_poem*.wav', 'Dagstuhl ChoirSet audio (CC BY-NC 4.0; project policy: course use only)'),
    ('*dcs_*.wav', 'Dagstuhl ChoirSet audio (CC BY-NC 4.0; project policy: course use only)'),
    ('*ecd_*.wav', 'ESMUC Choir Dataset audio (keep datasets out of git)'),
    ('*.env', 'environment file, may hold secrets'),
    ('*.pem', 'private key or certificate'),
    ('*.key', 'private key'),
    ('*id_rsa*', 'SSH key'),
    ('*id_ed25519*', 'SSH key'),
    ('*.p12', 'certificate bundle'),
    ('*credentials*.json', 'service-account or OAuth credentials'),
    ('*client_secret*.json', 'OAuth client secret'),
]

CODE_PATH_RE = re.compile(r'`([A-Za-z0-9_./-]+/[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,10})`')
LINK_RE = re.compile(r'(?<!\!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)|!\[[^\]\n]*\]\(([^)\s]+)\)')


def git(*args: str) -> bytes:
    return subprocess.run(['git', *args], check=True, capture_output=True).stdout


def repo_root() -> str:
    return git('rev-parse', '--show-toplevel').decode().strip()


def tracked(staged: bool) -> list[str]:
    out = git('ls-files', '-z') if not staged else git('ls-files', '-z', '--cached')
    return [p for p in out.decode('utf-8', 'surrogateescape').split('\0') if p]


def tracked_but_ignored() -> list[str]:
    out = git('ls-files', '-z', '-c', '-i', '--exclude-standard')
    return [p for p in out.decode('utf-8', 'surrogateescape').split('\0') if p]


def is_lfs_pointer(path: str, size: int) -> bool:
    if size > 1024:
        return False
    try:
        with open(path, 'rb') as f:
            return f.read(40).startswith(b'version https://git-lfs.github.com/spec/')
    except OSError:
        return False


def md_links(text: str):
    # skip fenced code blocks and inline code
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    text = re.sub(r'`[^`\n]*`', '', text)
    for m in LINK_RE.finditer(text):
        yield m.group(1) or m.group(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max-mb', type=float, default=25.0)
    ap.add_argument('--warn-mb', type=float, default=5.0)
    ap.add_argument('--max-total-mb', type=float, default=1024.0)
    ap.add_argument('--warn-total-mb', type=float, default=250.0)
    ap.add_argument('--strict-links', action='store_true', help='broken relative links are errors')
    ap.add_argument('--staged', action='store_true', help='check the index (what the next commit holds)')
    ap.add_argument('--json', help='write the full report to this path')
    ap.add_argument('--quiet', action='store_true', help='print only the summary and errors')
    args = ap.parse_args()

    root = repo_root()
    os.chdir(root)
    files = tracked(args.staged)
    fileset = set(files)
    dirs = {os.path.dirname(p) for p in files}
    for d in list(dirs):
        while d:
            d = os.path.dirname(d)
            dirs.add(d)

    errors: list[str] = []
    warns: list[str] = []
    total = 0
    sizes = {}
    for p in files:
        try:
            s = os.path.getsize(p)
        except OSError:
            continue  # deleted in the work tree but still in the index
        sizes[p] = s
        total += s
        if is_lfs_pointer(p, s):
            continue
        if s > args.max_mb * MIB:
            errors.append(f'size: {p} is {s / MIB:.1f} MiB (> {args.max_mb:g} MiB); move it to the shared drive or Git LFS')
        elif s > args.warn_mb * MIB:
            warns.append(f'size: {p} is {s / MIB:.1f} MiB (> {args.warn_mb:g} MiB)')

    if total > args.max_total_mb * MIB:
        errors.append(f'total: tracked files sum to {total / MIB:.1f} MiB (> {args.max_total_mb:g} MiB)')
    elif total > args.warn_total_mb * MIB:
        warns.append(f'total: tracked files sum to {total / MIB:.1f} MiB (> {args.warn_total_mb:g} MiB)')

    for p in files:
        for pat, why in NEVER:
            if fnmatch.fnmatch(p, pat) or fnmatch.fnmatch(os.path.basename(p), pat):
                errors.append(f'never-commit: {p} ({why})')
                break

    broken = []
    n_links = 0
    for p in files:
        if not p.lower().endswith('.md') or p not in sizes:
            continue
        try:
            text = open(p, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        for link in md_links(text):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', link) or link.startswith('#'):
                continue  # http:, https:, mailto:, or an in-page anchor
            n_links += 1
            target = link.split('#', 1)[0].split('?', 1)[0]
            target = target.replace('%20', ' ')
            if not target:
                continue
            rel = os.path.normpath(os.path.join(os.path.dirname(p), target)) if not target.startswith('/') \
                else os.path.normpath(target.lstrip('/'))
            if rel in fileset or rel in dirs:
                continue
            broken.append(f'link: {p} -> {link}')
    (errors if args.strict_links else warns).extend(broken)

    # Backtick path mentions such as `media/figures/fig_a.png` (warn only: some point outside the repo).
    n_mentions = 0
    for p in files:
        if not p.lower().endswith('.md') or p not in sizes:
            continue
        text = open(p, encoding='utf-8', errors='replace').read()
        for m in CODE_PATH_RE.finditer(text):
            x = m.group(1)
            n_mentions += 1
            cands = {os.path.normpath(x), os.path.normpath(os.path.join(os.path.dirname(p), x))}
            if any(c in fileset or c in dirs or os.path.exists(c) for c in cands):
                continue
            warns.append(f'path-mention: {p} mentions `{x}`, which is not in the repo')

    ign = tracked_but_ignored()
    if ign:
        ign_mb = sum(sizes.get(p, 0) for p in ign) / MIB
        warns.append(f'ignored-but-tracked: {len(ign)} files ({ign_mb:.1f} MiB) match .gitignore rules '
                     f'but are tracked, e.g. {ign[0]}')

    print(f'repo health: {len(files)} tracked files, {total / MIB:.1f} MiB; {n_links} relative Markdown links '
          f'checked, {len(broken)} broken; {n_mentions} backtick paths checked; {len(errors)} errors, '
          f'{len(warns)} warnings')
    if not args.quiet:
        for w in warns:
            print('WARN  ' + w)
    for e in errors:
        print('ERROR ' + e)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(dict(files=len(files), total_bytes=total, links_checked=n_links, broken_links=broken,
                           tracked_but_ignored=ign, errors=errors, warnings=warns), f, indent=1)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
