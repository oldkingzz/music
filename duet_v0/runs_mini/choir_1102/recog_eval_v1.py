"""How many sung notes does it take to tell which song she is singing, as the library grows? (run choir_1102;
Vincent 2026-11-02: "after the library grows, can you tell which song from a few notes I sing?" - offline, no page change)

Libraries (melodies from the repo's bench data, train + val; titles never printed): POP909 (Chinese pop, full songs),
folk (ESMUC / Essen-style tunes), pop (Hooktheory sections), plus Twinkle and v11.
A simulated singer sings the first n DISTINCT notes of a song from its beginning (repeats merged): her intervals are the
written ones times a compression s (narrow singing, like Vincent's 0.76-0.85) plus pitch noise; 'vincent' also drops
each note with 10 % chance (the tracker misses it). Any key (only intervals are used).
Matcher (vectorised over the library): cost of her n-1 intervals against a song's first intervals, for compressions
s in 0.70..1.05, sum of min(1, ((her - s * written) / 1.0)^2); also every variant with one written note skipped
(a note she missed, +0.5). The song with the lowest cost wins; ties share the credit. Top-1 and top-3 accuracy per n.
Songs whose first 13 distinct notes are identical to another song's are counted once (duplicates in the data).

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/recog_eval_v1.py
"""
import json
import os

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
OUT = os.path.join(ABS, 'runs_mini/choir_1102/recog_eval_v1.json')
NMAX = 13
rng = np.random.default_rng(20261102)


def distinct(sop):
    out = []
    for n in sop:
        m = int(round(n[2]))
        if not out or out[-1] != m:
            out.append(m)
    return out


def load(name):
    raw = json.load(open(os.path.join(ABS, 'duet_beat/bench/data', name + '.json')))
    return [distinct(s['sop']) for part in ('train', 'val') for s in raw[part]]


lib_parts = {'pop909': load('pop909'), 'folk': load('folk'), 'pop': load('pop')}
choir = json.load(open(os.path.join(ABS, 'futures/v4_selfiter/choir_songs_v5.json')))['songs']
lib_parts['choir'] = [distinct(s['melody']) for s in choir.values()]


def build(names):
    seqs, seen = [], set()
    for nm in names:
        for d in lib_parts[nm]:
            if len(d) < NMAX:
                continue
            key = tuple(np.diff(d[:NMAX]))
            if key in seen:
                continue
            seen.add(key)
            seqs.append(d[:NMAX])
    return np.array(seqs, dtype=float)


S_GRID = np.arange(0.70, 1.051, 0.05)


def costs(her_iv, lib):
    """her_iv: (n-1,) intervals; lib: (N, NMAX) pitches. Returns (N,) best cost over compressions and one skip."""
    n1 = len(her_iv)
    best = np.full(len(lib), np.inf)
    variants = [lib[:, :n1 + 1]]                            # no skip
    pens = [0.0]
    for d in range(1, n1 + 1):                              # skip written note d (she missed it)
        variants.append(np.concatenate([lib[:, :d], lib[:, d + 1:n1 + 2]], axis=1))
        pens.append(0.5)
    for v, pen in zip(variants, pens):
        iv = np.diff(v, axis=1)                             # (N, n-1)
        for s in S_GRID:
            c = np.minimum(1.0, ((her_iv[None, :] - s * iv) / 1.0) ** 2).sum(1) + pen
            best = np.minimum(best, c)
    return best


def sing(d, n, style):
    """Her pitches for the first n distinct written notes d (any key)."""
    if style == 'clean':
        s, noise, drop = rng.uniform(0.95, 1.0), 0.15, 0.0
    else:
        s, noise, drop = rng.uniform(0.75, 0.95), 0.35, 0.10
    w = np.array(d[:n + 2], dtype=float)
    keep = [0] + [i for i in range(1, len(w)) if rng.random() >= drop]
    w = w[keep][:n]
    if len(w) < n:
        return None
    return w[0] + s * (w - w[0]) + rng.normal(0.0, noise, len(w))


def evaluate(lib, n_queries=300):
    res = {}
    idx = rng.choice(len(lib), size=min(n_queries, len(lib)), replace=False)
    for style in ('clean', 'vincent'):
        for n in (3, 4, 5, 6, 8, 10, 12):
            top1 = top3 = tot = 0.0
            for q in idx:
                her = sing(list(lib[q]), n, style)
                if her is None:
                    continue
                c = costs(np.diff(her), lib)
                mine = c[q]
                better = int(np.sum(c < mine - 1e-9))
                ties = int(np.sum(np.abs(c - mine) <= 1e-9))
                top1 += (1.0 / ties) if better == 0 else 0.0
                top3 += min(1.0, max(0, 3 - better) / ties) if better < 3 else 0.0
                tot += 1
            res[f'{style} n={n}'] = {'top1': round(top1 / tot, 3), 'top3': round(top3 / tot, 3)}
    return res


report = {}
for label, names in (('2 songs (now)', ['choir']), ('POP909', ['pop909']), ('POP909 + folk', ['pop909', 'folk']),
                     ('all (POP909 + folk + pop sections)', ['pop909', 'folk', 'pop'])):
    lib = build(names)
    r = evaluate(lib)
    report[label] = {'songs': len(lib), 'acc': r}
    print(f'{label}: {len(lib)} distinct openings', flush=True)
    for style in ('clean', 'vincent'):
        print('  ', style, '  '.join(f"n={k.split('=')[1]}: {v['top1']:.2f}/{v['top3']:.2f}"
                                    for k, v in r.items() if k.startswith(style)), flush=True)
with open(OUT, 'x') as fh:
    json.dump(report, fh, indent=1)
