"""Scores for one simulated song and pooled scores with song-level bootstrap (spec section 11).

Counted from beat J on (Duet may not sing before it has listened):
  M1  chord guessed right            z_hat == Bach's chord label
  M2  Duet's note is in Bach's chord  (triad label), over the beats Duet sings
  M2b Duet's note is among the pitch classes Bach actually sounds at that beat onset
  M3  consonant with her (unison/octave, 3rds, 5th, 6ths), weighted by how long they overlap
  M4  parallel fifths/octaves per 100 Duet moves
  M5  share of beats Duet sings
  U   net good notes per beat: (in-chord notes - out-of-chord notes) / beats   (criterion for q_min)
"""
import numpy as np

from theory import CONSONANT, CT

KEYS = ('m1', 'm2', 'm2b', 'm3', 'm5', 'u')


def song_counts(song, beats, J):
    t, sop = song['tonic'], song['sop']
    info = song['beats']
    c = dict(m1_n=0, m1_d=0, in_n=0, out_n=0, m2_d=0, m2b_n=0, m2b_d=0, m3_n=0.0, m3_d=0.0,
             par=0, moves=0, sung=0, rows=0)
    prev = None
    for b in beats:
        k = b.k
        if k < J:
            prev = None
            continue
        c['rows'] += 1
        zt = info[k]['z']
        if b.z_hat >= 0 and zt >= 0:
            c['m1_d'] += 1
            c['m1_n'] += int(b.z_hat == zt)
        if b.h < 0:
            prev = None
            continue
        c['sung'] += 1
        pc = (b.h - t) % 12
        if zt >= 0:
            c['m2_d'] += 1
            if pc in CT[zt]:
                c['in_n'] += 1
            else:
                c['out_n'] += 1
        ons = info[k]['onset_pcs']
        if ons:
            c['m2b_d'] += 1
            c['m2b_n'] += int(pc in ons)
        for on, off, m in sop:
            if on >= k + 1:
                break
            ov = min(off, k + 1) - max(on, k)
            if ov > 1e-9:
                c['m3_d'] += ov
                c['m3_n'] += ov * (abs(m - b.h) % 12 in CONSONANT)
        s_now = _sounding(sop, k)
        if prev is not None:
            hp, sp = prev
            if b.h != hp:
                c['moves'] += 1
                if sp is not None and s_now is not None and s_now != sp and np.sign(s_now - sp) == np.sign(b.h - hp):
                    a, z = (sp - hp) % 12, (s_now - b.h) % 12
                    if a == z and a in (0, 7):
                        c['par'] += 1
        prev = (b.h, s_now)
    return c


def _sounding(sop, k):
    for on, off, m in sop:
        if on <= k + 1e-6 < off:
            return m
        if on > k + 1e-6:
            return None
    return None


def summarize(cs):
    s = {k: sum(c[k] for c in cs) for k in cs[0]}
    div = lambda a, b: (a / b) if b else float('nan')  # noqa: E731
    return {
        'm1': div(s['m1_n'], s['m1_d']),
        'm2': div(s['in_n'], s['m2_d']),
        'm2b': div(s['m2b_n'], s['m2b_d']),
        'm3': div(s['m3_n'], s['m3_d']),
        'm4': div(100.0 * s['par'], s['moves']),
        'm5': div(s['sung'], s['rows']),
        'u': div(s['in_n'] - s['out_n'], s['rows']),
        'beats': s['rows'], 'sung': s['sung'],
    }


def bootstrap(cs, n=2000, seed=5160):
    """95 % intervals by resampling whole songs."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cs))
    draws = {k: [] for k in ('m1', 'm2', 'm2b', 'm3', 'm4', 'm5', 'u')}
    for _ in range(n):
        smp = summarize([cs[i] for i in rng.choice(idx, len(idx))])
        for k in draws:
            draws[k].append(smp[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def paired_diff(cs_a, cs_b, key, n=2000, seed=5160):
    """Difference a - b in one pooled metric, with a paired song-level bootstrap interval."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cs_a))
    d0 = summarize(cs_a)[key] - summarize(cs_b)[key]
    ds = []
    for _ in range(n):
        sel = rng.choice(idx, len(idx))
        ds.append(summarize([cs_a[i] for i in sel])[key] - summarize([cs_b[i] for i in sel])[key])
    return float(d0), [float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5))]
