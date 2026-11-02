"""Shared data + per-beat features for the D4 chord leaderboard (melody -> 7 diatonic triads).

All pitch classes are relative to the song tonic. One song = {'tonic','M','K','beats','sop'} as in
duet_beat/bench/data/*.json and duet_beat/data.py (Bach).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DB = os.path.join(REPO, 'duet_beat')
if DB not in sys.path:
    sys.path.insert(0, DB)

STYLES = ('bach', 'folk', 'pop', 'pop909')
NF = 12 + 12 + 1 + 4 + 1 + 4 + 4 + 1   # dur, onset, rest, pos4, downbeat, meter, style, is_long_note


def load_all():
    from data import load
    b = load()['split']
    out = {'bach': {k: b[k] for k in ('train', 'val', 'test')}}
    for n in ('folk', 'pop', 'pop909'):
        d = json.load(open(os.path.join(DB, 'bench', 'data', f'{n}.json')))
        out[n] = {k: d[k] for k in ('train', 'val', 'test')}
    return out


def dur_onset(song):
    """(K,12) time each relative pc sounds in each beat (beat fractions), (K,12) note starts."""
    K, t = song['K'], song['tonic']
    dur = np.zeros((K, 12))
    ons = np.zeros((K, 12))
    for on, off, m in song['sop']:
        pc = (m - t) % 12
        k0 = int(math.floor(on + 1e-6))
        if 0 <= k0 < K:
            ons[k0, pc] = 1.0
        for k in range(max(0, int(math.floor(on))), min(K, int(math.ceil(off)))):
            ov = min(off, k + 1) - max(on, k)
            if ov > 1e-9:
                dur[k, pc] += ov
    return dur, ons


def features(song, style):
    K, M = song['K'], song['M']
    dur, ons = dur_onset(song)
    X = np.zeros((K, NF), np.float32)
    X[:, :12] = np.minimum(dur, 1.0)
    X[:, 12:24] = ons
    X[:, 24] = dur.sum(1) < 1e-6
    pos = np.array([b['pos'] for b in song['beats']])
    X[np.arange(K), 25 + np.minimum(pos, 3)] = 1
    X[:, 29] = pos == 0
    X[:, 30 + {2: 0, 3: 1, 4: 2}.get(M, 3)] = 1
    X[:, 34 + STYLES.index(style)] = 1
    X[:, 38] = (dur.max(1) >= 0.99)
    return X


def labels(song):
    return np.array([b['z'] for b in song['beats']], int)
