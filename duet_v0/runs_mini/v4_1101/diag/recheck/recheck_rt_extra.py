"""Extra context for recheck_rt.py (VAL only; no test split exists). Not part of the requested definitions.

  a) base rates: share of the rt_acc notes whose true chord is I, and of the most common chord (per dataset)
  b) beat-level accuracy over labelled beats k >= 4: forward filter at end of beat (alpha) vs forward-backward (gamma)
  c) m1 fed the CLEAN (true) melody instead of the heard one: rt_acc and music_nd (what the 10 % mishearing costs)
  d) label change rate per beat in val, and m1's transition self-probability per bar-position key
Writes one NEW json next to this script (open mode 'x') and prints a summary.
"""
import importlib
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
_bad = sorted(k for k in os.environ if k.startswith(('M1_', 'M3_')) or k == 'DUET_HIST')
if _bad:
    sys.exit(f'refusing to run: environment has {_bad}')

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))

import numpy as np  # noqa: E402

import bench3  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402

M1 = importlib.import_module('systems.m1_filter_v5lo')
J = bench3.J
S = M1.S
LAM = M1.P['lam']


class RecFilter(M1.Filter):
    def start(self, meta):
        super().start(meta)
        self.rec = []

    def belief(self, k):
        b = super().belief(k)
        self.rec.append((k, np.array(b, dtype=float, copy=True)))
        return b


def run_clean(system, song):
    """bench3.run_song, but fed her TRUE melody (no mishearing)."""
    notes = sorted(song['sop'], key=lambda n: n[0])
    system.start({'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'pos': [b['pos'] for b in song['beats']]})
    out, prev = [], None
    for i, (on, off, m) in enumerate(notes):
        out.append(int(system.step(i, float(on), int(m), None if prev is None else float(prev[1] - prev[0]))))
        prev = (on, off)
    return out


def evidence(model, sop, tonic, K):
    rel = [[on, off, (m - tonic) % 12] for on, off, m in sop]
    obs = M1.beat_obs(rel, K)
    E = np.zeros((K, S))
    for k in range(K):
        if obs[k].sum() > 0:
            E[k] = LAM * (model.logB @ obs[k])
    return E, obs


def forward_backward(model, E, pos, M):
    K = len(E)
    L = np.exp(E - E.max(1, keepdims=True))
    T = [None] + [model.A[M1.tkey(pos[k], M)] for k in range(1, K)]
    al = np.zeros((K, S))
    a = model.pi * L[0]
    al[0] = a / a.sum()
    for k in range(1, K):
        a = (al[k - 1] @ T[k]) * L[k]
        al[k] = a / a.sum()
    be = np.ones((K, S))
    for k in range(K - 2, -1, -1):
        b = T[k + 1] @ (L[k + 1] * be[k + 1])
        be[k] = b / b.sum()
    g = al * be
    g /= g.sum(1, keepdims=True)
    return g, al


def nd_add(acc, song, hs):
    true = sorted(song['sop'], key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    t = song['tonic']
    for i, (on, off, m) in enumerate(true):
        h = hs[i]
        for k in range(max(J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            acc[0] += ov
            if h >= 0:
                iv = abs(m - h) % 12
                acc[1] += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])


def main():
    t0 = time.time()
    base = M1.build(bench3.train_sets())
    model = base.m
    rec = RecFilter(model)
    out = {'per_ds': {}}
    for ds, songs in bench3.eval_sets('val').items():
        cnt = np.zeros(S)
        n = rt_clean = 0
        nd_clean = [0.0, 0.0]
        bt = dict(n=0, filt=0, smooth=0, n_snd=0, filt_snd=0, smooth_snd=0)
        ch = dict(pairs=0, changes=0)
        for s in songs:
            z = [b['z'] for b in s['beats']]
            pos = [b['pos'] for b in s['beats']]
            K = len(z)
            hsong = bench3.B2.heard(s)
            E, obs = evidence(model, hsong['sop'], s['tonic'], K)
            g, al = forward_backward(model, E, pos, s['M'])
            for k in range(J, K):
                if z[k] < 0:
                    continue
                bt['n'] += 1
                bt['filt'] += int(np.argmax(al[k])) == z[k]
                bt['smooth'] += int(np.argmax(g[k])) == z[k]
                if obs[k].sum() > 0:
                    bt['n_snd'] += 1
                    bt['filt_snd'] += int(np.argmax(al[k])) == z[k]
                    bt['smooth_snd'] += int(np.argmax(g[k])) == z[k]
            for k in range(max(J, 1), K):
                if z[k] >= 0 and z[k - 1] >= 0:
                    ch['pairs'] += 1
                    ch['changes'] += z[k] != z[k - 1]
            hs_clean = run_clean(rec, s)
            nd_add(nd_clean, s, hs_clean)
            tnotes = sorted(s['sop'], key=lambda x: x[0])
            for i, (on, off, m) in enumerate(tnotes):
                k = int(on)
                if k < J or k >= K or z[k] < 0:
                    continue
                kk, b = rec.rec[i]
                assert kk == k
                n += 1
                cnt[z[k]] += 1
                rt_clean += int(np.argmax(b)) == z[k]
        out['per_ds'][ds] = {
            'share_I': cnt[0] / n, 'share_most_common': float(cnt.max() / n), 'most_common': int(cnt.argmax()),
            'rt_acc_clean_melody': rt_clean / n, 'music_nd_clean_melody': nd_clean[1] / nd_clean[0],
            'beat_acc_filter_end_of_beat': bt['filt'] / bt['n'], 'beat_acc_smooth': bt['smooth'] / bt['n'],
            'beat_acc_filter_sounding': bt['filt_snd'] / bt['n_snd'], 'beat_acc_smooth_sounding': bt['smooth_snd'] / bt['n_snd'],
            'labelled_beats': bt['n'], 'chord_change_rate_per_beat': ch['changes'] / ch['pairs'], 'notes': n}
    P = out['per_ds']
    dss = list(P)
    keys = [k for k in P[dss[0]] if isinstance(P[dss[0]][k], float)]
    out['mean'] = {k: sum(P[d][k] for d in dss) / len(dss) for k in keys}
    out['A_self_prob'] = {str(t): float(np.mean(np.diag(model.A[t]))) for t in range(model.A.shape[0])}
    out['secs'] = time.time() - t0
    path = os.path.join(HERE, f"recheck_rt_extra_result_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('wrote', path)
    for k in keys:
        print(f"{k:30s} mean {out['mean'][k]:.4f} | " + ' '.join(f"{d} {P[d][k]:.4f}" for d in dss))
    print('most_common', {d: P[d]['most_common'] for d in dss}, 'labelled_beats', {d: P[d]['labelled_beats'] for d in dss})
    print('A_self_prob (mean diag per tkey 0-3: 4/4 bar pos, 4: downbeat other meters, 5: other beats)', out['A_self_prob'])
    print(f"secs {out['secs']:.1f}")


if __name__ == '__main__':
    main()
