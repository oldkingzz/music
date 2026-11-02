"""fit_trk_tab_E_v1 (run v5rt_1101, round 3, idea E): the live tracker's confusion as a lookup table for a tracker-aware
hear, fitted on TRAIN renders only (models/E_tracker_seq_v1.json from train_tracker_seq_E_v1.py: 135 train Bach
chorales, voices sung and sung_low pooled, every answered note in hear() order).

For every answered note the system can see (no truth): c1 = heard - previous answered pitch (clipped to -5..5), the onset
class g (snapped onset on a beat / on a half beat / on a quarter / not snapped) x dt class (time since the previous
answered onset < 0.35, < 0.75, >= 0.75 beat), and ret = the heard pitch equals the one two answered notes back. Target:
e = true - heard pitch of the true note the answered note belongs to (live_bench_rt.tracker_rows' matching), e in
-6..6. Also P(split fragment) (the answered note is the 2nd+ answered note of one true note).

Smoothed by backoff: P0(e) -> P1(e | c1) -> P2(e | c1, onset x dt class) -> P3(e | c1, onset x dt class, ret), each level
(N + a * parent) / (n + a), a = 4. The first answered note of a song has its own row. Written to
models/E_trk_tab_v1.npz: P3 [11, 12, 2, 13], PF3 [11, 12, 2] (P(fragment)), F [13] (first note), N3 counts, EV.

  .venv/bin/python runs_mini/v5rt_1101/r3/E/fit_trk_tab_E_v1.py --out runs_mini/v5rt_1101/models/E_trk_tab_v1.npz
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

SEQ = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/models/E_tracker_seq_v1.json'
MODELS = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/models'
EV = np.arange(-6, 7)
A = 4.0


def gidx(on):
    if abs(on * 4 - round(on * 4)) > 1e-6:
        return 3
    f = round((on % 1) * 4) % 4
    return (0, 2, 1, 2)[f]                     # beat 0, half 1, quarter 2, raw 3


def dtb3(dt):
    return 0 if dt < 0.35 else 1 if dt < 0.75 else 2


def feats(seq, j):
    """(c1b, gc, ret) of answered note j given the answered notes before it; None for the first."""
    if j == 0:
        return None
    on, h = seq[j]
    p_on, p1 = seq[j - 1]
    c1b = int(np.clip(h - p1, -5, 5)) + 5
    gc = gidx(on) * 3 + dtb3(on - p_on)
    ret = int(j >= 2 and h == seq[j - 2][1])
    return c1b, gc, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(MODELS + os.sep) or not path.endswith('.npz') or os.path.exists(path):
        sys.exit('--out must be a new .npz under ' + MODELS)
    d = json.load(open(SEQ))
    assert d['split'] == 'train', d['split']
    ix = {k: j for j, k in enumerate(d['fields'])}
    seqs = defaultdict(list)
    for r in d['rows']:
        seqs[(r[0], r[1])].append(r)
    N3 = np.zeros((11, 12, 2, len(EV)))
    NF3 = np.zeros((11, 12, 2))
    F = np.zeros(len(EV))
    n_skip = 0
    for key, rows in seqs.items():
        rows.sort(key=lambda r: r[ix['i']])
        seq = [(r[ix['onset']], r[ix['heard']]) for r in rows]
        for j, r in enumerate(rows):
            if r[ix['k']] < 0:
                continue
            e = r[ix['true_midi']] - r[ix['heard']]
            if abs(e) > 6:
                n_skip += 1
                continue
            fz = feats(seq, j)
            if fz is None:
                F[e + 6] += 1
                continue
            c1b, gc, ret = fz
            N3[c1b, gc, ret, e + 6] += 1
            NF3[c1b, gc, ret] += bool(r[ix['frag']])
    N2, N1, N0 = N3.sum(2), N3.sum((1, 2)), N3.sum((0, 1, 2))
    P0 = (N0 + 1.0) / (N0.sum() + len(EV))
    P1 = (N1 + A * P0[None]) / (N1.sum(-1, keepdims=True) + A)
    P2 = (N2 + A * P1[:, None]) / (N2.sum(-1, keepdims=True) + A)
    P3 = (N3 + A * P2[:, :, None]) / (N3.sum(-1, keepdims=True) + A)
    n3, n2, n1 = N3.sum(-1), N2.sum(-1), N1.sum(-1)
    f0 = NF3.sum() / n3.sum()
    f1 = (NF3.sum((1, 2)) + A * f0) / (n1 + A)
    f2 = (NF3.sum(2) + A * f1[:, None]) / (n2 + A)
    PF3 = (NF3 + A * f2[:, :, None]) / (n3 + A)
    Fp = (F + 1.0) / (F.sum() + len(EV))
    np.savez(path, P3=P3, PF3=PF3, F=Fp, N3=N3, NF3=NF3, EV=EV)
    print(f"train answered notes used {int(n3.sum() + F.sum())} (first {int(F.sum())}, |e| > 6 skipped {n_skip}); "
          f"P(right) overall {N0[6] / N0.sum():.3f}; fragments {NF3.sum() / n3.sum():.3f} -> {path}", flush=True)
    names = ['beat', 'half', 'q', 'raw']
    for c1b in range(11):
        row = []
        for gc in (2, 5, 1, 4, 0, 3, 6, 9, 10):
            pr = P3[c1b, gc, 0, 6]
            row.append(f"{names[gc // 3]}{gc % 3}:{pr:.2f}/{PF3[c1b, gc, 0]:.2f}")
        print(f"c1 {c1b - 5:+d} P(right)/P(frag): " + ' '.join(row), flush=True)


if __name__ == '__main__':
    main()
