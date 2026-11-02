"""Idea A prior v2 (run v5rt_1101, round 1), TRAIN only. Writes runs_mini/v5rt_1101/models/A_prior_v2.npz (refuses to
overwrite). The pitch table is v1's (copied from A_prior_v1.npz, fitted by train_prior_A_v1.py). The rhythm table
adds a 6th outcome:
  cov_beat  beats her last heard note sounds inside [q, min(int(q) + 1, first hear point after q)): a note chosen at q
            for her held note can stay to the end of beat int(q) (the chord does not change inside a beat), while
            cov_hold (v1) stops at q + 0.5.
Everything else (features, back-off, dataset balance) as v1.
"""
import bisect
import os
import sys

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'A'))
import train_prior_A_v1 as V1  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'models', 'A_prior_v2.npz')
SRC = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'models', 'A_prior_v1.npz')
L, EPS, A = V1.L, V1.EPS, V1.A


def rhythm_rows(song):
    notes = sorted(((float(on), float(off)) for on, off, m in song['sop']), key=lambda n: n[0])
    ons = [n[0] for n in notes]
    hps = [o + L for o in ons]
    K, M = song['K'], song['M']
    rows = []
    q = 0.0
    while q < K:
        c = bisect.bisect_right(hps, q) - 1
        if c >= 0:
            on_c, off_c = notes[c]
            k = int(q + 1e-9)
            s_b = min(48, int(round((q - on_c) * 12)))
            rest_b = 0 if off_c + L > q else 1 + min(3, int((q - off_c) / 0.5))
            phase = 0 if abs(q - k) < 1e-9 else 1
            posb = V1.tkey(song['beats'][k]['pos'] if k < len(song['beats']) else k % M, M)
            ioi_b = 49 if c == 0 else min(48, int(round((on_c - notes[c - 1][0]) * 12)))
            u = c + 1 if c + 1 < len(notes) else None
            att_new = att_held = cov_new = cov_next = 0.0
            if u is not None:
                on_u, off_u = notes[u]
                if on_u < q + EPS and off_u > q:
                    att_new = 1.0
                e_u = min(off_u, on_u + L)
                cov_new = max(0.0, min(e_u, q + 0.5) - max(on_u, q))
                cov_next = max(0.0, min(e_u, q + 0.5 + L) - max(on_u, q + 0.5))
                hear_u = on_u + L
            else:
                hear_u = float('inf')
            if not att_new and on_c <= q < off_c:
                att_held = 1.0
            cov_hold = max(0.0, min(off_c, q + 0.5, hear_u) - q)
            cov_beat = max(0.0, min(off_c, k + 1, hear_u) - q)
            rows.append((s_b, rest_b, phase, posb, ioi_b, att_new, att_held, cov_new, cov_hold, cov_next, cov_beat))
        q += 0.5
    return rows


def main():
    if os.path.exists(OUT):
        sys.exit(f'refusing to overwrite {OUT}')
    src = np.load(SRC)
    tr = V1.bench3.train_sets()
    nn = {ds: sum(len(s['sop']) for s in songs) for ds, songs in tr.items()}
    tot = sum(nn.values())
    wds = {ds: tot / len(nn) / n for ds, n in nn.items()}
    shape = (49, 5, 2, 6, 50)
    F = 6
    Sw = np.zeros(shape + (F,))
    Sn = np.zeros(shape)
    Ww = np.zeros(shape)
    for ds, songs in tr.items():
        for s in songs:
            for row in rhythm_rows(s):
                key = row[:5]
                Sw[key] += wds[ds] * np.array(row[5:])
                Ww[key] += wds[ds]
                Sn[key] += 1.0
    G0 = Sw.sum((0, 1, 2, 3, 4)) / Ww.sum()

    def lvl(axes_keep, parent):
        ax = tuple(i for i in range(5) if i not in axes_keep)
        sw, ww, sn = Sw.sum(ax), Ww.sum(ax), Sn.sum(ax)
        mean = sw / np.maximum(ww, 1e-12)[..., None]
        lam = (sn / (sn + A))[..., None]
        return lam * mean + (1 - lam) * parent

    R1 = lvl((0, 1), G0[None, None, :])
    R2 = lvl((0, 1, 2), R1[:, :, None, :])
    R3 = lvl((0, 1, 2, 3), R2[:, :, :, None, :])
    R4 = lvl((0, 1, 2, 3, 4), R3[:, :, :, :, None, :])
    print('rhythm v2: plan points', int(Sn.sum()), 'global means',
          dict(zip(('att_new', 'att_held', 'cov_new', 'cov_hold', 'cov_next', 'cov_beat'), np.round(G0, 4).tolist())),
          flush=True)
    with open(OUT, 'xb') as fh:
        np.savez(fh, mel=src['mel'], rhy=R4.astype(np.float32), mel_n=src['mel_n'], rhy_n=Sn.astype(np.float32),
                 L=np.float32(L), eps=np.float32(EPS), A=np.float32(A))
    print('wrote', OUT, os.path.getsize(OUT), 'bytes', flush=True)


if __name__ == '__main__':
    main()
