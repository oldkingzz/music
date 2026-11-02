"""Idea H (run v5rt_1101, round 4): a LAG-AWARE rhythm table for the plan, TRAIN only (bench3.train_sets(): bach,
folk, pop, pop909). Writes runs_mini/v5rt_1101/models/H_lagrhy_v1.npz (refuses to overwrite). Val only for every
score; this bundle has no test split.

Same features, back-off and dataset balance as runs_mini/v5rt_1101/r1/A/train_prior_A_v2.py (W's rhythm table, fitted
with one lag L = 0.1 for both 'what the plan knows' and 'when the hear answer replaces the plan's note'). Here the two
lags are separate, as they are live with a large calibrated round trip (bt_cal: AirPods both ways):
  LI = 0.7 beat   information lag: at plan point q the plan knows her notes with onset + LI <= q (on the 0.25 grid:
                  q - onset >= 0.75; a note that began 0.5 beat before q is NOT known). bt_cal: the hear call comes
                  about (round trip + detect + PRE) = 0.65 beat after her onset; bench_rt L = 0.5 (plan before hear on
                  ties): q - onset >= 0.75 as well.
  LS = 0.55 beat  sounding lag: the hear answer for a note sounds LS after its onset (bt_cal ~0.58, bench_rt 0.5), so a
                  note the plan picks at q covers her NEW note from its onset until then (W's table assumed 0.1).
Rows: every plan point q (0.5 grid) after her first known note; c = her last KNOWN note. Outcomes (7):
  att_new   a note of hers (any after c) starts in [q, q + EPS) and sounds past q
  att_held  no note starts at q and c still sounds at q (on_c <= q < off_c)
  cov_new   beats in [q, q + 0.5) in which a note after c sounds before its hear answer (onset + LS)
  cov_hold  beats c sounds in [q, min(q + 0.5, first hear answer after q))
  cov_next  beats in [q + 0.5, q + 0.5 + LS) of notes after c before their hear answer (unused by the plan; kept for
            the same slot order as W's table)
  cov_beat  beats c sounds in [q, min(int(q) + 1, first hear answer after q))
  att_unh   a note after c started before q (not known yet) and still sounds at q, and no note starts at q
"""
import bisect
import os
import sys

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'models', 'H_lagrhy_v1.npz')
LI = 0.7
LS = 0.55
EPS = 0.01
A = 20.0                 # back-off strength (raw counts), as W's table


def tkey(pos, M):
    return int(pos) % 4 if M == 4 else (4 if int(pos) % M == 0 else 5)


def rhythm_rows(song):
    notes = sorted(((float(on), float(off)) for on, off, m in song['sop']), key=lambda n: n[0])
    ons = [n[0] for n in notes]
    kps = [o + LI for o in ons]
    K, M = song['K'], song['M']
    rows = []
    q = 0.0
    while q < K:
        c = bisect.bisect_right(kps, q + 1e-9) - 1
        if c >= 0:
            on_c, off_c = notes[c]
            k = int(q + 1e-9)
            s_b = min(48, int(round((q - on_c) * 12)))
            rest_b = 0 if off_c + LI > q + 1e-9 else 1 + min(3, int((q - off_c) / 0.5))
            phase = 0 if abs(q - k) < 1e-9 else 1
            posb = tkey(song['beats'][k]['pos'] if k < len(song['beats']) else k % M, M)
            ioi_b = 49 if c == 0 else min(48, int(round((on_c - notes[c - 1][0]) * 12)))
            att_new = att_held = att_unh = cov_new = cov_next = 0.0
            hear_u = float('inf')
            for u in range(c + 1, len(notes)):
                on_u, off_u = notes[u]
                if on_u >= q + 0.5 + LS:
                    break
                e_u = min(off_u, on_u + LS)
                cov_new += max(0.0, min(e_u, q + 0.5) - max(on_u, q))
                cov_next += max(0.0, min(e_u, q + 0.5 + LS) - max(on_u, q + 0.5))
                if hear_u == float('inf'):
                    hear_u = on_u + LS
                if q <= on_u < q + EPS and off_u > q:
                    att_new = 1.0
            if not att_new:
                for u in range(c + 1, len(notes)):
                    on_u, off_u = notes[u]
                    if on_u >= q:
                        break
                    if off_u > q:
                        att_unh = 1.0
                if not att_unh and on_c <= q < off_c:
                    att_held = 1.0
            cov_hold = max(0.0, min(off_c, q + 0.5, hear_u) - q)
            cov_beat = max(0.0, min(off_c, k + 1, hear_u) - q)
            rows.append((s_b, rest_b, phase, posb, ioi_b, att_new, att_held, cov_new, cov_hold, cov_next, cov_beat,
                         att_unh))
        q += 0.5
    return rows


def main():
    if os.path.exists(OUT):
        sys.exit(f'refusing to overwrite {OUT}')
    tr = bench3.train_sets()
    nn = {ds: sum(len(s['sop']) for s in songs) for ds, songs in tr.items()}
    tot = sum(nn.values())
    wds = {ds: tot / len(nn) / n for ds, n in nn.items()}
    print('train notes', nn, flush=True)
    shape = (49, 5, 2, 6, 50)
    F = 7
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
    names = ('att_new', 'att_held', 'cov_new', 'cov_hold', 'cov_next', 'cov_beat', 'att_unh')
    print('lag rhythm table: plan points', int(Sn.sum()), 'global means', dict(zip(names, np.round(G0, 4).tolist())),
          flush=True)
    for ds, songs in tr.items():
        rr = np.array([r[5:] for s in songs for r in rhythm_rows(s)])
        print(ds, 'rows', len(rr), 'means', dict(zip(names, np.round(rr.mean(0), 4).tolist())), flush=True)
    with open(OUT, 'xb') as fh:
        np.savez(fh, rhy=R4.astype(np.float32), rhy_n=Sn.astype(np.float32), LI=np.float32(LI), LS=np.float32(LS),
                 eps=np.float32(EPS), A=np.float32(A))
    print('wrote', OUT, os.path.getsize(OUT), 'bytes', flush=True)


if __name__ == '__main__':
    main()
