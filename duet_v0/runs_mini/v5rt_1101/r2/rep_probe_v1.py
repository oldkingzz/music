"""R2 probe (run v5rt_1101, round 2): how predictable is her next note from REPEATS inside the same song?
Diagnostic only (no system). For every note j of hers, the earlier heard notes 0..j-1 are searched for passages whose
last n notes match the last n heard notes before j (same heard pitch, same inter-onset interval; up to one pitch
mismatch allowed, since 10 % of heard pitches are wrong). The continuations of the matching passages vote for note j's
pitch and inter-onset interval. Reports coverage and top-1 accuracy (against her TRUE pitch) by match length.
Val (or train) with bench3's seed-1 mishearing. Val only numbers are tuned-on-val diagnostics.

  python rep_probe_v1.py [--split val|train] [--max_songs N]
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402


def runs(hm, on):
    n = len(hm)
    ioi = np.array([-1.0] + [on[i] - on[i - 1] for i in range(1, n)])
    hm = np.asarray(hm)
    Ep = hm[:, None] == hm[None, :]
    Er = np.abs(ioi[:, None] - ioi[None, :]) < 1e-6
    R0 = np.zeros((n, n), dtype=np.int32)
    R1 = np.zeros((n, n), dtype=np.int32)
    for a in range(n):
        if a == 0:
            R0[a] = (Ep[a] & Er[a])
            R1[a] = Er[a]
            continue
        prev0 = np.concatenate(([0], R0[a - 1, :-1]))
        prev1 = np.concatenate(([0], R1[a - 1, :-1]))
        R0[a] = np.where(Ep[a] & Er[a], prev0 + 1, 0)
        R1[a] = np.where(Er[a], np.where(Ep[a], prev1 + 1, prev0 + 1), 0)
    return R0, R1, ioi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--max_songs', type=int, default=0)
    ap.add_argument('--nmin', type=int, default=3)
    a = ap.parse_args()
    sets = B.bench3.eval_sets('val') if a.split == 'val' else B.bench3.train_sets()
    for ds, songs in sets.items():
        if a.max_songs:
            songs = songs[:a.max_songs]
        st = defaultdict(Counter)
        for s in songs:
            hn = sorted(B.HEARD1(s)['sop'], key=lambda x: x[0])
            tn = sorted(s['sop'], key=lambda x: x[0])
            on = [float(x[0]) for x in hn]
            hm = [int(x[2]) for x in hn]
            tm = [int(x[2]) for x in tn]
            n = len(hn)
            if n < 4:
                continue
            R0, R1, ioi = runs(hm, on)
            for j in range(2, n):
                if on[j] < B.J:
                    continue
                a_ = j - 1
                r = R1[a_, :a_]                           # earlier passages ending at b < a_ (continuation b + 1 <= a_)
                best = int(r.max()) if len(r) else 0
                lb = 0 if best < a.nmin else (1 if best < 5 else (2 if best < 8 else 3))
                st[lb]['notes'] += 1
                if lb == 0:
                    continue
                vp, vi = Counter(), Counter()
                for b in np.nonzero(r >= max(a.nmin, best - 2))[0]:
                    w = 2.0 ** (r[b] - best)
                    vp[hm[b + 1]] += w
                    vi[round(on[b + 1] - on[b], 4)] += w
                p1, wp = vp.most_common(1)[0]
                i1, wi = vi.most_common(1)[0]
                tot = sum(vp.values())
                st[lb]['pitch_ok'] += p1 == tm[j]
                st[lb]['ioi_ok'] += abs(i1 - (on[j] - on[j - 1])) < 1e-3
                st[lb]['both_ok'] += (p1 == tm[j]) and abs(i1 - (on[j] - on[j - 1])) < 1e-3
                st[lb]['conf'] += wp / tot
                st[lb]['repeat_prev'] += tm[j] == hm[j - 1]
        tot = sum(v['notes'] for v in st.values())
        out = []
        for lb in sorted(st):
            v = st[lb]
            nn = v['notes']
            if lb == 0:
                out.append(f'no-match {nn / tot:.3f}')
            else:
                out.append(f"len{['', '3-4', '5-7', '8+'][lb]} cov {nn / tot:.3f} pitch {v['pitch_ok'] / nn:.3f} "
                           f"ioi {v['ioi_ok'] / nn:.3f} both {v['both_ok'] / nn:.3f} conf {v['conf'] / nn:.3f} "
                           f"rep_prev {v['repeat_prev'] / nn:.3f}")
        print(f'{ds} ({len(songs)} songs, {tot} notes): ' + ' | '.join(out), flush=True)


if __name__ == '__main__':
    main()
