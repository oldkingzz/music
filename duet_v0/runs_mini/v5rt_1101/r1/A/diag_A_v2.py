"""Idea A diagnostic v2 (run v5rt_1101, round 1). VAL ONLY. Same as diag_A_v1.py (imports its Logged and integ) but
each decision's effect is integrated only until the next hear point (her next onset + L) or Duet's next change,
whichever comes first: v1 integrated up to Duet's next change only, which credited a plan with time the reactive
choice would have covered anyway. Also prints the whole-run counted-time split (first L / rest) for the setting.

  python diag_A_v2.py '<json overrides>' [--sub N] [--L 0.1]
"""
import argparse
import bisect
import json
import os
import sys
from collections import defaultdict

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'A'))
from diag_A_v1 import B, BASE, Logged, core, integ, r2_mix_v2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg')
    ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--L', type=float, default=0.1)
    a = ap.parse_args()
    B.bench3.B2.heard = B.HEARD1
    train = B.bench3.train_sets()
    obj = Logged(r2_mix_v2.build(train), core.Prior(), dict(BASE, **json.loads(a.cfg)))
    L = a.L
    for ds, songs in B.bench3.eval_sets('val').items():
        if a.sub:
            songs = songs[:a.sub]
        agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0])  # n, hit_new, hit_old, harsh_new, harsh_old, fresh
        tot = defaultdict(float)
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, _, _ = B.simulate(obj, meta, notes, L)
            acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                                 'on_time_right', 'changes', 'late_entry'), 0.0)
            B.score_song(s, T, H, L, acc)
            for k, v in acc.items():
                tot[k] += v
            true = sorted(s['sop'], key=lambda n: n[0])
            hps = [o + L for o, _, _ in notes]
            for kind, x, hnew, hold, onset in obj.log:
                t0 = x if kind == 'plan' else onset + L
                j = bisect.bisect_right(T, t0)
                t1 = T[j] if j < len(T) else s['K']
                jh = bisect.bisect_right(hps, t0 + 1e-12)
                if jh < len(hps):
                    t1 = min(t1, hps[jh])
                hn, hh = integ(s, t0, t1, hnew, L)
                ho, ho_h = integ(s, t0, t1, hold, L)
                fresh = False
                if kind == 'plan':
                    near = [n for n in true if abs(n[0] - x) < 0.01 or (n[0] <= x < n[1])]
                    if any(abs(n[0] - x) < 0.01 for n in near):
                        cat = 'onset'
                    elif near:
                        cat = 'hold'
                    else:
                        cat = 'rest'
                    sound = [n for n in true if n[0] <= x < n[1]]
                    fresh = bool(sound) and hnew >= 0 and abs(sound[-1][2] - hnew) % 12 in B.HARSH and x >= B.J
                    cat = f"plan_{cat}{'_sil' if hnew < 0 else ''}"
                else:
                    cat = 'hear_keep'
                r = agg[cat]
                r[0] += 1
                r[1] += hn
                r[2] += ho
                r[3] += hh
                r[4] += ho_h
                r[5] += fresh
        d, n_notes = tot['d'], tot['notes']
        print(f"{ds}: rt_nd {tot['hit'] / d:.4f} harsh {tot['harsh'] / d:.4f}; first-L share {tot['d_first'] / d:.3f} "
              f"hit {tot['hit_first'] / tot['d_first']:.3f} harsh {tot['harsh_first'] / tot['d_first']:.3f}; rest hit "
              f"{(tot['hit'] - tot['hit_first']) / (d - tot['d_first']):.3f} harsh "
              f"{(tot['harsh'] - tot['harsh_first']) / (d - tot['d_first']):.3f}; notes {int(n_notes)}", flush=True)
        for cat, r in sorted(agg.items()):
            print(f'   {cat:16s} n {r[0]:6d} ({100 * r[0] / max(n_notes, 1):5.1f} per 100 notes)  '
                  f'hit gain {(r[1] - r[2]) / d:+.4f} of counted time  harsh change {(r[3] - r[4]) / d:+.4f}  '
                  f'fresh clashes {100 * r[5] / max(n_notes, 1):.2f} per 100 notes', flush=True)


if __name__ == '__main__':
    main()
