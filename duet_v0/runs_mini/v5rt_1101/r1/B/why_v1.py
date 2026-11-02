"""Idea B: WHY r1_B_cons_v1 wins or loses against reactive r2_mix_v2 (val only, L = 0.1, seed1; diagnostic only, it
reads true pitches and chords after the fact). Uses offline_d4's exact replay (same as the scored system, checked:
bench_rt gives the same rt_nd and harsh) and splits the hit / harsh time difference, per dataset, into
  first-L window of each of her notes ([onset, onset + L)), by what Duet did at her onset:
    same      Duet sounds the same note as reactive just after her onset
    ontime    Duet moved early and kept that note after hearing her (one attack, on time)
    stumble   Duet moved early and changed again by the hear point (wrong-then-corrected)
    silent    Duet is silent just after her onset (reactive is not)
  rest of each of her notes ([onset + L, offset)):
    same      same note as reactive just after the hear point and no later change by Duet
    hearalt   the hear rule replaced r2_mix_v2's choice (probably misheard note)
    midmove   a plan point changed Duet's note later inside her note (mostly chord changes under a held note)
Each cell is (mine - reactive) hit time / counted time, so the cells of a dataset add up to its rt_nd delta.

  python why_v1.py <new out json>
"""
import bisect
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'B'))
import offline_d4 as O4  # noqa: E402
import blib_d3 as BL3  # noqa: E402
from systems import r2_mix_v2  # noqa: E402

B = O4.B
DSN = O4.DSN
L = 0.1
P = dict(LW=0.1, WH=0.4, LAM=1.0, MU=2.0, KAP=0.05, RHO=0.08, TAU=0.5, PMIN=0.0, BELOW=1, WS=0.0, SIL=0, KS=0.0,
         SUSP=0.7, LAMH=1.0, MUH=1.0, KSH=0.0, SILH=1, ROBH=1, MARGH=0.0)          # r1_B_cons_v1's RULE


def integ(T, H, on, off, k0, k1, m, t, z):
    """(hit time, harsh time) of Duet's step function against her true note m on [k0, k1) within beat-labelled time."""
    hit = harsh = 0.0
    for k in range(max(B.J, int(k0)), int(k1) + 1):
        a, b = max(k0, k), min(k1, k + 1)
        if b - a <= 1e-9 or k >= len(z) or z[k] < 0:
            continue
        for x0, x1, h in B.pieces(T, H, a, b):
            w = x1 - x0
            hit += w * B.hit(h, m, t, z[k])
            harsh += w * (h >= 0 and abs(m - h) % 12 in B.HARSH)
    return hit, harsh


def main():
    out_path = sys.argv[1]
    if os.path.exists(out_path):
        sys.exit('refusing to overwrite ' + out_path)
    B.bench3.B2.heard = B.HEARD1
    train = B.bench3.train_sets()
    rec = O4.Rec(r2_mix_v2.build(train), BL3.Probe(BL3.Priors(train), beta=0.0, adapt=4.0))
    res = {}
    for ds, songs in B.bench3.eval_sets('val').items():
        cells = {}
        counts = {}
        d_tot = 0.0
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, _, _ = B.simulate(rec, meta, notes, L)
            cnt = dict.fromkeys(('move', 'sil', 'keep', 'susp_alt'), 0)
            T2, H2 = O4.compose(s, rec.tabs, rec.hears, T, P, {'risk': 0.0, 'n': 0, 'harsh': 0}, cnt)
            true = sorted(s['sop'], key=lambda n: n[0])
            z = [b['z'] for b in s['beats']]
            t = s['tonic']
            for i, (on, off, m) in enumerate(true):
                for k in range(max(B.J, int(on)), int(off) + 1):
                    ov = min(off, k + 1) - max(on, k)
                    if ov > 1e-9 and k < len(z) and z[k] >= 0:
                        d_tot += ov
                if off <= B.J:
                    continue
                v_on, v_h = B.at(T2, H2, on), B.at(T2, H2, on + L)
                r_on, r_h = B.at(T, H, on), B.at(T, H, on + L)
                if v_on == r_on:
                    c1 = 'same'
                elif v_on < 0:
                    c1 = 'silent'
                elif v_on == v_h:
                    c1 = 'ontime'
                else:
                    c1 = 'stumble'
                e = min(off, on + L)
                a1, b1 = integ(T2, H2, on, off, on, e, m, t, z)
                a0, b0 = integ(T, H, on, off, on, e, m, t, z)
                key = 'first_' + c1
                c = cells.setdefault(key, [0.0, 0.0])
                c[0] += a1 - a0
                c[1] += b1 - b0
                counts[key] = counts.get(key, 0) + (on >= B.J)
                if off > on + L:
                    j0 = bisect.bisect_right(T2, on + L)
                    j1 = bisect.bisect_left(T2, off)
                    mid = any(H2[j] != H2[j - 1] for j in range(max(j0, 1), j1))
                    if v_h != r_h:
                        c2 = 'hearalt'
                    elif mid:
                        c2 = 'midmove'
                    else:
                        c2 = 'same'
                    a1, b1 = integ(T2, H2, on, off, on + L, off, m, t, z)
                    a0, b0 = integ(T, H, on, off, on + L, off, m, t, z)
                    key = 'rest_' + c2
                    c = cells.setdefault(key, [0.0, 0.0])
                    c[0] += a1 - a0
                    c[1] += b1 - b0
                    counts[key] = counts.get(key, 0) + (on >= B.J)
        res[ds] = {k: {'d_hit': v[0] / d_tot, 'd_harsh': v[1] / d_tot, 'notes': counts.get(k, 0)}
                   for k, v in sorted(cells.items())}
        tot_hit = sum(v['d_hit'] for v in res[ds].values())
        tot_harsh = sum(v['d_harsh'] for v in res[ds].values())
        print(f'{ds}: total d_rt_nd {tot_hit:+.4f} d_harsh {tot_harsh:+.4f}', flush=True)
        for k, v in res[ds].items():
            print(f"   {k:14s} notes {v['notes']:6d}  d_hit {v['d_hit']:+.4f}  d_harsh {v['d_harsh']:+.4f}", flush=True)
    keys = sorted({k for r in res.values() for k in r})
    print('headline (mean of datasets):', flush=True)
    for k in keys:
        print(f"   {k:14s} d_hit {sum(res[d].get(k, {'d_hit': 0})['d_hit'] for d in DSN) / 4:+.4f}  d_harsh "
              f"{sum(res[d].get(k, {'d_harsh': 0})['d_harsh'] for d in DSN) / 4:+.4f}", flush=True)
    with open(out_path, 'x') as fh:
        json.dump({'system': 'r1_B_cons_v1 (offline replay)', 'L': L, 'seed': 'seed1', 'split': 'val', 'P': P,
                   'rows': res}, fh, indent=1)


if __name__ == '__main__':
    main()
