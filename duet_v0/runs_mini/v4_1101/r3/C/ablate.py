"""One-off ablation for v4 round 3, implementer C (not a system, not a scored candidate): where does r3_style_mix_v3's gain
come from, the style-specific HMMs or the softer evidence weight lam 1.0 (m1's HMM uses lam 1.5)?

Configs are instances of r3_style_mix_v3.StyleMix (no class, module or file changed); a single pooled style is exactly m1's
HMM filter with evidence weight lam. music_nd / double computed exactly as rescore_nd.py on all val songs (tuned on val):
  A pooled lam 1.5, a 0.5   (must equal r2_mix_v1's recorded nd: sanity)
  B pooled lam 1.0, a 0.5   (lam only, no styles)
  C styles v1 lam 1.0, a 0.5 (must equal r3_style_mix_v3's recorded nd: sanity)
  D pooled lam 1.0, a 0.7   (lam only, at the champion's alpha)
  E styles v1 with pooled lam 1.5, pop/pop909 lam 1.0, a 0.5 (styles at m1's lam)
Writes runs_mini/v4_1101/r3/C/ablate_<time>.json (mode 'x').
"""
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402
from systems import m1_filter_v5lo, r3_style_mix_v3  # noqa: E402

OUTDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'r3', 'C')
NDDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'nd')
DS = ('bach', 'folk', 'pop', 'pop909')


def nd(system, sets):
    """music_nd and double per dataset, computed exactly as rescore_nd.py does (same loops, same order)."""
    out = {}
    for ds, songs in sets.items():
        num = den = dbl = 0.0
        for s in songs:
            hs = bench3.run_song(system, s)
            true = sorted(s['sop'], key=lambda n: n[0])
            z = [b['z'] for b in s['beats']]
            t = s['tonic']
            for i, (on, off, m) in enumerate(true):
                h = hs[i]
                for k in range(max(bench3.J, int(on)), int(off) + 1):
                    ov = min(off, k + 1) - max(on, k)
                    if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                        continue
                    den += ov
                    if h >= 0:
                        iv = abs(m - h) % 12
                        num += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])
                        dbl += ov * (iv == 0)
        out[ds] = {'music_nd': num / den, 'double': dbl / den}
    out['headline_nd'] = sum(v['music_nd'] for v in out.values()) / 4
    return out


def recorded(name):
    with open(os.path.join(NDDIR, name, 'results', f'nd_{name}.json')) as fh:
        return json.load(fh)


def main():
    t0 = time.time()
    p_before = dict(m1_filter_v5lo.P)
    train = bench3.train_sets()
    every = bench3.eval_sets('val')
    full = r3_style_mix_v3.build(train)          # styles v1 (pooled, pop, pop909), lam 1.0 each, a 0.5
    one = {M: v[:1] for M, v in full.mprior.items()}

    def make(models, lams, rh, mprior, a):
        x = r3_style_mix_v3.StyleMix(models, lams, rh, mprior, full.ens)
        x.a = a
        return x

    pooled = [full.models[0]]
    cfgs = (('A pooled lam 1.5, a 0.5', make(pooled, [1.5], full.rh[:1], one, 0.5), 'r2_mix_v1'),
            ('B pooled lam 1.0, a 0.5', make(pooled, [1.0], full.rh[:1], one, 0.5), None),
            ('C styles v1 lam 1.0, a 0.5', full, 'r3_style_mix_v3'),
            ('D pooled lam 1.0, a 0.7', make(pooled, [1.0], full.rh[:1], one, 0.7), None),
            ('E styles v1 pooled lam 1.5, a 0.5', make(full.models, [1.5, 1.0, 1.0], full.rh, full.mprior, 0.5), None))
    res = {'what': 'ablation of r3_style_mix_v3 (val, tuned on val); A and C are sanity reproductions', 'configs': {}}
    for label, system, ref in cfgs:
        got = nd(system, every)
        row = {d: {'music_nd': round(got[d]['music_nd'], 5), 'double': round(got[d]['double'], 5)} for d in DS}
        row['headline_nd'] = round(got['headline_nd'], 6)
        if ref:
            rec = recorded(ref)
            row['reproduces_' + ref] = all(got[d]['music_nd'] == rec[d]['music_nd'] and got[d]['double'] == rec[d]['double']
                                           for d in DS)
        res['configs'][label] = row
        print(label, json.dumps(row), flush=True)
    res['m1_P_unchanged'] = dict(m1_filter_v5lo.P) == p_before
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"ablate_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('m1_P_unchanged', res['m1_P_unchanged'], 'secs', res['secs'], 'wrote', path)


if __name__ == '__main__':
    main()
