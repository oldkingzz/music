"""Idea B offline evaluator v6 (round 1, run v5rt_1101; val only, tuned on val): offline_d5 with a product grid
{"base": {...}, "vary": {"KEY": [values], ...}} and a compact summary sorted by rt_nd among the rules that keep
fresh clashes at or below reactive's (headline and every dataset within +1.0) and harsh at or below reactive's.

  python offline_d6.py <grid spec json> <new out json> [--beta 0] [--adapt 0] [--seed2] [--L 0.1]
"""
import argparse
import itertools
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'B'))
import offline_d4 as O4  # noqa: E402
import blib_d3 as BL3  # noqa: E402
from systems import r2_mix_v2  # noqa: E402

B = O4.B
DSN = O4.DSN


def expand(spec):
    keys = list(spec['vary'])
    out = []
    for vals in itertools.product(*(spec['vary'][k] for k in keys)):
        P = dict(spec['base'])
        P.update(dict(zip(keys, vals)))
        out.append(P)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('out')
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--adapt', type=float, default=0.0)
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--L', type=float, default=0.1)
    ap.add_argument('--top', type=int, default=25)
    a = ap.parse_args()
    spec = json.loads(a.spec) if a.spec.strip().startswith('{') else json.load(open(a.spec))
    grid = expand(spec)
    if os.path.exists(a.out):
        sys.exit('refusing to overwrite ' + a.out)
    heard = B.heard_seed2 if a.seed2 else B.HEARD1
    B.bench3.B2.heard = heard
    train = B.bench3.train_sets()
    pri = BL3.Priors(train)
    rec = O4.Rec(r2_mix_v2.build(train), BL3.Probe(pri, beta=a.beta, adapt=a.adapt))
    runs = {}
    t1 = time.time()
    for ds, songs in B.bench3.eval_sets('val').items():
        runs[ds] = []
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, up, uh = B.simulate(rec, meta, notes, a.L)
            runs[ds].append((s, rec.tabs, rec.hears, T, H))
    print(f'beta {a.beta} adapt {a.adapt} L {a.L} {"seed2" if a.seed2 else "seed1"}: pass {time.time() - t1:.0f}s, '
          f'{len(grid)} rules', flush=True)
    h0, r0 = O4.score_all(runs, None, a.L)
    res = [{'P': None, 'head': h0, 'rows': r0}]
    t2 = time.time()
    for P in grid:
        h, r = O4.score_all(runs, P, a.L)
        res.append({'P': P, 'head': h, 'rows': r})
    print(f'grid {time.time() - t2:.0f}s', flush=True)
    with open(a.out, 'x') as fh:
        json.dump({'seed': 'seed2' if a.seed2 else 'seed1', 'L': a.L, 'beta': a.beta, 'adapt': a.adapt,
                   'split': 'val', 'spec': spec, 'results': res}, fh)
    vk = list(spec['vary'])

    def ok(x):
        h, r = x['head'], x['rows']
        return (h['fresh100'] <= h0['fresh100'] and h['harsh'] <= h0['harsh']
                and all(r[d]['fresh100'] <= r0[d]['fresh100'] + 1.0 for d in DSN)
                and all(r[d]['harsh'] <= r0[d]['harsh'] + 0.005 for d in DSN)
                and all(r[d]['rt_nd'] >= r0[d]['rt_nd'] - 0.01 for d in DSN))
    good = sorted((x for x in res[1:] if ok(x)), key=lambda x: -x['head']['rt_nd'])
    print(f'{len(good)} of {len(grid)} rules pass the harsh / fresh / per-dataset checks; best by rt_nd:', flush=True)
    for x in good[:a.top]:
        h, r = x['head'], x['rows']
        print(' '.join(f'{k}={x["P"][k]}' for k in vk) + f" | d_rt {h['rt_nd'] - h0['rt_nd']:+.4f} d_harsh "
              f"{h['harsh'] - h0['harsh']:+.4f} fresh {h['fresh100']:.2f} moves {h['moves100']:.1f} alt "
              f"{h['alt100']:.1f} | " + ' '.join(f"{d} {r[d]['rt_nd'] - r0[d]['rt_nd']:+.4f}/"
                                                 f"{r[d]['fresh100']:.2f}" for d in DSN), flush=True)
    best_all = sorted(res[1:], key=lambda x: -x['head']['rt_nd'])[:5]
    print('best by rt_nd regardless of checks:', flush=True)
    for x in best_all:
        h = x['head']
        print(' '.join(f'{k}={x["P"][k]}' for k in vk) + f" | d_rt {h['rt_nd'] - h0['rt_nd']:+.4f} d_harsh "
              f"{h['harsh'] - h0['harsh']:+.4f} fresh {h['fresh100']:.2f}", flush=True)


if __name__ == '__main__':
    main()
