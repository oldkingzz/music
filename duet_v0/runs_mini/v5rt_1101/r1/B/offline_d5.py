"""Idea B offline evaluator v5 (round 1, run v5rt_1101; val only, tuned on val): offline_d4's exact replay (its Rec,
plan_rule, hear_rule, compose and score_all, imported unchanged) with the probe of blib_d3 (chord-aware next pitch with
--beta, per-song adaptive onset prior with --adapt).

  python offline_d5.py <grid json or file> <new out json> [--beta 0] [--adapt 0] [--seed2] [--L 0.1]
"""
import argparse
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'B'))
import offline_d4 as O4  # noqa: E402  (bench_rt, attack_metric, the v3 systems path)
import blib_d3 as BL3  # noqa: E402
from systems import r2_mix_v2  # noqa: E402

B = O4.B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('grid')
    ap.add_argument('out')
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--adapt', type=float, default=0.0)
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--L', type=float, default=0.1)
    a = ap.parse_args()
    grid = json.loads(a.grid) if a.grid.strip().startswith('[') else json.load(open(a.grid))
    if os.path.exists(a.out):
        sys.exit('refusing to overwrite ' + a.out)
    heard = B.heard_seed2 if a.seed2 else B.HEARD1
    B.bench3.B2.heard = heard
    t0 = time.time()
    train = B.bench3.train_sets()
    pri = BL3.Priors(train)
    print(f'priors {time.time() - t0:.1f}s beta {a.beta} adapt {a.adapt} L {a.L} {"seed2" if a.seed2 else "seed1"}',
          flush=True)
    rec = O4.Rec(r2_mix_v2.build(train), BL3.Probe(pri, beta=a.beta, adapt=a.adapt))
    runs = {}
    t1 = time.time()
    for ds, songs in B.bench3.eval_sets('val').items():
        runs[ds] = []
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, up, uh = B.simulate(rec, meta, notes, a.L)
            assert len(T) == len(rec.hears), (len(T), len(rec.hears))
            runs[ds].append((s, rec.tabs, rec.hears, T, H))
    print(f'reactive pass with probes {time.time() - t1:.1f}s', flush=True)
    res = []
    h0, r0 = O4.score_all(runs, None, a.L)
    print('reactive  ' + O4.fmt(h0, r0), flush=True)
    res.append({'P': None, 'head': h0, 'rows': r0})
    for P in grid:
        t2 = time.time()
        h, r = O4.score_all(runs, P, a.L)
        res.append({'P': P, 'head': h, 'rows': r})
        print(json.dumps(P) + f' ({time.time() - t2:.0f}s)\n          ' + O4.fmt(h, r, h0, r0), flush=True)
    with open(a.out, 'x') as fh:
        json.dump({'seed': 'seed2' if a.seed2 else 'seed1', 'L': a.L, 'beta': a.beta, 'adapt': a.adapt,
                   'split': 'val', 'results': res}, fh)


if __name__ == '__main__':
    main()
