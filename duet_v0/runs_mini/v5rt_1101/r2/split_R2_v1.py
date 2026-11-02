"""R2 (run v5rt_1101, round 2): where does hit / harsh time come from? Same split as round 1's
runs_mini/v5rt_1101/r1/C/split_C_v1.py (its song_split is imported unchanged): her counted time is cut into
  first   the first L of each of her notes          rest0   the rest of her note inside its onset beat
  cont=   later beats of a held note, same chord   cont!=  later beats of a held note across a chord change
and share of time, hit rate and harsh rate are printed per part and dataset, plus the headline (mean over datasets)
contribution of each part. bench_rt.load / simulate unchanged; val, seed 1. Val only, tuned on val.

  python split_R2_v1.py <spec> [<spec> ...] [--L 0.1] --out <new .json under runs_mini/v5rt_1101/r2/>
"""
import argparse
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'C'))
from split_C_v1 import B, PARTS, song_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('specs', nargs='+')
    ap.add_argument('--L', type=float, default=0.1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert out.startswith(HERE + os.sep) and out.endswith('.json') and not os.path.exists(out), out
    B.bench3.B2.heard = B.HEARD1
    train = B.bench3.train_sets()
    res = {}
    for spec in a.specs:
        obj = B.load(spec, train)
        res[spec] = {}
        for ds, songs in B.bench3.eval_sets('val').items():
            acc = {p: {'d': 0.0, 'hit': 0.0, 'harsh': 0.0} for p in PARTS}
            for s in songs:
                notes = [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(s)['sop'], key=lambda n: n[0])]
                meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
                if isinstance(obj, B.Oracle):
                    obj.tell(notes)
                T, H, _, _ = B.simulate(obj, meta, notes, a.L)
                song_split(s, T, H, a.L, acc)
            D = sum(v['d'] for v in acc.values())
            res[spec][ds] = {p: {'share': v['d'] / D, 'hit': v['hit'] / max(v['d'], 1e-9),
                                 'harsh': v['harsh'] / max(v['d'], 1e-9)} for p, v in acc.items()}
        head = {p: sum(res[spec][ds][p]['share'] * res[spec][ds][p]['hit'] for ds in res[spec]) / len(res[spec])
                for p in PARTS}
        res[spec]['headline_hit_contrib'] = head
        print(f'{spec}: headline hit contribution ' + ' '.join(f'{p} {v:.4f}' for p, v in head.items()) + ' | ' +
              ' | '.join(f"{ds} " + ' '.join(f"{p} {r[p]['hit']:.3f}/{r[p]['harsh']:.3f}" for p in PARTS)
                         for ds, r in res[spec].items() if ds != 'headline_hit_contrib'), flush=True)
    with open(out, 'x') as fh:
        json.dump({'L': a.L, 'seed': 'seed1', 'split': 'val', 'res': res}, fh, indent=1)


if __name__ == '__main__':
    main()
