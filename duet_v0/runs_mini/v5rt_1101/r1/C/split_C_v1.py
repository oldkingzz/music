"""v5rt_1101 r1 idea C: where does hit / harsh time come from? Runs a bench_rt spec (bench_rt.load / simulate /
pieces / hit unchanged) on val (seed1) and splits her counted time into
  first   the first L of each of her notes (cut at its end)
  rest0   the rest of her note inside its onset beat
  cont=   later beats of her note whose chord equals the chord of the beat before (held across, chord unchanged)
  cont!=  later beats of her note whose chord differs from the beat before (held across a chord change)
and prints share of time, hit rate and harsh rate per part and dataset. Val only, tuned on val.

  python split_C_v1.py <spec> [--L 0.1] --out <new .json under r1/C/>
"""
import argparse
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402

PARTS = ('first', 'rest0', 'cont=', 'cont!=')


def song_split(song, T, H, L, acc):
    true = sorted(song['sop'], key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    t = song['tonic']
    for on, off, m in true:
        e = on + L
        k0 = int(on)
        for k in range(max(B.J, k0), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            for x0, x1, h in B.pieces(T, H, max(on, k), min(off, k + 1)):
                g = B.hit(h, m, t, z[k])
                bad = h >= 0 and abs(m - h) % 12 in B.HARSH
                segs = []
                if k == k0:
                    f = max(0.0, min(x1, e) - x0)
                    segs = [('first', f), ('rest0', (x1 - x0) - f)]
                else:
                    same = k - 1 >= 0 and z[k - 1] == z[k]
                    segs = [('cont=' if same else 'cont!=', x1 - x0)]
                for p, w in segs:
                    if w <= 0:
                        continue
                    a = acc[p]
                    a['d'] += w
                    a['hit'] += w * g
                    a['harsh'] += w * bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--L', type=float, default=0.1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert out.startswith(HERE + os.sep) and not os.path.exists(out), out
    obj = B.load(a.spec, B.bench3.train_sets())
    res = {}
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
        res[ds] = {p: {'share': v['d'] / D, 'hit': v['hit'] / max(v['d'], 1e-9), 'harsh': v['harsh'] / max(v['d'], 1e-9)}
                   for p, v in acc.items()}
        print(f'{ds}: ' + ' | '.join(f"{p} share {r['share']:.3f} hit {r['hit']:.3f} harsh {r['harsh']:.3f}"
                                     for p, r in res[ds].items()), flush=True)
    with open(out, 'x') as fh:
        json.dump({'spec': a.spec, 'L': a.L, 'seed': 'seed1', 'split': 'val', 'parts': res}, fh, indent=1)


if __name__ == '__main__':
    main()
