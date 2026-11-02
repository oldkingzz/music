"""Idea A check (run v5rt_1101, round 1). VAL ONLY. attack_metric.py judges a Duet attack at time x against her TRUE
note sounding at x. pop909 onsets sit 0.0002-0.0003 beat after the 0.5-beat plan grid, so a plan attack at q is judged
against her OLD note although it sounds together with her new one (6 ms is simultaneous to the ear). This recounts
fresh clashes judging every attack against her note sounding at x + SHIFT (SHIFT = 0.01 beat = 6 ms at 100 bpm); on
the other datasets (onsets on the grid) almost nothing changes. Same notes, same J / labelled-beat rule as
attack_metric.py; uses bench_rt.load / simulate unchanged.

  python attack_honest_A_v1.py <spec> [<spec> ...] [--L 0.1]
"""
import argparse
import bisect
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402

SHIFT = 0.01


def counts(song, T, H, acc):
    true = sorted(song['sop'], key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    for on, off, m in true:
        k0 = int(on)
        if on >= B.J and k0 < len(z) and z[k0] >= 0:
            acc['notes'] += 1
    ons = [n[0] for n in true]
    for j, x in enumerate(T):
        h = H[j]
        if h < 0 or (j > 0 and H[j - 1] == h):
            continue
        for name, y in (('plain', x), ('shift', x + SHIFT)):
            i = bisect.bisect_right(ons, y) - 1
            if i < 0 or not (true[i][0] <= y < true[i][1]):
                continue
            k = int(x)
            if x < B.J or k >= len(z) or z[k] < 0:
                continue
            acc[name] += abs(true[i][2] - h) % 12 in B.HARSH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('specs', nargs='+')
    ap.add_argument('--L', type=float, default=0.1)
    a = ap.parse_args()
    B.bench3.B2.heard = B.HEARD1
    train = B.bench3.train_sets()
    for spec in a.specs:
        obj = B.load(spec, train)
        out = []
        tot = {'plain': 0.0, 'shift': 0.0}
        for ds, songs in B.bench3.eval_sets('val').items():
            acc = {'notes': 0.0, 'plain': 0.0, 'shift': 0.0}
            for s in songs:
                notes = [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(s)['sop'], key=lambda n: n[0])]
                meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
                T, H, _, _ = B.simulate(obj, meta, notes, a.L)
                counts(s, T, H, acc)
            p, q = 100 * acc['plain'] / acc['notes'], 100 * acc['shift'] / acc['notes']
            tot['plain'] += p / 4
            tot['shift'] += q / 4
            out.append(f'{ds} {p:.2f}->{q:.2f}')
        print(f"{spec} L={a.L:g}: fresh clashes per 100 notes, as attack_metric -> judged 6 ms later: headline "
              f"{tot['plain']:.2f}->{tot['shift']:.2f} | " + ' '.join(out), flush=True)


if __name__ == '__main__':
    main()
