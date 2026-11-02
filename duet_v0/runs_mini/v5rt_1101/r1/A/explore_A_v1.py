"""Idea A (run v5rt_1101, round 1), read-only data look. Val only for scores (tuned on val); train only for the model.
Prints: (1) per dataset (train and val) where her onsets sit on the 0.5 grid, durations, how her next TRUE pitch moves;
(2) reactive:r2_mix_v2 at L = 0.1 on val split into first-L / rest of the onset beat / later beats of a held note, with
hit and harsh shares, and what the reactive choice would score in the first L if it had sounded at the onset."""
import bisect
import math
import os
import sys
from collections import Counter

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402

L = 0.1


def grid_stats(name, songs):
    n = on_grid = near = q25 = other = 0
    dur, rep, step2, step4, big = [], 0, 0, 0, 0
    nn = 0
    for s in songs:
        notes = sorted(s['sop'], key=lambda x: x[0])
        for i, (on, off, m) in enumerate(notes):
            n += 1
            x = on / 0.5
            fr = x - math.floor(x + 1e-9)
            if abs(fr) < 1e-9:
                on_grid += 1
            elif fr * 0.5 < 0.002:
                near += 1
            elif abs(fr - 0.5) < 1e-6:
                q25 += 1
            else:
                other += 1
            dur.append(off - on)
            if i + 1 < len(notes):
                nn += 1
                d = notes[i + 1][2] - m
                rep += d == 0
                step2 += abs(d) <= 2
                step4 += abs(d) <= 4
                big += abs(d) > 7
    dur.sort()
    print(f'{name:14s} notes {n:6d}: onset on 0.5 grid {on_grid / n:.3f}, within 0.002 after grid {near / n:.3f}, '
          f'on .25 {q25 / n:.3f}, other {other / n:.3f}; dur median {dur[len(dur) // 2]:.3f} mean {sum(dur) / n:.3f}; '
          f'next pitch repeat {rep / nn:.3f} |d|<=2 {step2 / nn:.3f} |d|<=4 {step4 / nn:.3f} |d|>7 {big / nn:.3f}',
          flush=True)


def decompose(ds, songs, obj):
    acc = Counter()
    for s in songs:
        notes = [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(s)['sop'], key=lambda n: n[0])]
        meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
        T, H, _, _ = B.simulate(obj, meta, notes, L)
        true = sorted(s['sop'], key=lambda n: n[0])
        z = [b['z'] for b in s['beats']]
        t = s['tonic']
        for on, off, m in true:
            h_r = B.at(T, H, on + L)                     # the reactive choice for this note
            k0 = int(on)
            for k in range(max(B.J, int(on)), int(off) + 1):
                ov = min(off, k + 1) - max(on, k)
                if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                    continue
                for x0, x1, h in B.pieces(T, H, max(on, k), min(off, k + 1)):
                    w = x1 - x0
                    f = max(0.0, min(x1, on + L) - x0)
                    g = B.hit(h, m, t, z[k])
                    bad = h >= 0 and abs(m - h) % 12 in B.HARSH
                    cat = 'first' if f > 0 else ('onbeat' if k == k0 else 'held')
                    if f > 0 and f < w:
                        acc[('first', 'd')] += f
                        acc[('first', 'hit')] += f * g
                        acc[('first', 'harsh')] += f * bad
                        acc[('first', 'hit_r')] += f * B.hit(h_r, m, t, z[k])
                        acc[('first', 'harsh_r')] += f * (h_r >= 0 and abs(m - h_r) % 12 in B.HARSH)
                        w -= f
                        cat = 'onbeat' if k == k0 else 'held'
                    elif f > 0:
                        acc[('first', 'hit_r')] += w * B.hit(h_r, m, t, z[k])
                        acc[('first', 'harsh_r')] += w * (h_r >= 0 and abs(m - h_r) % 12 in B.HARSH)
                    acc[(cat, 'd')] += w
                    acc[(cat, 'hit')] += w * g
                    acc[(cat, 'harsh')] += w * bad
                    if cat == 'held':
                        chg = z[k] != z[k0] if k0 < len(z) else True
                        acc[('held_chg' if chg else 'held_same', 'd')] += w
                        acc[('held_chg' if chg else 'held_same', 'hit')] += w * g
                        best = any(B.hit(hh, m, t, z[k]) for hh in range(53, 78))
                        acc[('held_chg' if chg else 'held_same', 'possible')] += w * best
    tot = sum(acc[(c, 'd')] for c in ('first', 'onbeat', 'held'))
    out = [f'{ds:7s} counted {tot:9.1f} beats:']
    for c in ('first', 'onbeat', 'held', 'held_chg', 'held_same'):
        d = acc[(c, 'd')] or 1e-9
        extra = ''
        if c == 'first':
            extra = f" | reactive note from the onset: hit {acc[('first', 'hit_r')] / d:.3f} harsh {acc[('first', 'harsh_r')] / d:.3f}"
        if c.startswith('held_'):
            extra = f" | some alto note would hit {acc[(c, 'possible')] / d:.3f}"
        out.append(f'  {c:9s} share {acc[(c, "d")] / tot:.3f} hit {acc[(c, "hit")] / d:.3f} harsh {acc[(c, "harsh")] / d:.3f}{extra}')
    print('\n'.join(out), flush=True)


if __name__ == '__main__':
    tr = B.bench3.train_sets()
    for ds, songs in tr.items():
        grid_stats('train ' + ds, songs)
    va = B.bench3.eval_sets('val')
    for ds, songs in va.items():
        grid_stats('val ' + ds, songs)
    obj = B.load('reactive:r2_mix_v2', tr)
    for ds, songs in va.items():
        decompose(ds, songs, obj)
