"""v5rt_1101 round 1, idea C: read-only data look for the joint forecaster (train + val, val only for numbers).
Prints: songs / notes per dataset, share of note onsets on the 0.5-beat plan grid (exact and within 0.01 beat),
share of plan points q at which a note starts, and n-gram baselines for her next pitch class at plan points
(a) P(onset at q | bar position, last note still sounding) and (b) P(next pc | last heard pc) from train, scored on val.
"""
import math
import os
import random
import sys
import zlib
from collections import Counter, defaultdict

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402

TOL = 0.01
L = 0.1


def plan_rows(song, heard_sop):
    """Yield (q, last heard note index or -1, onset at q?, pc of the note at q or None, chord at q) for plan points."""
    t = song['tonic']
    true = sorted(song['sop'], key=lambda n: n[0])
    hs = sorted(heard_sop, key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    K = song['K']
    j = 0
    nxt = 0
    for i in range(int(math.ceil(K / 0.5)) + 1):
        q = 0.5 * i
        if q >= K:
            break
        while nxt < len(hs) and hs[nxt][0] + L <= q:
            nxt += 1
        last = nxt - 1
        while j < len(true) and true[j][0] < q - 1e-9:
            j += 1
        on = j < len(true) and true[j][0] < q + TOL
        pc = (true[j][2] - t) % 12 if on else None
        zq = z[int(q)] if int(q) < len(z) else -1
        yield q, last, on, pc, zq, hs


def main():
    tr = bench3.train_sets()
    va = bench3.eval_sets('val')
    for name, sets in (('train', tr), ('val', va)):
        for ds, songs in sets.items():
            n = sum(len(s['sop']) for s in songs)
            ons = [on for s in songs for on, off, m in s['sop']]
            ex = sum(abs(o * 2 - round(o * 2)) < 1e-9 for o in ons) / max(len(ons), 1)
            near = sum((o * 2 - math.floor(o * 2 + 1e-9)) / 2 < TOL for o in ons) / max(len(ons), 1)
            durs = sorted(off - on for s in songs for on, off, m in s['sop'])
            print(f'{name} {ds}: songs {len(songs)} notes {n} onsets on grid exact {ex:.3f} within {TOL} {near:.3f} '
                  f'median dur {durs[len(durs) // 2]:.3f}', flush=True)
    # n-gram tables from train (clean melody as input here; val uses bench3's heard melody)
    on_tab = defaultdict(lambda: [1.0, 1.0])
    pc_tab = defaultdict(lambda: [0.5] * 12)
    for ds, songs in tr.items():
        for s in songs:
            for q, last, on, pc, zq, hs in plan_rows(s, s['sop']):
                sounding = last >= 0 and hs[last][1] + L > q
                key = (int(q) % s['M'] if q == int(q) else -1, sounding)
                on_tab[key][int(on)] += 1
                if on and last >= 0:
                    pc_tab[(hs[last][2] - s['tonic']) % 12][pc] += 1
    for ds, songs in va.items():
        npt = non = 0
        ll = 0.0
        acc = Counter()
        for s in songs:
            h = bench3.B2.heard(s)
            for q, last, on, pc, zq, hs in plan_rows(s, h['sop']):
                if q < bench3.J:
                    continue
                npt += 1
                non += on
                sounding = last >= 0 and hs[last][1] + L > q
                key = (int(q) % s['M'] if q == int(q) else -1, sounding)
                c = on_tab[key]
                p = c[1] / (c[0] + c[1])
                acc['a_right'] += (p >= 0.5) == on
                ll += -math.log(p if on else 1 - p)
                if on and last >= 0:
                    row = pc_tab[(hs[last][2] - s['tonic']) % 12]
                    order = sorted(range(12), key=lambda x: -row[x])
                    acc['b_n'] += 1
                    acc['b_top1'] += order[0] == pc
                    acc['b_top3'] += pc in order[:3]
        print(f'val {ds}: plan points {npt} with onset {non / npt:.3f}; n-gram (a) acc {acc["a_right"] / npt:.3f} '
              f'nll {ll / npt:.3f}; (b) bigram top1 {acc["b_top1"] / acc["b_n"]:.3f} top3 {acc["b_top3"] / acc["b_n"]:.3f}',
              flush=True)


if __name__ == '__main__':
    main()
