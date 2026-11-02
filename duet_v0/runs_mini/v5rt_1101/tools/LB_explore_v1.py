"""Round 3 phase 1 (live bench): facts about the val / train Bach chorales the live bench renders, and how long one
render + tracker pass takes. Read-only; prints only. Val only for scoring later; train used only for the tracker profile.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/tools/LB_explore_v1.py
"""
import os
import sys
import time
from collections import Counter

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402  (v3 + duet_beat paths)
import numpy as np  # noqa: E402

bench3 = B.bench3


def facts(name, songs):
    print(f'== {name}: {len(songs)} songs', flush=True)
    durs, gaps, lo, hi, pickups, posbad, rests, firsts, Ks, ns = Counter(), 0, 999, 0, 0, 0, 0, Counter(), [], []
    for s in songs:
        sop = sorted(s['sop'], key=lambda n: n[0])
        pos = [b['pos'] for b in s['beats']]
        if any(p != k % s['M'] for k, p in enumerate(pos)):
            posbad += 1
        if pos and pos[0] != 0:
            pickups += 1
        Ks.append(s['K'])
        ns.append(len(sop))
        firsts[round(sop[0][0], 3)] += 1
        for j, (on, off, m) in enumerate(sop):
            durs[round(off - on, 3)] += 1
            lo, hi = min(lo, m), max(hi, m)
            if j + 1 < len(sop):
                g = sop[j + 1][0] - off
                if g > 1e-9:
                    rests += 1
                if g < -1e-9:
                    gaps += 1
    print(f'K min/median/max {min(Ks)}/{int(np.median(Ks))}/{max(Ks)}; notes/song median {int(np.median(ns))}; '
          f'midi {lo}..{hi}; songs with pos != k % M: {posbad}; pos[0] != 0: {pickups}; rests between notes {rests}; '
          f'overlaps {gaps}', flush=True)
    print('durations (beats):', sorted(durs.items())[:20], flush=True)
    print('first onsets:', sorted(firsts.items())[:10], flush=True)
    s = songs[0]
    print('example id', s.get('id'), 'tonic', s['tonic'], 'M', s['M'], 'K', s['K'], 'keys', sorted(s.keys()), flush=True)
    print('pos head', [b['pos'] for b in s['beats']][:12], 'z head', [b['z'] for b in s['beats']][:12], flush=True)


val = bench3.eval_sets('val')['bach']
train = bench3.train_sets()['bach']
facts('val bach', val)
facts('train bach', train)
print('val ids', [s['id'] for s in val], flush=True)
print('train ids head', [s['id'] for s in train[:8]], flush=True)
from songs import twinkle  # noqa: E402
tw = twinkle()
print('twinkle K', tw['K'], 'notes', len(tw['sop']), 'id', tw['id'], flush=True)

# one render + one tracker pass, timed
sys.path.insert(0, ABS)
import singer  # noqa: E402
from dsp import HOP, SR  # noqa: E402
from follow import FeatureTracker  # noqa: E402
s = val[0]
T_b = 60.0 / 90
notes = [(3.0 + on * T_b, 3.0 + off * T_b, m) for on, off, m in sorted(s['sop'], key=lambda n: n[0])]
c0 = time.perf_counter()
y, info = singer.render(notes)
c1 = time.perf_counter()
ft = FeatureTracker()
for b in range(len(y) // HOP):
    ft.step(y[b * HOP:(b + 1) * HOP].astype(float))
c2 = time.perf_counter()
print(f'render {len(y) / SR:.1f} s audio in {c1 - c0:.2f} s; tracker {len(y) // HOP} blocks in {c2 - c1:.2f} s; '
      f'kinds {Counter(i["kind"] for i in info)}', flush=True)
