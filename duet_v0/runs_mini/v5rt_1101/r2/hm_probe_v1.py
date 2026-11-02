"""R2 probe (run v5rt_1101, round 2): when the copy model aligns her current note with an earlier passage (shift s
beats, a whole number), how often is the TRUE chord of the current beat the true chord s beats earlier? Diagnostic
only; prints per dataset, by match length, for train (mishearing seed crc32 ^ 7100) and val (bench3's seed 1).
Val numbers are a tuned-on-val diagnostic.
"""
import os
import random
import sys
import zlib
from collections import defaultdict

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402
from rt_systems.r2_lib_v1 import Copy  # noqa: E402


def mishear(song, seed):
    rng = random.Random(zlib.crc32(str(song.get('id', song.get('title', ''))).encode()) ^ seed)
    out = []
    for on, off, m in sorted(song['sop'], key=lambda n: n[0]):
        hm = m
        if rng.random() < 0.1:
            pc = (m + rng.randint(1, 11)) % 12
            hm = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        out.append((float(on), float(off), int(m), int(hm)))
    return out


def probe(sets, heard_fn, label, max_songs=0):
    for ds, songs in sets.items():
        if max_songs:
            songs = songs[:max_songs]
        st = defaultdict(lambda: [0, 0, 0])
        for s in songs:
            z = [b['z'] for b in s['beats']]
            ns = heard_fn(s)
            cp = Copy(None)
            cp.start()
            for a, (on, off, m, hm) in enumerate(ns):
                cp.add(on, hm)
                r = cp.r1
                if len(r) == 0:
                    continue
                best = int(r.max())
                if best < 4:
                    continue
                b = int(np.nonzero(r == best)[0][-1])
                d = cp.on[a] - cp.on[b]
                sh = int(round(d))
                if sh < 1 or abs(d - sh) > 0.02:
                    continue
                k = int(on)
                if k - sh < 0 or k >= len(z) or z[k] < 0 or z[k - sh] < 0:
                    continue
                key = (min(best, 12) // 4, int(sh % s['M'] == 0))
                st[key][0] += 1
                st[key][1] += z[k] == z[k - sh]
                st[key][2] += z[k] == z[k - 1] if k >= 1 else 0
        out = ' | '.join(f"len{4 * lb}+ bar{bar}: n {v[0]} agree {v[1] / max(v[0], 1):.3f} (same as prev beat "
                         f"{v[2] / max(v[0], 1):.3f})" for (lb, bar), v in sorted(st.items()))
        print(f'{label} {ds}: {out}', flush=True)


def main():
    tr = B.bench3.train_sets()
    probe(tr, lambda s: mishear(s, 7100), 'train', max_songs=600)
    val = B.bench3.eval_sets('val')
    probe(val, lambda s: [(float(o), float(f), int(m), int(h)) for (o, f, m), (_, _, h) in
                          zip(sorted(s['sop'], key=lambda n: n[0]),
                              sorted(B.HEARD1(s)['sop'], key=lambda n: n[0]))], 'val')


if __name__ == '__main__':
    main()
