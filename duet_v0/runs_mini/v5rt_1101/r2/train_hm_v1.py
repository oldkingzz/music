"""R2 (run v5rt_1101, round 2): reliability of the harmonic memory, fitted on TRAIN only (no val data is read).
Writes runs_mini/v5rt_1101/models/R2_hm_v1.npz (refuses to overwrite).

The copy model (rt_systems.r2_lib_v1.Copy) is fed each train song's heard notes (bench3's 10 % mishearing, seed
crc32(id) ^ 7100). After every note it aligns her current note with the most recent earlier passage of the longest
match (>= 4 notes; time shift s a whole number of beats, within 0.02). r[len bucket, bar] = P(true chord of the current
beat == true chord s beats earlier), len bucket 4-7 / 8-11 / 12+ notes, bar = s is a whole number of bars. Dataset
balanced (each train set carries the same total weight), smoothed toward the bar-only rate with n / (n + 20).
The same pass gives P(chord of beat k == chord of beat k - 1) for reference.
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

OUT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'models', 'R2_hm_v1.npz')
A = 20.0


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


def hm_lb(best):
    return 0 if best < 8 else 1 if best < 12 else 2


def main():
    if os.path.exists(OUT):
        sys.exit(f'refusing to overwrite {OUT}')
    tr = B.bench3.train_sets()
    cnt = {ds: defaultdict(lambda: [0, 0]) for ds in tr}
    for ds, songs in tr.items():
        for s in songs:
            z = [b['z'] for b in s['beats']]
            ns = mishear(s, 7100)
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
                c = cnt[ds][(hm_lb(best), int(sh % s['M'] == 0))]
                c[0] += 1
                c[1] += z[k] == z[k - sh]
    S, W, N = np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((3, 2))
    for ds, cc in cnt.items():
        n_ds = sum(v[0] for v in cc.values())
        wd = 1.0 / max(n_ds, 1)
        for (lb, bar), (n, ok) in cc.items():
            S[lb, bar] += wd * ok
            W[lb, bar] += wd * n
            N[lb, bar] += n
        print(ds, {f'len{lb} bar{bar}': (n, round(ok / max(n, 1), 3)) for (lb, bar), (n, ok) in sorted(cc.items())},
              flush=True)
    par = S.sum(0) / np.maximum(W.sum(0), 1e-12)
    lam = N / (N + A)
    r = lam * S / np.maximum(W, 1e-12) + (1 - lam) * par[None, :]
    print('r[len bucket 4-7 / 8-11 / 12+, bar 0/1] (dataset-balanced):')
    print(np.round(r, 3))
    with open(OUT, 'xb') as fh:
        np.savez(fh, r=r.astype(np.float32), n=N.astype(np.float32))
    print('wrote', OUT, os.path.getsize(OUT), 'bytes', flush=True)


if __name__ == '__main__':
    main()
