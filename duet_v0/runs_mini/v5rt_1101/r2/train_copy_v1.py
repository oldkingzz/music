"""R2 (run v5rt_1101, round 2): calibrate the within-song copy model (rt_systems.r2_lib_v1.Copy) on TRAIN only
(bench3.train_sets(): bach, folk, pop, pop909; no val data is read). Writes runs_mini/v5rt_1101/models/R2_copy_v1.npz
(refuses to overwrite).

Each train song is heard with bench3's 10 % mishearing, drawn twice with fixed seeds (crc32(id) ^ 7100 / 7101). The
copy model is fed the heard notes in order; whenever it makes a prediction for the next note we record
  pitch: is its top pitch her TRUE next pitch?   by (match length bucket, vote share bucket, number of matches bucket,
         bucket of A's melody prior P for that pitch (A_prior_v2 'mel', the table the systems use))
  ioi:   is its top inter-onset interval the real one (within 0.01 beat)?   by (length, vote share, matches)
Accuracies are dataset-balanced (each train set carries the same total weight) and backed off to coarser cells with
n / (n + 20) (n = raw count).
"""
import os
import random
import sys
import zlib
from collections import defaultdict

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402  (puts the v3 / duet_beat / rt_systems paths in place)
from rt_systems.r1_A_core_v2 import Prior  # noqa: E402
from rt_systems.r2_lib_v1 import Copy, ngram_b  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'models', 'R2_copy_v1.npz')
SEEDS = (7100, 7101)
NOISE = 0.10
A = 20.0


def mishear(song, seed):
    rng = random.Random(zlib.crc32(str(song.get('id', song.get('title', ''))).encode()) ^ seed)
    out = []
    for on, off, m in sorted(song['sop'], key=lambda n: n[0]):
        hm = m
        if rng.random() < NOISE:
            pc = (m + rng.randint(1, 11)) % 12
            hm = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        out.append((float(on), float(off), int(m), int(hm)))
    return out


def ncls(on, song):
    k = int(on + 1e-9)
    pos = song['beats'][k]['pos'] if k < len(song['beats']) else k % song['M']
    if on - k < 0.01:
        return 0 if int(pos) % song['M'] == 0 else 1
    return 2


def backed(Sw, Ww, Sn, axes_order):
    """Smoothed mean of Sw / Ww over the full key, backing off by summing out axes (last listed first)."""
    g = Sw.sum() / max(Ww.sum(), 1e-12)
    par = np.full((), g)
    keep = []
    for ax in axes_order:
        keep.append(ax)
        other = tuple(i for i in range(Sw.ndim) if i not in keep)
        sw, ww, sn = Sw.sum(other), Ww.sum(other), Sn.sum(other)
        mean = sw / np.maximum(ww, 1e-12)
        lam = sn / (sn + A)
        par = lam * mean + (1 - lam) * np.expand_dims(par, -1)
    return par


def main():
    if os.path.exists(OUT):
        sys.exit(f'refusing to overwrite {OUT}')
    prior = Prior()
    tr = B.bench3.train_sets()
    recs = defaultdict(list)
    stats = defaultdict(lambda: defaultdict(float))
    for ds, songs in tr.items():
        for s in songs:
            t = s['tonic']
            for seed in SEEDS:
                ns = mishear(s, seed)
                cp = Copy(None)
                cp.start()
                for a, (on, off, m, hm) in enumerate(ns):
                    if a >= 1:
                        st = stats[ds]
                        st['notes'] += 1
                        m_c = ns[a - 1][3]
                        m_p = ns[a - 2][3] if a >= 2 else None
                        iv = 11 if m_p is None else int(np.clip(m_c - m_p, -5, 5)) + 5
                        pmel = prior.mel[(m_c - t) % 12, iv, ncls(on, s)]
                        st['ngram_ok'] += int(np.argmax(pmel)) - 12 + m_c == m
                        pred = cp.pred
                        if pred is not None:
                            d = pred['top'] - m_c
                            p_ng = float(pmel[d + 12]) if abs(d) <= 12 else 0.0
                            ok_p = int(pred['top'] == m)
                            ok_i = int(abs(pred['ioi'] - (on - ns[a - 1][0])) < 0.01)
                            recs[ds].append((pred['lb'], pred['pb'], pred['nb'], ngram_b(p_ng), ok_p, pred['ib'], ok_i))
                            st['pred'] += 1
                            st['pred_ok'] += ok_p
                            st['pred_ngram_ok'] += int(np.argmax(pmel)) - 12 + m_c == m
                            st['ioi_ok'] += ok_i
                    cp.add(on, hm)
    tot = sum(len(v) for v in recs.values())
    Sp, Wp, Np = np.zeros((4, 4, 3, 4)), np.zeros((4, 4, 3, 4)), np.zeros((4, 4, 3, 4))
    Si, Wi, Ni = np.zeros((4, 4, 3)), np.zeros((4, 4, 3)), np.zeros((4, 4, 3))
    for ds, rr in recs.items():
        wd = tot / len(recs) / max(len(rr), 1)
        for lb, pb, nb, gb, okp, ib, oki in rr:
            Sp[lb, pb, nb, gb] += wd * okp
            Wp[lb, pb, nb, gb] += wd
            Np[lb, pb, nb, gb] += 1
            Si[lb, ib, nb] += wd * oki
            Wi[lb, ib, nb] += wd
            Ni[lb, ib, nb] += 1
    acc_p = backed(Sp, Wp, Np, (0, 1, 2, 3))
    acc_i = backed(Si, Wi, Ni, (0, 1, 2))
    for ds, st in stats.items():
        n, p = st['notes'], max(st['pred'], 1)
        print(f"{ds}: notes {int(n)} copy predicts {st['pred'] / n:.3f} of them; there copy top-1 {st['pred_ok'] / p:.3f} "
              f"vs A prior top-1 {st['pred_ngram_ok'] / p:.3f}, ioi {st['ioi_ok'] / p:.3f}; A prior top-1 on all notes "
              f"{st['ngram_ok'] / n:.3f}", flush=True)
    print('acc_p by length bucket (mean over cells with data):',
          [round(float((acc_p[lb] * Np[lb]).sum() / max(Np[lb].sum(), 1)), 3) for lb in range(4)])
    print('acc_i by length bucket:', [round(float((acc_i[lb] * Ni[lb]).sum() / max(Ni[lb].sum(), 1)), 3) for lb in range(4)])
    print('acc_p[len, share] (marginal over matches, prior bucket):')
    print(np.round((Sp.sum((2, 3)) / np.maximum(Wp.sum((2, 3)), 1e-12)), 3))
    print('acc_p[len 8-11, share >= 0.95, matches, prior bucket]:')
    print(np.round(acc_p[2, 3], 3))
    with open(OUT, 'xb') as fh:
        np.savez(fh, acc_p=acc_p.astype(np.float32), acc_i=acc_i.astype(np.float32), n_p=Np.astype(np.float32),
                 n_i=Ni.astype(np.float32), seeds=np.array(SEEDS), A=np.float32(A))
    print('wrote', OUT, os.path.getsize(OUT), 'bytes', flush=True)


if __name__ == '__main__':
    main()
