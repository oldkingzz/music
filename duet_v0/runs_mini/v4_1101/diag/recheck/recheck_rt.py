"""Independent recheck of m1_filter_v5lo's REAL-TIME chord belief. VAL split only (this bundle has no test split
on purpose). Written from the task definitions; it does not use tools/diag_rt.py or any other diag/ file.

  1) rt_acc        argmax of the belief m1 computes inside step() == true z[k]; her notes as bench3.run_song feeds
                   them (heard melody, sorted by onset), k = int(onset), counted when k >= J (=4) and z[k] >= 0
  2) rt_acc by the note's index inside its chord span (maximal run of consecutive beats with the same label z >= 0):
                   1, 2, 3, >= 4. Main = index over ALL notes with onset in the span (literal definition);
                   also 'clip4' = index counted only over notes with k >= 4 (sensitivity).
  3) told_nd       music_nd (rescore_nd.py's exact formula) when m1's chooser gets one-hot(true z[k]) (m1's own
                   belief when z[k] < 0). Main = told at every k; 'told4' = told only for k >= 4 (sensitivity).
  4) smooth_nd / smooth_acc   belief(k) = forward-backward posterior of beat k over the whole heard song with m1's
                   own Model (pi, A[tkey(pos, M)], lam * logB @ time-per-relative-pc), m1's chooser unchanged.
  5) sanity        m1 unchanged, music_nd headline (expected 0.620637625406759).
Writes one NEW json next to this script (open mode 'x': never overwrites) and prints a summary.
"""
import importlib
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
_bad = sorted(k for k in os.environ if k.startswith(('M1_', 'M3_')) or k == 'DUET_HIST')
if _bad:
    sys.exit(f'refusing to run: environment has {_bad}')

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))

import numpy as np  # noqa: E402

import bench3  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402

M1 = importlib.import_module('systems.m1_filter_v5lo')
J = bench3.J
S = M1.S
LAM = M1.P['lam']
EXPECT = 0.620637625406759
BUCKETS = ('1', '2', '3', '>=4')
VARIANTS = ('m1', 'rec', 'told', 'told4', 'smooth')


def bucket(i):
    return str(i) if i <= 3 else '>=4'


class RecFilter(M1.Filter):
    """m1 unchanged; records the belief step() computes; checks our forward pass against m1's own alpha."""
    alpha_fw = None

    def start(self, meta):
        super().start(meta)
        self.rec = []
        self.alpha_diff = 0.0
        self.alpha_n = 0

    def belief(self, k):
        if self.alpha is not None and self.done == k and 1 <= k <= len(self.alpha_fw):
            self.alpha_diff = max(self.alpha_diff, float(np.abs(self.alpha - self.alpha_fw[k - 1]).max()))
            self.alpha_n += 1
        b = super().belief(k)
        self.rec.append((k, np.array(b, dtype=float, copy=True)))
        return b


class ToldFilter(M1.Filter):
    """m1's chooser; belief = one-hot(true z[k]) when z[k] >= 0 (and k >= first), else m1's own belief."""
    zt = ()
    first = 0

    def start(self, meta):
        super().start(meta)
        self.n_told = 0
        self.n_own = 0

    def belief(self, k):
        b = super().belief(k)
        if k >= self.first and 0 <= k < len(self.zt) and self.zt[k] >= 0:
            self.n_told += 1
            oh = np.zeros(S)
            oh[self.zt[k]] = 1.0
            return oh
        self.n_own += 1
        return b


class SmoothFilter(M1.Filter):
    """m1's chooser; belief = forward-backward posterior of beat k over the whole heard song."""
    gamma = None

    def start(self, meta):
        super().start(meta)
        self.n_fallback = 0

    def belief(self, k):
        if 0 <= k < len(self.gamma):
            return self.gamma[k].copy()
        self.n_fallback += 1
        return super().belief(k)


def evidence(model, hsop, tonic, K):
    """lam * logB @ (time each tonic-relative pc sounds in beat k), over the WHOLE heard song; 0 for silent beats."""
    rel = [[on, off, (m - tonic) % 12] for on, off, m in hsop]
    obs = M1.beat_obs(rel, K)
    E = np.zeros((K, S))
    for k in range(K):
        if obs[k].sum() > 0:
            E[k] = LAM * (model.logB @ obs[k])
    return E


def forward_backward(model, E, pos, M):
    K = len(E)
    L = np.exp(E - E.max(1, keepdims=True))
    T = [None] + [model.A[M1.tkey(pos[k], M)] for k in range(1, K)]      # T[k]: z_{k-1} -> z_k (rows = from)
    al = np.zeros((K, S))
    a = model.pi * L[0]
    al[0] = a / a.sum()
    for k in range(1, K):
        a = (al[k - 1] @ T[k]) * L[k]
        al[k] = a / a.sum()
    be = np.ones((K, S))
    for k in range(K - 2, -1, -1):
        b = T[k + 1] @ (L[k + 1] * be[k + 1])
        be[k] = b / b.sum()
    g = al * be
    g /= g.sum(1, keepdims=True)
    return g, al


def nd_add(acc, song, hs, cross=None):
    """rescore_nd.py's music_nd accumulation, same loop order (so the float sums match bit for bit)."""
    true = sorted(song['sop'], key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    t = song['tonic']
    for i, (on, off, m) in enumerate(true):
        h = hs[i]
        kon = int(on)
        for k in range(max(J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            acc[0] += ov
            if h >= 0:
                iv = abs(m - h) % 12
                acc[1] += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])
                acc[2] += ov * (iv == 0)
            if cross is not None:
                cross[0] += ov
                if not (0 <= kon < len(z)) or z[kon] != z[k]:
                    cross[1] += ov


def main():
    t0 = time.time()
    base = M1.build(bench3.train_sets())                                 # m1 unchanged (Filter of the cached Model)
    model = base.m
    rec, told, told4, smo = RecFilter(model), ToldFilter(model), ToldFilter(model), SmoothFilter(model)
    told4.first = J
    out = {'per_ds': {}, 'checks': {}}
    chk = dict(K_mismatch=0, hs_rec_mismatch_songs=0, rec_len_mismatch=0, alpha_maxdiff=0.0, alpha_compared=0,
               ev_maxdiff=0.0, ev_beats_diff=0, ev_compared=0, smooth_fallback=0, told_notes=0, told_own=0,
               told4_notes=0, told4_own=0, neg_onset=0, near_int_onset=0, songs=0, notes=0)
    for ds, songs in bench3.eval_sets('val').items():
        nd = {v: [0.0, 0.0, 0.0] for v in VARIANTS}
        cross = [0.0, 0.0]
        a = dict(n=0, rt=0, sm=0, ties=0, w=0.0, wrt=0.0, wsm=0.0, beyond=0, unlab=0, early=0)
        bi = {b: dict(n=0, rt=0, sm=0) for b in BUCKETS}
        bc = {b: dict(n=0, rt=0) for b in BUCKETS}
        for s in songs:
            chk['songs'] += 1
            z = [b['z'] for b in s['beats']]
            pos = [b['pos'] for b in s['beats']]
            K = len(z)
            chk['K_mismatch'] += int(K != s['K'])
            hsong = bench3.B2.heard(s)
            hnotes = sorted(hsong['sop'], key=lambda n: n[0])                # as bench3.run_song feeds them
            tnotes = sorted(s['sop'], key=lambda n: n[0])
            assert [n[0] for n in hnotes] == [n[0] for n in tnotes] and [n[1] for n in hnotes] == [n[1] for n in tnotes]
            chk['notes'] += len(hnotes)
            E = evidence(model, hsong['sop'], s['tonic'], K)
            g, al = forward_backward(model, E, pos, s['M'])

            hs_m1 = bench3.run_song(base, s)
            rec.alpha_fw = al
            hs_rec = bench3.run_song(rec, s)
            chk['hs_rec_mismatch_songs'] += int(hs_rec != hs_m1)
            chk['rec_len_mismatch'] += int(len(rec.rec) != len(hnotes))
            chk['alpha_maxdiff'] = max(chk['alpha_maxdiff'], rec.alpha_diff)
            chk['alpha_compared'] += rec.alpha_n
            ev = np.array(rec.ev).reshape(-1, S)                             # m1's own evidence of finished beats
            n_ev = min(len(ev), K)
            if n_ev:
                d = np.abs(ev[:n_ev] - E[:n_ev]).max(1)
                chk['ev_maxdiff'] = max(chk['ev_maxdiff'], float(d.max()))
                chk['ev_beats_diff'] += int((d > 1e-9).sum())
                chk['ev_compared'] += n_ev
            told.zt = z
            told4.zt = z
            hs_told = bench3.run_song(told, s)
            chk['told_notes'] += told.n_told
            chk['told_own'] += told.n_own
            hs_told4 = bench3.run_song(told4, s)
            chk['told4_notes'] += told4.n_told
            chk['told4_own'] += told4.n_own
            smo.gamma = g
            hs_sm = bench3.run_song(smo, s)
            chk['smooth_fallback'] += smo.n_fallback

            nd_add(nd['m1'], s, hs_m1, cross)
            nd_add(nd['rec'], s, hs_rec)
            nd_add(nd['told'], s, hs_told)
            nd_add(nd['told4'], s, hs_told4)
            nd_add(nd['smooth'], s, hs_sm)

            span = [-1] * K
            sid = -1
            for k in range(K):
                if z[k] >= 0:
                    if not (k > 0 and z[k - 1] == z[k]):
                        sid += 1
                    span[k] = sid
            c_lit, c_clip = {}, {}
            for i, (on, off, m) in enumerate(hnotes):
                k = int(on)
                kk, b = rec.rec[i]
                assert kk == k
                chk['neg_onset'] += int(on < 0)
                chk['near_int_onset'] += int(on - np.floor(on) > 1 - 1e-4)
                idx_lit = idx_clip = None
                if 0 <= k < K and span[k] >= 0:
                    c_lit[span[k]] = c_lit.get(span[k], 0) + 1
                    idx_lit = c_lit[span[k]]
                    if k >= J:
                        c_clip[span[k]] = c_clip.get(span[k], 0) + 1
                        idx_clip = c_clip[span[k]]
                if k < J:
                    a['early'] += 1
                    continue
                if k >= K:
                    a['beyond'] += 1
                    continue
                if z[k] < 0:
                    a['unlab'] += 1
                    continue
                ok = int(np.argmax(b)) == z[k]
                oks = int(np.argmax(g[k])) == z[k]
                w = off - on
                a['n'] += 1
                a['rt'] += ok
                a['sm'] += oks
                a['ties'] += int((b == b.max()).sum() > 1)
                a['w'] += w
                a['wrt'] += w * ok
                a['wsm'] += w * oks
                bb = bi[bucket(idx_lit)]
                bb['n'] += 1
                bb['rt'] += ok
                bb['sm'] += oks
                cc = bc[bucket(idx_clip)]
                cc['n'] += 1
                cc['rt'] += ok
        r = {v: {'music_nd': nd[v][1] / nd[v][0], 'double': nd[v][2] / nd[v][0]} for v in VARIANTS}
        out['per_ds'][ds] = {
            'nd': r,
            'rt_acc': a['rt'] / a['n'], 'smooth_acc': a['sm'] / a['n'],
            'rt_acc_durw': a['wrt'] / a['w'], 'smooth_acc_durw': a['wsm'] / a['w'],
            'notes_counted': a['n'], 'argmax_ties': a['ties'],
            'notes_excluded': {'k<4': a['early'], 'k>=len(beats)': a['beyond'], 'z<0': a['unlab']},
            'by_index': {b: {'n': bi[b]['n'], 'rt_acc': bi[b]['rt'] / bi[b]['n'] if bi[b]['n'] else None,
                             'smooth_acc': bi[b]['sm'] / bi[b]['n'] if bi[b]['n'] else None} for b in BUCKETS},
            'by_index_clip4': {b: {'n': bc[b]['n'], 'rt_acc': bc[b]['rt'] / bc[b]['n'] if bc[b]['n'] else None}
                               for b in BUCKETS},
            'scored_time_onset_chord_differs': cross[1] / cross[0],
            'songs': len(songs)}
        print(ds, 'done', f'{time.time() - t0:.0f}s', flush=True)
    P = out['per_ds']
    dss = list(P)

    def mean(f):
        vals = [f(P[d]) for d in dss]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if len(vals) == len(dss) else float('nan')

    out['headline_nd'] = {v: sum(P[d]['nd'][v]['music_nd'] for d in dss) / 4 for v in VARIANTS}
    out['rt_acc_mean'] = mean(lambda p: p['rt_acc'])
    out['smooth_acc_mean'] = mean(lambda p: p['smooth_acc'])
    out['rt_acc_durw_mean'] = mean(lambda p: p['rt_acc_durw'])
    out['smooth_acc_durw_mean'] = mean(lambda p: p['smooth_acc_durw'])
    out['by_index_mean'] = {b: mean(lambda p: p['by_index'][b]['rt_acc']) for b in BUCKETS}
    out['by_index_smooth_mean'] = {b: mean(lambda p: p['by_index'][b]['smooth_acc']) for b in BUCKETS}
    out['by_index_clip4_mean'] = {b: mean(lambda p: p['by_index_clip4'][b]['rt_acc']) for b in BUCKETS}
    out['sanity'] = {'m1_headline_nd': out['headline_nd']['m1'], 'expected': EXPECT,
                     'abs_diff': abs(out['headline_nd']['m1'] - EXPECT)}
    out['checks'] = chk
    out['datasets'] = dss
    out['secs'] = time.time() - t0
    path = os.path.join(HERE, f"recheck_rt_result_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('wrote', path)
    print('sanity m1 headline_nd', repr(out['headline_nd']['m1']), 'expected', EXPECT)
    for v in VARIANTS:
        print(f"{v:7s} headline_nd {out['headline_nd'][v]:.6f} " +
              ' '.join(f"{d} {P[d]['nd'][v]['music_nd']:.6f}" for d in dss))
    print('rt_acc     ', f"{out['rt_acc_mean']:.6f}", ' '.join(f"{d} {P[d]['rt_acc']:.6f}" for d in dss))
    print('smooth_acc ', f"{out['smooth_acc_mean']:.6f}", ' '.join(f"{d} {P[d]['smooth_acc']:.6f}" for d in dss))
    print('rt_acc_durw', f"{out['rt_acc_durw_mean']:.6f}", ' '.join(f"{d} {P[d]['rt_acc_durw']:.6f}" for d in dss))
    for b in BUCKETS:
        print(f"idx {b:3s} rt {out['by_index_mean'][b]:.6f} smooth {out['by_index_smooth_mean'][b]:.6f} "
              f"clip4 {out['by_index_clip4_mean'][b]:.6f} | " +
              ' '.join(f"{d} {P[d]['by_index'][b]['rt_acc']:.4f}(n={P[d]['by_index'][b]['n']})" for d in dss))
    print('checks', json.dumps(chk))
    print(f"secs {out['secs']:.1f}")


if __name__ == '__main__':
    main()
