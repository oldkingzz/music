"""Get better by listening: HMM + a soft guess of her next note from a melody model that adapts
within the song (per-song n-gram cache of heard beats, backed off to a Bach-trained n-gram),
optionally with the chord transitions adapted online from the HMM's own filtered soft counts.

The guess enters as a mixture likelihood log sum_pc P(pc) B[z, pc] (soft hint).
Causal: for beat k only beats j < jstar are used fully and beat jstar partially (frac);
no chord labels of the evaluated song are read (z_true is copied into Beat only for scoring).

Variants (module constants): BETA cache backoff strength, ORDERS cache contexts, LAM_H hint weight,
ADAPT_A pseudo-count weight for transition adaptation (0 = off).

  python bench_systems/online_adapt.py     # four-part score by how long Duet has listened
"""
import math
from collections import Counter, defaultdict

import numpy as np

from model import ChordHMM, soprano_obs
from planner import Beat, Planner
from theory import N_CHORDS

BETA = 1.0          # cache backoff: P_m = (c_m + BETA * P_{m-1}) / (N_m + BETA)
ORDERS = (1, 2, 3, 4)
LAM_H = 1.5
ADAPT_A = 0.0       # 0: chord transitions fixed; >0: pseudo-count of the Bach rows
BACH_N = 2
BACH_BETA = 2.0
import os as _os
if _os.environ.get('OA_NOCACHE'):
    ORDERS = ()
for _k in ('BETA', 'LAM_H', 'ADAPT_A', 'BACH_BETA'):
    if _os.environ.get('OA_' + _k):
        globals()[_k] = float(_os.environ['OA_' + _k])


def _o(song, j, frac=1.0):
    m = soprano_obs(song['sop'], j, frac)[1]
    return None if m is None else (m - song['tonic']) % 12


class BachNGram:
    """Interpolated backoff n-gram over beat pitch classes, keyed by bar position."""

    def __init__(self, n):
        self.n = n
        self.c = defaultdict(Counter)

    def fit(self, songs):
        for s in songs:
            o = [_o(s, k) for k in range(s['K'])]
            for k in range(s['K']):
                if o[k] is None:
                    continue
                for m in range(self.n + 1):
                    if k - m < 0:
                        break
                    self.c[(m, s['beats'][k]['pos'], tuple(o[k - m:k]))][o[k]] += 1
        return self

    def dist(self, ctx, pos):
        p = np.full(12, 1.0 / 12)
        for m in range(self.n + 1):
            if m > len(ctx):
                break
            c = self.c.get((m, pos, tuple(ctx[len(ctx) - m:])))
            if not c:
                break
            v = np.zeros(12)
            for pc, n in c.items():
                v[pc] = n
            p = (v + BACH_BETA * p) / (v.sum() + BACH_BETA)
        return p


class System:
    def __init__(self, train):
        self.model = ChordHMM().fit(train['bach'])
        self.ng = BachNGram(BACH_N).fit(train['bach'])
        self.note_guess = {}
        self.listened = []          # (k, n, d) four-part per beat, for the listening curve

    # ---------------------------------------------------------------- her next note
    def melody_dist(self, o, k, pos_k, cache):
        ctx = o[max(0, k - max(max(ORDERS, default=0), BACH_N)):k]
        if any(x is None for x in ctx[-1:]):
            ctx_b = []
        else:
            ctx_b = ctx
        p = self.ng.dist(ctx_b, pos_k)
        for m in ORDERS:
            if m > len(ctx) or any(x is None for x in ctx[len(ctx) - m:]):
                break
            c = cache.get((m, pos_k, tuple(ctx[len(ctx) - m:])))
            if not c:
                break
            v = np.zeros(12)
            for pc, n in c.items():
                v[pc] = n
            p = (v + BETA * p) / (v.sum() + BETA)
        return p

    @staticmethod
    def add_cache(cache, o, j, pos_j):
        if o[j] is None:
            return
        for m in ORDERS:
            if j - m < 0:
                break
            ctx = tuple(o[j - m:j])
            if any(x is None for x in ctx):
                break
            cache[(m, pos_j, ctx)][o[j]] += 1

    # ---------------------------------------------------------------- the loop (planner.run 'hmm')
    def run(self, song, cfg):
        model = self.model
        t, M, K = song['tonic'], song['M'], song['K']
        pos = [b['pos'] for b in song['beats']]
        ztrue = [b['z'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        planner = Planner(t, cfg, model)
        lam = cfg.lam
        logB = model.logB
        keys = sorted({model.trans_key(p, M) for p in range(max(M, 4))})
        base = {key: model.mat(key) for key in keys}
        Aad = dict(base)
        C = {key: np.zeros((N_CHORDS, N_CHORDS)) for key in keys}

        def ll_pc(pc):
            return np.zeros(N_CHORDS) if pc is None else logB[:, pc]

        def predict(post, p):
            return post @ Aad[model.trans_key(p, M)]

        o = [None] * K                 # fully heard beats only
        alpha = []                     # filtered posterior after each fully heard beat
        cache = defaultdict(Counter)
        heard = 0                      # beats 0..heard-1 are fully heard and processed

        if ADAPT_A > 0:
            class _M:                  # adapted transitions for the planner
                def mat(self_, key):
                    return Aad[key]
            planner.model = _M()

        out, joined, q_prev, h_last, her_last = [], False, 0.0, None, None
        for k in range(K):
            when = k - D
            jstar = math.floor(when)
            frac = when - jstar
            # absorb newly completed beats: j < jstar is fully heard
            while heard < max(0, jstar):
                j = heard
                o[j] = _o(song, j)
                pi = model.init if j == 0 else predict(alpha[j - 1], pos[j])
                lj = ll_pc(o[j])
                alpha.append(model.update(pi, lj))
                self.add_cache(cache, o, j, pos[j])
                if ADAPT_A > 0 and j >= 1:
                    key = model.trans_key(pos[j], M)
                    xi = alpha[j - 1][:, None] * Aad[key] * np.exp(lj - lj.max())[None, :]
                    C[key] += xi / xi.sum()
                    Aad[key] = (ADAPT_A * base[key] + C[key])
                    Aad[key] = Aad[key] / Aad[key].sum(1, keepdims=True)
                    planner._trans_cache.pop(key, None)
                heard += 1
            if jstar >= 0:
                _, mid = soprano_obs(song['sop'], jstar, frac)
                if mid is None and frac < 1:
                    for jj in range(jstar - 1, -1, -1):
                        mid = soprano_obs(song['sop'], jj)[1]
                        if mid is not None:
                            break
                her_last = mid if mid is not None else her_last
            if jstar < 0:
                prior = model.init
                for i in range(1, k + 1):
                    prior = predict(prior, pos[i])
            else:
                pi = model.init if jstar == 0 else predict(alpha[jstar - 1], pos[jstar])
                opart = _o(song, jstar, frac)
                post = model.update(pi, ll_pc(opart))
                for i in range(jstar + 1, k + 1):
                    post = predict(post, pos[i])
                prior = post
                if jstar >= 1:
                    oc = list(o[:jstar]) + [opart] + [None] * (k - jstar - 1)
                    P = self.melody_dist(oc, k, pos[k], cache)
                    self.note_guess[k] = int(np.argmax(P))
                    llh = LAM_H * np.log(model.B @ P + 1e-12)
                    prior = model.update(prior, llh)

            bt = Beat(k=k, pos=pos[k], z_true=ztrue[k])
            horizon = [model.trans_key(pos[i], M) for i in range(k + 1, min(K, k + cfg.N))]
            z0, h0, path = planner.plan(prior, horizon, h_last, her_last)
            q = float(prior[z0])
            bt.z_hat, bt.q, bt.pi, bt.plan = z0, q, [round(float(x), 4) for x in prior], path
            if not joined and k >= cfg.J and q >= cfg.q_min and q_prev >= cfg.q_min:
                joined = True
            if joined:
                if q >= cfg.q_min:
                    bt.action, bt.h = 'sing', h0
                elif h_last is not None and (h_last - t) % 12 in __import__('theory').CT[z0]:
                    bt.action, bt.h = 'hold', h_last
                else:
                    bt.action = 'silent'
                if bt.h >= 0:
                    h_last = bt.h
            q_prev = q
            out.append(bt)
        return out


def build(train):
    return System(train)


def _curve(system_name='online_adapt'):
    """Four-part score on val bach by beats listened: [8,16), [16,32), [32,inf)."""
    import importlib
    from bench import config, eval_sets, train_sets
    from dataclasses import replace
    from theory import CT
    cfg = config()
    sysm = importlib.import_module(f'bench_systems.{system_name}').build(train_sets())
    bins = [(8, 16), (16, 32), (32, 10 ** 9)]
    acc = {b: [0.0, 0.0] for b in bins}
    for s in eval_sets('val')['bach']:
        c = replace(cfg, bpm=s['bpm']) if 'bpm' in s else cfg
        for b in sysm.run(s, c):
            if b.h < 0 or b.z_hat < 0:
                continue
            for lo, hi in bins:
                if lo <= b.k < hi:
                    for on, off, m in s['sop']:
                        ov = min(off, b.k + 1) - max(on, b.k)
                        if ov > 1e-9:
                            acc[(lo, hi)][1] += ov
                            acc[(lo, hi)][0] += ov * ((m - s['tonic']) % 12 in CT[b.z_hat])
    return {f'{lo}-{hi if hi < 10**9 else "end"}': (n / d if d else float('nan'), d) for (lo, hi), (n, d) in acc.items()}


if __name__ == '__main__':
    import sys
    for name in sys.argv[1:] or ['online_adapt', 'repeat', 'hmm']:
        print(name, {k: (round(v[0], 3), round(v[1], 1)) for k, v in _curve(name).items()})
