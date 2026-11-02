"""style_hmm: style-aware chord model (one chord HMM per style, averaged online) + combo's chooser.

Train: one ChordHMM (transitions + emissions) per chord-labelled train set (bach, folk, pop, pop909,
whichever exist) plus one pooled over all of them. Emissions are made robust to the 10 % mishearing:
B' = (1 - r) B + r (1 - B) / 11 (a heard pc is the true pc w.p. 1 - r, else one of the other 11).
Run: every HMM filters the HEARD melody causally in parallel. Each model's weight is its running
predictive likelihood of the heard beat notes, log w_s += log P_s(o_j | o_<j) (with forgetting rho,
so the mixture can switch style inside a song; Bayesian model averaging / switching). Her note
distribution for beat k is
    q = ngram_BMA(o) * (sum_s w_s prior_s @ B_s) ^ b_hmm
then the per-song cache n-gram, repeat and eps floor exactly as combo, and Duet's note h is chosen by
EXPECTED two-voice consonance with her absolute pitch (combo's chooser, wd doubling penalty).
Causal and streamable: beat j enters any memory only when fully heard (j < jstar = floor(k - D));
the partly heard beat jstar is read only up to the cut. No labels of the evaluated song are read.
"""
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from bench_systems.combo import CONS, CTM, UNI, _usable
from bench_systems.expected_support import SoftNGram
from model import ChordHMM, soprano_obs
from planner import Beat
from theory import ALTO_MID, CT, N_CHORDS

# frozen copy of combo.P (as of 19:18) so other agents' edits to combo.py do not move these numbers
P = dict(n=3, beta=4.0, b_hmm=1.0, cache_beta=3.0, orders=4, eps=0.05, wd=0.05, vl=0.0, vc=0.0,
         bma_floor=0.1)
P.update(dict(r=0.10, rho=0.97, pooled=1, styles=1, hmm_floor=0.05, tie=0))
P.update(json.loads(os.environ.get('DUET_SH', '{}')))


def _robust(m, r):
    B = (1 - r) * m.B + r * (1 - m.B) / 11.0
    m.B = B / B.sum(1, keepdims=True)
    m.logB = np.log(m.B)
    return m


class System:
    def __init__(self, train):
        sets = {k: _usable(v) for k, v in train.items() if k != 'essen'}
        sets = {k: v for k, v in sets.items() if len(v) >= 20}
        self.hmms, self.names = [], []
        if P['styles']:
            for k, v in sets.items():
                self.hmms.append(_robust(ChordHMM().fit(v), P['r']))
                self.names.append(k)
        if P['pooled'] or not self.hmms:
            self.hmms.append(_robust(ChordHMM().fit([s for v in sets.values() for s in v]), P['r']))
            self.names.append('all')
        nsongs = [s for k in train for s in _usable(train.get(k, []))]
        self.ng = SoftNGram(int(P['n']), P['beta']).fit(nsongs)
        self.ngs = [self.ng] + [SoftNGram(int(P['n']), P['beta']).fit(_usable(train[k]))
                                for k in train if len(_usable(train[k])) >= 20]
        self.note_guess = {}
        self.wlog = Counter()

    def run(self, song, cfg):
        t, M, K, sop = song['tonic'], song['M'], song['K'], song['sop']
        hmms = self.hmms
        S = len(hmms)
        pos = [b['pos'] for b in song['beats']]
        ztrue = [b['z'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        lo, hi = cfg.alto
        H = np.arange(lo, hi + 1)
        Hf = H.astype(float)
        orders = int(P['orders'])
        cb = P['cache_beta']
        rho, fl = P['rho'], P['hmm_floor']

        def rel(m):
            return None if m is None else (m - t) % 12

        o_full = [None] * K
        mid_full = [None] * K
        seen = Counter()
        cache = defaultdict(Counter)
        alpha = [None] * S              # per-style filtered chord posterior after last full beat
        logw_h = np.zeros(S)            # per-style running log predictive likelihood
        logw = np.zeros(len(self.ngs))
        heard = 0
        out, h_last, her_last = [], None, None
        for k in range(K):
            when = k - D
            jstar = math.floor(when)
            frac = when - jstar
            while heard < max(0, jstar):
                j = heard
                _, m = soprano_obs(sop, j)
                mid_full[j] = m
                o_full[j] = rel(m)
                for s, mod in enumerate(hmms):
                    pi = mod.init if j == 0 else mod.predict(alpha[s], pos[j], M)
                    if o_full[j] is not None:
                        pj = float(pi @ mod.B[:, o_full[j]])
                        logw_h[s] = rho * logw_h[s] + math.log((1 - fl) * pj + fl / 12)
                    alpha[s] = mod.update(pi, mod.loglik(o_full[j], 'long', cfg.lam))
                if o_full[j] is not None:
                    for i_m, ngm in enumerate(self.ngs):
                        pj = ngm.predict(o_full, j, pos[j])[o_full[j]]
                        logw[i_m] += math.log((1 - P['bma_floor']) * pj + P['bma_floor'] / 12)
                    seen[o_full[j]] += 1
                    for mm in range(1, orders + 1):
                        if j - mm < 0:
                            break
                        ctx = tuple(o_full[j - mm:j])
                        if None in ctx:
                            break
                        wgt = 0.3 if (seen[ctx[-1]] <= 1 and j > 8) else 1.0
                        cache[(mm, pos[j], ctx)][o_full[j]] += wgt
                heard += 1
            opart = None
            priors = []
            if jstar >= 0:
                _, mid = soprano_obs(sop, jstar, frac)
                opart = rel(mid)
                if mid is None:
                    for jj in range(jstar - 1, -1, -1):
                        if mid_full[jj] is not None:
                            mid = mid_full[jj]
                            break
                her_last = mid if mid is not None else her_last
            for s, mod in enumerate(hmms):
                if jstar >= 0:
                    pi = mod.init if jstar == 0 else mod.predict(alpha[s], pos[jstar], M)
                    post = mod.update(pi, mod.loglik(opart, 'long', cfg.lam))
                    for i in range(jstar + 1, k + 1):
                        post = mod.predict(post, pos[i], M)
                else:
                    post = mod.init
                    for i in range(1, k + 1):
                        post = mod.predict(post, pos[i], M)
                priors.append(post)
            ws = np.exp(logw_h - logw_h.max())
            ws = ws / ws.sum()
            m_hmm = sum(w_ * (p_ @ mod.B) for w_, p_, mod in zip(ws, priors, hmms))
            o = list(o_full[:max(jstar, 0)])
            if jstar >= 0:
                o.append(opart)
            o += [None] * (k - len(o))
            wm = np.exp(logw - logw.max())
            wm = wm / wm.sum()
            qn = sum(w_ * ngm.predict(o, k, pos[k]) for w_, ngm in zip(wm, self.ngs))
            q = qn * m_hmm ** P['b_hmm']
            q = q / q.sum()
            ctx = o[max(0, k - orders):k]
            for mm in range(1, orders + 1):
                if mm > len(ctx) or None in ctx[len(ctx) - mm:]:
                    break
                c = cache.get((mm, pos[k], tuple(ctx[len(ctx) - mm:])))
                if not c:
                    break
                v = np.zeros(12)
                for pc, n in c.items():
                    v[pc] = n
                q = (v + cb * q) / (v.sum() + cb)
            q = (1 - P['eps']) * q + P['eps'] / 12
            self.note_guess[k] = int(np.argmax(q))
            ref = her_last if her_last is not None else 72
            pcs = np.arange(12)
            her_abs = ref + ((pcs + t - ref + 6) % 12) - 6
            iv = np.abs(her_abs[None, :] - H[:, None]) % 12
            e2 = CONS[iv] @ q
            ed = UNI[iv] @ q
            sc = e2 - P['wd'] * ed - P['vc'] * np.abs(Hf - ALTO_MID)
            if h_last is not None:
                sc = sc - P['vl'] * np.abs(Hf - h_last)
            i = int(np.argmax(sc))
            h0 = int(H[i])
            hp = (h0 - t) % 12
            zs = [z for z in range(N_CHORDS) if hp in CT[z]]
            e4 = CTM @ q
            z0 = max(zs, key=lambda z: e4[z]) if zs else int(np.argmax(e4))
            bt = Beat(k=k, pos=pos[k], z_true=ztrue[k], z_hat=z0, q=float(e2[i]))
            if k >= cfg.J:
                bt.action, bt.h = 'sing', h0
                h_last = h0
            out.append(bt)
        self.wlog[self.names[int(np.argmax(logw_h))]] += 1
        return out


def build(train):
    return System(train)
