"""combo: best non-neural combination (tuned defaults: repeat OFF w_rep=0, it hurt val; no voice-leading
penalty vl=vc=0; n-gram = online Bayesian average of one SoftNGram per train set + all sets), decided by EXPECTED two-voice consonance with her true note.

At the decision time for beat k (k - D) Duet forms q = P(her pitch class in beat k) from what it has
heard (with 10 % mishearing) so far:
  prior   SoftNGram over her beat notes (expected_support's), trained on every train melody set,
          times the causal chord-HMM belief pushed through the emission table (prior @ B)^b_hmm
  cache   per-song n-gram of heard beats (online_adapt), orders 1..4, backed off onto the prior;
          a context note that is an isolated outlier (never heard elsewhere in the song) is down-weighted
  repeat  tolerant long-context repeat match (long_repeat): earlier beats j at the same bar position
          whose last C heard beats agree with the last C heard beats now, allowing up to `mis` mismatched
          beats (mishearing), exact or transposed; votes mixed in by agreement
  floor   q = (1 - eps) q + eps / 12
Then it picks Duet's MIDI note h in the alto range maximizing
    E[consonant(|her_abs - h| mod 12)] - wd * E[unison/octave] - vl |h - h_prev| - vc |h - alto mid|
her_abs for each pc is placed nearest her latest heard pitch (so above/below is handled exactly).
Duet always sings from beat J (silence is never good). z_hat = the chord containing h with most mass.
Causal and streamable: beat j enters the song memory only once fully heard (j < jstar); the partly
heard beat jstar is read up to the cut; nothing of the evaluated song's labels is read.
"""
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from bench_systems.expected_support import SoftNGram
from model import ChordHMM, soprano_obs
from planner import Beat
from theory import ALTO_MID, CONSONANT, CT, N_CHORDS

P = dict(n=3, beta=4.0, b_hmm=1.0, cache_beta=3.0, orders=4, w_rep=0.0, Cs=(8, 6, 4, 3), mis=1,
         tw=0.7, agree=0.4, eps=0.05, wd=0.05, vl=0.0, vc=0.0, below=0.0, ng_sets='all',
         hmm_sets='bach', diatonic=0, uni_g=0.0, vec_cache=0, nopos=0, bma=1, bma_floor=0.1)
P.update(json.loads(os.environ.get('DUET_COMBO', '{}')))

CONS = np.array([1.0 if i in CONSONANT else 0.0 for i in range(12)])
UNI = np.zeros(12)
UNI[0] = 1.0
CTM = np.zeros((N_CHORDS, 12))
for _z in range(N_CHORDS):
    CTM[_z, list(CT[_z])] = 1.0
MAJ = {0, 2, 4, 5, 7, 9, 11}


def _usable(songs):
    return [s for s in songs if s.get('sop') and 'tonic' in s and s.get('beats') and s.get('K')]


class System:
    def __init__(self, train):
        hs = ['bach'] if P['hmm_sets'] == 'bach' else [k for k in train if k != 'essen']
        hsongs = [s for k in hs for s in _usable(train.get(k, []))]
        self.model = ChordHMM().fit(hsongs)
        ns = list(train) if P['ng_sets'] == 'all' else ['bach']
        nsongs = [s for k in ns for s in _usable(train.get(k, []))]
        self.ng = SoftNGram(int(P['n']), P['beta']).fit(nsongs)
        # one n-gram per train set, averaged online by how well each predicts THIS song (causal BMA)
        self.ngs = [self.ng] + [SoftNGram(int(P['n']), P['beta']).fit(_usable(train[k]))
                                for k in train if len(_usable(train[k])) >= 20] if P['bma'] else [self.ng]
        self.note_guess = {}

    def run(self, song, cfg):
        model, t, M, K, sop = self.model, song['tonic'], song['M'], song['K'], song['sop']
        pos = [b['pos'] for b in song['beats']]
        ztrue = [b['z'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        lo, hi = cfg.alto
        H = np.arange(lo, hi + 1)
        if P['diatonic']:
            H = H[np.array([(h - t) % 12 in MAJ for h in H])]
        Hf = H.astype(float)
        orders = int(P['orders'])
        Cs = tuple(int(c) for c in P['Cs'])
        cb = P['cache_beta']

        def rel(m):
            return None if m is None else (m - t) % 12

        o_full = [None] * K           # heard longest pc per beat (fully heard beats only)
        o_cut = [None] * K            # same beat cut at frac (for repeat matching against partial ctx)
        mid_full = [None] * K
        seen = Counter()
        uni = np.zeros(12)
        cache = defaultdict(Counter)
        alpha_prev = None             # filtered chord posterior after the last fully heard beat
        logw = np.zeros(len(self.ngs))
        heard = 0
        out, h_last, her_last = [], None, None
        for k in range(K):
            when = k - D
            jstar = math.floor(when)
            frac = when - jstar
            # ---- absorb newly completed beats (j < jstar): streaming memory
            while heard < max(0, jstar):
                j = heard
                _, m = soprano_obs(sop, j)
                mid_full[j] = m
                o_full[j] = rel(m)
                o_cut[j] = rel(soprano_obs(sop, j, frac)[1]) if frac > 1e-9 else None
                pi = model.init if j == 0 else model.predict(alpha_prev, pos[j], M)
                alpha_prev = model.update(pi, model.loglik(o_full[j], 'long', cfg.lam))
                if o_full[j] is not None and len(self.ngs) > 1:
                    for i_m, ngm in enumerate(self.ngs):
                        pj = ngm.predict(o_full, j, pos[j])[o_full[j]]
                        logw[i_m] += math.log((1 - P['bma_floor']) * pj + P['bma_floor'] / 12)
                if o_full[j] is not None:
                    seen[o_full[j]] += 1
                    vj = np.roll(soprano_obs(sop, j)[0], -t)
                    uni += vj / max(vj.sum(), 1e-9)
                    for mm in range(1, orders + 1):
                        if j - mm < 0:
                            break
                        ctx = tuple(o_full[j - mm:j])
                        if None in ctx:
                            break
                        wgt = 1.0
                        if seen[ctx[-1]] <= 1 and j > 8:      # isolated outlier context: likely misheard
                            wgt = 0.3
                        key = (mm, -1 if P['nopos'] else pos[j], ctx)
                        if P['vec_cache']:
                            for pc in range(12):
                                if vj[pc] > 0:
                                    cache[key][pc] += wgt * vj[pc] / vj.sum()
                        else:
                            cache[key][o_full[j]] += wgt
                heard += 1
            # ---- partly heard beat jstar
            opart = None
            if jstar >= 0:
                _, mid = soprano_obs(sop, jstar, frac)
                opart = rel(mid)
                if mid is None:
                    for jj in range(jstar - 1, -1, -1):
                        if mid_full[jj] is not None:
                            mid = mid_full[jj]
                            break
                her_last = mid if mid is not None else her_last
                pi = model.init if jstar == 0 else model.predict(alpha_prev, pos[jstar], M)
                post = model.update(pi, model.loglik(opart, 'long', cfg.lam))
                for i in range(jstar + 1, k + 1):
                    post = model.predict(post, pos[i], M)
                prior = post
            else:
                prior = model.init
                for i in range(1, k + 1):
                    prior = model.predict(prior, pos[i], M)
            o = list(o_full[:max(jstar, 0)])
            if jstar >= 0:
                o.append(opart)
            o += [None] * (k - len(o))
            # ---- her note distribution
            if len(self.ngs) > 1:
                wm = np.exp(logw - logw.max())
                wm = wm / wm.sum()
                qn = sum(w_ * ngm.predict(o, k, pos[k]) for w_, ngm in zip(wm, self.ngs))
            else:
                qn = self.ng.predict(o, k, pos[k])
            q = qn * (prior @ model.B) ** P['b_hmm']
            if P['uni_g'] > 0 and uni.sum() > 0:
                q = q * ((uni + 0.5) / (uni.sum() + 6)) ** P['uni_g']
            q = q / q.sum()
            ctx = o[max(0, k - orders):k]
            for mm in range(1, orders + 1):
                if mm > len(ctx) or None in ctx[len(ctx) - mm:]:
                    break
                c = cache.get((mm, -1 if P['nopos'] else pos[k], tuple(ctx[len(ctx) - mm:])))
                if not c:
                    break
                v = np.zeros(12)
                for pc, n in c.items():
                    v[pc] = n
                q = (v + cb * q) / (v.sum() + cb)
            if P['w_rep'] > 0 and jstar >= 2:
                g = self._repeat(o_full, o_cut, opart, pos, k, jstar, Cs)
                if g is not None:
                    rv, share = g
                    w = P['w_rep'] * min(1.0, share / max(P['agree'], 1e-9)) if share >= P['agree'] * 0.5 else 0.0
                    q = w * rv + (1 - w) * q
            q = (1 - P['eps']) * q + P['eps'] / 12
            self.note_guess[k] = int(np.argmax(q))
            # ---- choose h by expected consonance with her absolute pitch
            ref = her_last if her_last is not None else 72
            pcs = np.arange(12)
            her_abs = ref + ((pcs + t - ref + 6) % 12) - 6          # nearest to ref with that pc
            iv = np.abs(her_abs[None, :] - H[:, None]) % 12           # |H| x 12
            e2 = CONS[iv] @ q
            ed = UNI[iv] @ q
            sc = e2 - P['wd'] * ed - P['vc'] * np.abs(Hf - ALTO_MID)
            if h_last is not None:
                sc = sc - P['vl'] * np.abs(Hf - h_last)
            if her_last is not None:
                sc = sc - P['below'] * (H >= her_last)
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
        return out

    def _repeat(self, o_full, o_cut, opart, pos, k, jstar, Cs):
        """Tolerant repeat: context = o_full[jstar-C+1 .. jstar-1] + opart (cut beat)."""
        mis = int(P['mis'])
        for C in Cs:
            if jstar < C + 1:
                continue
            cur = o_full[jstar - C + 1:jstar] + [opart]
            if sum(x is not None for x in cur) < max(2, C - 1):
                continue
            allow = mis if C >= 4 else 0
            votes = np.zeros(12)
            for j in range(jstar - 1, C, -1):         # target beat j fully heard; its cut ctx beat j-1
                if pos[j] != pos[k] or o_full[j] is None:
                    continue
                past = o_full[j - C:j - 1] + [o_cut[j - 1]]
                bad = sum(a != b for a, b in zip(cur, past))
                w = 1.0 + 0.1 * (j / max(1, k))
                if bad <= allow:
                    votes[o_full[j]] += w * (1.0 if bad == 0 else 0.6)
                    continue
                # transposed (sequence): same intervals
                if P['tw'] > 0 and None not in cur and None not in past and len(set(cur)) >= 2:
                    d = [(a - b) % 12 for a, b in zip(cur, past)]
                    top, n = Counter(d).most_common(1)[0]
                    if top != 0 and C - n <= allow:
                        votes[(o_full[j] + top) % 12] += P['tw'] * w * (1.0 if n == C else 0.6)
            if votes.sum() > 0:
                return votes / votes.sum(), float(votes.max() / votes.sum())
        return None


def build(train):
    return System(train)
