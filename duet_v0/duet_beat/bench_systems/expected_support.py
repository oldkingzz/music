"""Decide by expected score, not by the most likely chord.

At the decision time for beat k (k - D) Duet forms a distribution q over the pitch class she will
sing in beat k (time share), from what it has heard so far:
  hmm    the causal chord belief for beat k pushed through the emission table: prior @ B
  ngram  a soft n-gram over her melody (last heard beat notes + bar position, interpolated backoff,
         targets are time-share 12-vectors), trained on Bach sopranos
  repeat when exp_ceiling.Repeat finds an earlier place in THIS song whose last 2 beats match,
         what she sang next there (mixed in with weight w_rep)
Then it picks the (chord z, note h) pair, h a chord tone of z in the alto range and below her latest
note, maximizing
    P(her pc in CT[z])                       expected four-part support
  + w2 * P(her pc consonant with h)          expected two-part consonance (Duet below her)
  - vl * |h - h_prev| - vc * |h - alto mid|  small voice-leading penalty
Causal: uses soprano_obs(sop, j) only for j < jstar = floor(k - D) and soprano_obs(sop, jstar, frac);
no chord labels of the evaluated song. Joining is as in planner.run (k >= J and confidence >= q_min
twice), with confidence = the expected four-part support of the chosen chord.
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

from exp_ceiling import Repeat
from model import ChordHMM, soprano_obs
from planner import Beat
from theory import ALTO_MID, CONSONANT, CT, N_CHORDS

P = dict(n=3, beta=4.0, combine='prod', b_hmm=1.0, w_rep=0.7, w2=0.5, vl=0.02, vc=0.005, rep_C=2)
P.update(json.loads(os.environ.get('DUET_ES', '{}')))

CTM = np.zeros((N_CHORDS, 12))
for _z in range(N_CHORDS):
    CTM[_z, list(CT[_z])] = 1.0
CONS = np.array([1.0 if i in CONSONANT else 0.0 for i in range(12)])


def _rel_vec(sop, k, t, frac=1.0):
    vec, mid = soprano_obs(sop, k, frac)
    return np.roll(vec, -t), (None if mid is None else (mid - t) % 12)


class SoftNGram:
    """P(her time-share pitch-class vector in beat k | last m beat notes, bar position), m = n..0,
    interpolated: p_m = lam * counts_m / N_m + (1 - lam) * p_{m-1}, lam = N_m / (N_m + beta)."""

    def __init__(self, n, beta):
        self.n, self.beta = n, beta
        self.c = defaultdict(lambda: np.zeros(12))

    def fit(self, songs):
        for s in songs:
            t, K = s['tonic'], s['K']
            vs, o = zip(*(_rel_vec(s['sop'], k, t) for k in range(K)))
            for k in range(K):
                v = vs[k]
                if v.sum() <= 0:
                    continue
                v = v / v.sum()
                pos = s['beats'][k]['pos']
                for m in range(0, min(self.n, k) + 1):
                    self.c[(m, pos, tuple(o[k - m:k]))] += v
        self.c = dict(self.c)
        return self

    def predict(self, o, k, pos):
        p = np.full(12, 1.0 / 12)
        for m in range(0, min(self.n, k) + 1):
            c = self.c.get((m, pos, tuple(o[k - m:k])))
            if c is None:
                break
            N = c.sum()
            lam = N / (N + self.beta)
            p = lam * c / N + (1 - lam) * p
        return p


class System:
    def __init__(self, train):
        self.model = ChordHMM().fit(train['bach'])
        self.ng = SoftNGram(int(P['n']), P['beta']).fit(train['bach'])
        self.note_guess = {}
        self.stats = {'beats': 0, 'sung': 0, 'rep': 0}

    def run(self, song, cfg):
        model, t, M, K, sop = self.model, song['tonic'], song['M'], song['K'], song['sop']
        pos = [b['pos'] for b in song['beats']]
        ztrue = [b['z'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        rep = Repeat(song, int(P['rep_C']), D) if P['w_rep'] > 0 else None
        lo, hi = cfg.alto
        st = [(z, h) for z in range(N_CHORDS) for h in range(lo, hi + 1) if (h - t) % 12 in CT[z]]
        Z = np.array([s[0] for s in st])
        H = np.array([s[1] for s in st])
        Hf = H.astype(float)
        hpc = (H - t) % 12
        # cons[i, pc] = 1 if her pc above Duet's note H[i] is consonant
        cons = CONS[(np.arange(12)[None, :] - hpc[:, None]) % 12]

        def ll(obs):
            return model.loglik(obs, 'long', cfg.lam)

        full = [_rel_vec(sop, j, t) for j in range(K)]          # used only for j < jstar below
        o_full = [x[1] for x in full]
        # pi[j] = chord belief for beat j before hearing it, built only from fully heard beats < j
        pi, alpha = [], None
        for j in range(K):
            p = model.init if j == 0 else model.predict(alpha, pos[j], M)
            pi.append(p)
            alpha = model.update(p, ll(o_full[j]))

        out, joined, q_prev, h_last, her_last = [], False, 0.0, None, None
        for k in range(K):
            when = k - D
            jstar = math.floor(when)
            frac = when - jstar
            o = list(o_full[:max(jstar, 0)])
            if jstar >= 0:
                _, pc_part = _rel_vec(sop, jstar, t, frac)
                _, mid = soprano_obs(sop, jstar, frac)
                if mid is None and frac < 1:
                    for jj in range(jstar - 1, -1, -1):
                        _, mid = soprano_obs(sop, jj)
                        if mid is not None:
                            break
                her_last = mid if mid is not None else her_last
                o.append(pc_part)
                post = model.update(pi[jstar], ll(pc_part))
                for i in range(jstar + 1, k + 1):
                    post = model.predict(post, pos[i], M)
                prior = post
            else:
                prior = pi[k]
            o += [None] * (k - len(o))
            # ---- her note distribution for beat k
            m_hmm = prior @ model.B
            m_ng = self.ng.predict(o, k, pos[k])
            if P['combine'] == 'prod':
                q = m_ng * m_hmm ** P['b_hmm']
            elif P['combine'] == 'mix':
                q = 0.5 * m_ng + 0.5 * m_hmm
            elif P['combine'] == 'ngram':
                q = m_ng.copy()
            else:
                q = m_hmm.copy()
            q = q / q.sum()
            g = rep(k, jstar, frac) if rep is not None else None
            if g is not None:
                rv = np.roll(g[0], -t)
                if rv.sum() > 0:
                    q = P['w_rep'] * rv / rv.sum() + (1 - P['w_rep']) * q
                    self.stats['rep'] += 1
            self.note_guess[k] = int(np.argmax(q))
            # ---- choose (z, h) by expected score
            e4 = CTM @ q
            e2 = cons @ q
            sc = e4[Z] + P['w2'] * e2 - P['vc'] * np.abs(Hf - ALTO_MID)
            if h_last is not None:
                sc = sc - P['vl'] * np.abs(Hf - h_last)
            if her_last is not None:
                ok = H <= her_last - 1
                if ok.any():
                    sc = np.where(ok, sc, -np.inf)
            i = int(np.argmax(sc))
            z0, h0 = int(Z[i]), int(H[i])
            conf = float(e4[z0])
            bt = Beat(k=k, pos=pos[k], z_true=ztrue[k], z_hat=z0, q=conf)
            if not joined and k >= cfg.J and conf >= cfg.q_min and q_prev >= cfg.q_min:
                joined = True
            if joined:
                if conf >= cfg.q_min:
                    bt.action, bt.h = 'sing', h0
                elif h_last is not None and (h_last - t) % 12 in CT[z0]:
                    bt.action, bt.h = 'hold', h_last
                else:
                    bt.action = 'silent'
                if bt.h >= 0:
                    h_last = bt.h
            if k >= cfg.J:
                self.stats['beats'] += 1
                self.stats['sung'] += int(bt.h >= 0)
            q_prev = conf
            out.append(bt)
        return out


def build(train):
    return System(train)
