"""Melody-aware chord HMM: the emission is autoregressive, P(o_k | z_k, o_prev), where o_k is the pitch
class she holds longest in beat k (relative to the tonic) and o_prev her last heard note before beat k.
Trained on Bach sopranos with backoff to P(o_k | z_k). The forward filter over chords uses this
emission; the decision for beat k uses the predicted joint P(z_k, o_k | heard) with o_prev = the last
note heard by the decision time. note_guess = argmax of the predicted marginal P(o_k | heard).

Causal: for beat k only beats j < jstar fully and beat jstar up to frac (decision time k - D) are read;
no chord labels of the evaluated song are used (song['beats'] only gives bar positions here).
"""
import math
import os

import numpy as np

from model import ChordHMM, soprano_obs
from planner import Beat, Planner
from theory import CT, N_CHORDS

S, NONE = N_CHORDS, 12
VARIANT = os.environ.get('MELODY_HMM_VARIANT', 'pow16')
BETA = float(os.environ.get('MELODY_HMM_BETA', '5.0'))     # backoff strength (pseudo-counts)
IN_CT = np.array([[pc in CT[z] for pc in range(12)] for z in range(S)], float)   # (z, pc)


def longest(sop, k, frac=1.0, t=0):
    mid = soprano_obs(sop, k, frac)[1]
    return None if mid is None else (mid - t) % 12


class System:
    def __init__(self, train):
        self.model = ChordHMM().fit(train['bach'])
        c3 = np.zeros((S, 13, 12))
        for s in train['bach']:
            t, prev = s['tonic'], NONE
            for k, b in enumerate(s['beats']):
                o = longest(s['sop'], k, 1.0, t)
                if o is None:
                    continue
                if b['z'] >= 0:
                    c3[b['z'], prev, o] += 1
                prev = o
        c2 = c3.sum(1)                                            # (z, o)
        p2 = (c2 + 1.0) / (c2.sum(-1, keepdims=True) + 12.0)
        self.E = (c3 + BETA * p2[:, None, :]) / (c3.sum(-1, keepdims=True) + BETA)   # P(o | z, prev)
        self.logE = np.log(self.E)
        self.note_guess = {}

    def run(self, song, cfg):
        m, sop, t, M, K = self.model, song['sop'], song['tonic'], song['M'], song['K']
        pos = [b['pos'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        planner = Planner(t, cfg, m)
        obs = [None] * K          # filled lazily, only for beats fully heard by the time they are used
        prevs = [NONE] * K
        pi_full, alpha_full = [], []

        def full_step(j):        # beat j fully heard; called only when j < jstar of the current decision
            p = m.init if j == 0 else m.predict(alpha_full[j - 1], pos[j], M)
            pv = NONE if j == 0 else (obs[j - 1] if obs[j - 1] is not None else prevs[j - 1])
            prevs[j] = pv
            obs[j] = longest(sop, j, 1.0, t)
            pi_full.append(p)
            alpha_full.append(p if obs[j] is None else m.update(p, self.logE[:, pv, obs[j]]))

        out, joined, q_prev, h_last, her_last = [], False, 0.0, None, None
        for k in range(K):
            when = k - D
            jstar = math.floor(when)
            frac = when - jstar
            while len(alpha_full) < max(0, jstar):
                full_step(len(alpha_full))
            if jstar >= 0:
                mid = soprano_obs(sop, jstar, frac)[1]
                if mid is None and frac < 1:
                    for jj in range(jstar - 1, -1, -1):
                        mid = soprano_obs(sop, jj)[1]
                        if mid is not None:
                            break
                her_last = mid if mid is not None else her_last
            # chord belief for beat k and her last note at decision time
            if jstar < 0:
                prior, pv = m.init, NONE
                for i in range(1, k + 1):
                    prior = m.predict(prior, pos[i], M)
            else:
                p = m.init if jstar == 0 else m.predict(alpha_full[jstar - 1], pos[jstar], M)
                pvj = NONE if jstar == 0 else (obs[jstar - 1] if obs[jstar - 1] is not None else prevs[jstar - 1])
                op = longest(sop, jstar, frac, t)
                post = p if op is None else m.update(p, self.logE[:, pvj, op])
                for i in range(jstar + 1, k + 1):
                    post = m.predict(post, pos[i], M)
                prior = post
                pv = op if op is not None else pvj
            joint = prior[:, None] * self.E[:, pv, :]               # P(z_k, o_k | heard)
            pred = joint.sum(0)                                     # P(o_k | heard)
            if pv != NONE or jstar >= 0:
                self.note_guess[k] = int(np.argmax(pred))
            support = IN_CT @ pred                                  # P(her note in chord z)
            if VARIANT == 'chord':
                dec = prior
            elif VARIANT == 'support':
                dec = support / support.sum()
            elif VARIANT == 'prod':
                dec = prior * support
                dec = dec / dec.sum()
            elif VARIANT.startswith('pow'):                         # prior * support^a
                a = float(VARIANT[3:])
                dec = prior * support ** a
                dec = dec / dec.sum()
            else:
                raise ValueError(VARIANT)

            bt = Beat(k=k, pos=pos[k], z_true=-1)
            horizon = [m.trans_key(pos[i], M) for i in range(k + 1, min(K, k + cfg.N))]
            z0, h0, path = planner.plan(dec, horizon, h_last, her_last)
            q = float(dec[z0])
            bt.z_hat, bt.q, bt.plan = z0, q, path
            if not joined and k >= cfg.J and q >= cfg.q_min and q_prev >= cfg.q_min:
                joined = True
            if joined:
                if q >= cfg.q_min:
                    bt.action, bt.h = 'sing', h0
                elif h_last is not None and (h_last - t) % 12 in CT[z0]:
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
