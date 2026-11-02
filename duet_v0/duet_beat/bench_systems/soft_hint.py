"""HMM + a SOFT guess of her next note, combined with the Repeat hint.

A melody model (interpolated Witten-Bell n-gram over relative pitch classes; context = bar position
of beat k + her last 1-2 heard notes, the last one cut at the decision time) gives P(pc) for beat k.
If Repeat (exp_ceiling, last 2 beats match earlier in this song) fires, its guess is used as heard
(as in the 'repeat' system). Otherwise the chord prior is updated with a soft likelihood:
  mix    ll(z) = w * log sum_pc P(pc) B[z, pc]
  ratio  ll(z) = log sum_pc B[z, pc] (P(pc) / P0(pc))^w    P0 = P(pc | bar position) (product of experts)
The soft likelihood goes through planner.run's hint by a model subclass whose loglik returns the
pending soft likelihood on the call that follows the hint. Causal: only beats < jstar fully and
beat jstar up to frac; never the evaluated song's labels.
"""
import os

import numpy as np

from exp_ceiling import Repeat
from exp_more import notes
from model import ChordHMM, soprano_obs
from planner import run

MODE = os.environ.get('SOFT_MODE', 'mix')
W = float(os.environ.get('SOFT_W', '3.0'))
ORDER = int(os.environ.get('SOFT_N', '2'))
BETA = float(os.environ.get('SOFT_BETA', '1.0'))
USE_REPEAT = os.environ.get('SOFT_REPEAT', '1') == '1'


class Melody:
    """Interpolated (Witten-Bell) n-gram: P(pc_k | pos_k, pc_{k-1}, ..., pc_{k-n})."""

    def __init__(self, n=2, beta=1.0):
        self.n, self.beta, self.c = n, beta, {}

    def fit(self, songs):
        for s in songs:
            if 'beats' not in s or 'sop' not in s:
                continue
            o = notes(s)
            for k in range(1, s['K']):
                if o[k] is None:
                    continue
                pos = s['beats'][k]['pos']
                for m in range(self.n + 1):
                    ctx = tuple(o[k - i] if k - i >= 0 else None for i in range(m, 0, -1))
                    self.c.setdefault((m, pos, ctx), np.zeros(12))[o[k]] += 1
        return self

    def predict(self, o, k, pos):
        p = np.full(12, 1.0 / 12)
        base = p
        for m in range(self.n + 1):
            ctx = tuple(o[k - i] if k - i >= 0 else None for i in range(m, 0, -1))
            c = self.c.get((m, pos, ctx))
            if c is None:
                break
            N, T = c.sum(), (c > 0).sum()
            lam = N / (N + self.beta * T)
            p = lam * c / N + (1 - lam) * p
            if m == 0:
                base = p
        return p, base


class SoftHMM(ChordHMM):
    _pending = None

    def loglik(self, obs, mode, lam=1.0):
        if self._pending is not None:
            ll, self._pending = self._pending, None
            return ll
        return super().loglik(obs, mode, lam)


class System:
    def __init__(self, train):
        self.model = SoftHMM().fit(train['bach'])
        mel = [s for name in ('bach', 'folk') for s in train.get(name, [])]
        self.mel = Melody(ORDER, BETA).fit(mel)
        self.note_guess = {}

    def soft_ll(self, P, P0):
        B = self.model.B
        if MODE == 'mix':
            return W * np.log(B @ P)
        r = (P / P0) ** W
        return np.log(B @ (r / r.sum()))

    def run(self, song, cfg):
        t = song['tonic']
        pos = [b['pos'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        rep = Repeat(song, 2, D) if USE_REPEAT else None
        model = self.model

        def hint(k, jstar, frac):
            if jstar < 1:
                return None
            if rep is not None:
                g = rep(k, jstar, frac)
                if g is not None:
                    if g[1] is not None:
                        self.note_guess[k] = (g[1] - t) % 12
                    return g
            o = []
            for j in range(jstar + 1):                 # heard beats; the last one up to the decision
                mid = soprano_obs(song['sop'], j, frac if j == jstar else 1.0)[1]
                o.append(None if mid is None else (mid - t) % 12)
            o += [None] * (k - jstar)
            P, P0 = self.mel.predict(o, k, pos[k])
            self.note_guess[k] = int(np.argmax(P))
            model._pending = self.soft_ll(P, P0)       # consumed by the very next loglik call
            return None, None
        return run(song, model, cfg, 'hmm', hint=hint)


def build(train):
    return System(train)
