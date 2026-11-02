"""HMM + a smarter 'remember the song' guess of her next beat (planner.run hint).

At decision time k - D, take her melody of the last C beats before the decision (sampled 12x per
beat, the partly heard beat included up to the cut) and look for every earlier place j in THIS song
(j fully heard, same bar position) whose C beats before j - D match it:
  exact       same pitches and rhythm
  transposed  same rhythm and intervals at another pitch (a sequence); the guess is shifted too
The longest C that has any match wins (C in CS, longest first); all matches at that length vote
(exact weight 1, transposed TW, more recent slightly more). The winner fires only if its share of
the vote is >= AGREE. With no match, a confidence-gated n-gram (trained on Bach) may fire instead.
Config mode is 'long', so the HMM only sees the guessed longest pitch: a hard guess, gated.
Causal: uses only her melody before k - D; no labels of the evaluated song.
"""
import math
from collections import Counter

import numpy as np

from exp_more import NGram
from model import ChordHMM, soprano_obs
from planner import run

GRID = 12
CS = (8, 6, 4, 3, 2)        # context lengths in beats, longest first
TW = 0.7                    # vote weight of a transposed match
TRANSPOSE = True
MIN_PITCHES_T = 2           # transposed match needs >= this many distinct pitches in the context
AGREE = 0.5                 # winner's vote share needed to fire
NG_FALLBACK = True
NG_CONF = 0.45              # n-gram fires only if its top note has this share of the counts
NG_MIN = 20                 # ... and the context was seen this often


class Hint:
    def __init__(self, song, D, ng, note_guess):
        self.sop, self.D, self.t = song['sop'], D, song['tonic']
        self.pos = [b['pos'] for b in song['beats']]
        self.K = song['K']
        self.ng, self.note_guess = ng, note_guess
        self.cD = math.ceil(D)
        self.phi = self.cD - D                     # cut k - D = (k - cD) + phi
        self.pad = (max(CS) + self.cD + 1) * GRID
        n = (self.K + 1) * GRID
        g = np.zeros(self.pad + n, dtype=int)
        xs = self.phi + (np.arange(n) + 0.5) / GRID
        for on, off, m in self.sop:
            g[self.pad + np.nonzero((xs >= on) & (xs < off))[0]] = m
        self.g = g
        self.full = [soprano_obs(self.sop, j) for j in range(self.K)]
        self.rel = [None if mid is None else (mid - self.t) % 12 for _, mid in self.full]

    def ctx(self, k, C):
        """Samples of [k - D - C, k - D)."""
        a = self.pad + (k - self.cD - C) * GRID
        return self.g[a:a + C * GRID]

    def match(self, k, jstar):
        for C in CS:
            if jstar < C:
                continue
            c = self.ctx(k, C)
            if not c.any():
                continue
            nz = c > 0
            distinct = len(set(c[nz].tolist()))
            votes = Counter()
            for j in range(jstar - 1, C - 1, -1):
                if self.pos[j] != self.pos[k] or self.full[j][1] is None:
                    continue
                d = self.ctx(j, C)
                w = 1.0 + 0.1 * (j / max(1, k))
                if np.array_equal(d, c):
                    votes[self.full[j][1]] += w
                elif TRANSPOSE and distinct >= MIN_PITCHES_T and np.array_equal(d > 0, nz):
                    diff = c[nz] - d[nz]
                    if (diff == diff[0]).all():
                        votes[self.full[j][1] + int(diff[0])] += TW * w
            if votes:
                mid, w = votes.most_common(1)[0]
                if w / sum(votes.values()) >= AGREE:
                    return mid
                return None
        return None

    def __call__(self, k, jstar, frac):
        if jstar < 1:
            return None
        mid = self.match(k, jstar)
        if mid is None and NG_FALLBACK:
            o = [self.rel[j] for j in range(jstar)]
            pm = soprano_obs(self.sop, jstar, frac)[1]
            o.append(None if pm is None else (pm - self.t) % 12)
            o += [None] * (k - jstar)
            for m in range(self.ng.n, -1, -1):
                cnt = self.ng.c.get((m, self.pos[k], tuple(o[max(0, k - m):k])))
                if cnt:
                    tot = sum(cnt.values())
                    pc, n = cnt.most_common(1)[0]
                    if tot >= NG_MIN and n / tot >= NG_CONF:
                        mid = 60 + (pc + self.t) % 12
                    break
        if mid is None:
            return None
        self.note_guess[k] = (mid - self.t) % 12
        vec = np.zeros(12)
        vec[mid % 12] = 1.0
        return vec, mid


class System:
    def __init__(self, train):
        self.model = ChordHMM().fit(train['bach'])
        self.ng = NGram(1).fit(train['bach'])
        self.note_guess = {}

    def run(self, song, cfg):
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        return run(song, self.model, cfg, 'hmm', hint=Hint(song, D, self.ng, self.note_guess))


def build(train):
    return System(train)
