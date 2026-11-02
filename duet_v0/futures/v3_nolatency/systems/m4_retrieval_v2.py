"""m4_retrieval_v2 = v1 retrieval + a causal chord-continuity filter (product of experts).

Extra history use: chords last; so the chord Duet believed under her previous note, pushed through a
note-to-note chord transition table (learned from training songs, separately for "same beat" and
"new beat at bar position b"), is a second opinion, multiplied with the retrieval guess:
    post_i  ~  p_retr_i * (post_{i-1} @ T[cond_i] / marg[cond_i]) ** GAMMA
With DUET_HIST = H >= 0 the filter is re-run from scratch over only the notes that started within the
last H beats, so no information older than H beats is used (except that each note's own retrieval
query looks back at most NMAX notes, also cut at H beats)."""
import json
import math
import os

import numpy as np

from systems.m4_retrieval_v1 import HIST, P, System as V1, note_feats

G = dict(gamma=0.7, eps=0.02)
G.update(json.loads(os.environ.get('DUET_M4F', '{}')))


class System(V1):
    def __init__(self, train):
        super().__init__(train)
        T = np.full((9, 7, 7), 0.1)                        # cond 0 = same beat, 1+b = new beat at pos b
        for songs in train.values():
            for s in songs:
                if not s.get('beats'):
                    continue
                z = [b['z'] for b in s['beats']]
                pos = [b['pos'] for b in s['beats']]
                sop = sorted(s['sop'], key=lambda n: n[0])
                f = note_feats(sop, s['tonic'], pos)
                for i in range(1, len(f)):
                    ka, kb = f[i - 1][1], f[i][1]
                    if kb >= len(z) or ka >= len(z) or z[ka] < 0 or z[kb] < 0:
                        continue
                    T[self.cond(ka, kb, f[i][2]), z[ka], z[kb]] += 1
        self.marg = T.sum(1) / T.sum((1, 2))[:, None]
        self.T = T / T.sum(2, keepdims=True)

    @staticmethod
    def cond(ka, kb, bp):
        return 0 if ka == kb else 1 + bp % 8

    def start(self, meta):
        super().start(meta)
        self.post = None
        self.hist = []                                     # (onset, cond, p_retr)

    def filt(self, prev, c, pr):
        q = prev @ self.T[c] / self.marg[c]
        x = pr * (q ** G['gamma'])
        x = x / x.sum()
        return (1 - G['eps']) * x + G['eps'] / 7

    def step(self, i, onset, midi, prev_dur):
        k = int(math.floor(onset + 0.02))
        bp = self.pos[k] % 8 if 0 <= k < len(self.pos) else 0
        ob = int(abs(onset + 0.02 - k) < 0.06)
        pc = (midi - self.t) % 12
        pr, sk = self.chord_dist(i, onset, pc, bp, ob)
        c = self.cond(self.kprev, k, bp) if i else 0
        self.kprev = k
        self.hist.append((onset, c, pr))
        if HIST < 0:
            post = pr if self.post is None else self.filt(self.post, c, pr)
        else:
            win = [h for h in self.hist if onset - h[0] <= HIST]
            post = win[0][2]
            for _, cc, pp in win[1:]:
                post = self.filt(post, cc, pp)
        self.post = post
        self.mem.setdefault(sk, []).append((onset, post))
        self.pcs.append(pc)
        self.ons.append(onset)
        h = self.plan(midi, post)
        self.prev_h, self.prev_m = h, midi
        return h


def build(train):
    return System(train)
