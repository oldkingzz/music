"""HMM + a melody model's hard guess of her next note (last heard note + bar position, backoff),
trained on Bach sopranos. The guess is fed to the HMM as if heard (planner.run hint)."""
import numpy as np

from exp_more import NGram
from model import ChordHMM, soprano_obs
from planner import run


class System:
    def __init__(self, train):
        self.model = ChordHMM().fit(train['bach'])
        self.ng = NGram(1).fit(train['bach'])
        self.note_guess = {}

    def run(self, song, cfg):
        t = song['tonic']
        pos = [b['pos'] for b in song['beats']]

        def hint(k, jstar, frac):
            if jstar < 1:
                return None
            o = []
            for j in range(jstar + 1):                 # heard beats; the last one up to the decision
                mid = soprano_obs(song['sop'], j, frac if j == jstar else 1.0)[1]
                o.append(None if mid is None else (mid - t) % 12)
            o += [None] * (k - jstar)
            pc = self.ng.predict(o, k, pos[k])
            if pc is None:
                return None
            self.note_guess[k] = pc
            vec = np.zeros(12)
            vec[(pc + t) % 12] = 1.0
            return vec, 60 + (pc + t) % 12
        return run(song, self.model, cfg, 'hmm', hint=hint)


def build(train):
    return System(train)
