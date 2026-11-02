"""m4_oracle -- CEILING, NOT A METHOD, NOT CAUSAL. It is told the song's TRUE chord on the beat of each
note (looked up from the eval songs' labels, like oracle_chord) and then uses exactly the m4 voice-
leading planner (same costs: leaps, parallel 5ths/8ves, unison/octave, mid range, contrary motion),
with p(chord) = 1 on the true chord. It shows how far m4_retrieval is from "knowing the chord".
Beats with no chord label (z = -1) fall back to the retrieval distribution."""
import math

import numpy as np

from systems.m4_retrieval_v1 import System as Ret


class Oracle(Ret):
    def __init__(self, train, lookup):
        super().__init__(train)
        self.lookup = lookup

    def start(self, meta):
        super().start(meta)
        self.meta = meta
        self.cand = self.lookup.get((meta['tonic'], meta['K'], tuple(meta['pos'][:16])), [])

    def step(self, i, onset, midi, prev_dur):
        # keys can collide (same key, length, meter): keep the songs whose note onsets match so far
        c2 = [c for c in self.cand if i < len(c[0]) and abs(c[0][i] - onset) < 1e-6]
        self.cand = c2 or self.cand
        self.z = self.cand[0][1] if self.cand else []
        k = int(math.floor(onset + 0.02))
        z = self.z[k] if 0 <= k < len(self.z) else -1
        if z < 0:
            return super().step(i, onset, midi, prev_dur)
        p = np.zeros(7)
        p[z] = 1.0
        self.pcs.append((midi - self.t) % 12)
        self.ons.append(onset)
        h = self.plan(midi, p)
        self.prev_h, self.prev_m = h, midi
        return h


def build(train):
    import bench3
    lookup = {}
    for split in ('val',):  # mini bundle: test split removed, oracle is val-only
        for songs in bench3.eval_sets(split).values():
            for s in songs:
                ons = tuple(n[0] for n in sorted(s['sop'], key=lambda n: n[0]))
                lookup.setdefault((s['tonic'], s['K'], tuple(b['pos'] for b in s['beats'][:16])), []).append(
                    (ons, [b['z'] for b in s['beats']]))
    return Oracle(train, lookup)
