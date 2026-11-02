"""m4_retrieval_v5 = v4 + a second retrieval query: "which pitch classes has she sung so far in this bar"
(a 12-bit set, plus her current note and bar position). Chords usually last a bar or half a bar, so
the notes of the current bar are direct evidence about it. The two retrieval guesses are combined
(MIX = 'lin': average, 'poe': product divided by the chord marginal). DUET_HIST also cuts the bar set
(only notes that started within the last H beats)."""
import json
import math
import os

import numpy as np

from systems.m4_retrieval_v4 import HIST, P, System as V4, note_feats

Q = dict(mix='lin', w=0.5, beta2=10.0)
Q.update(json.loads(os.environ.get('DUET_M4B', '{}')))


def bar_mask(pcs, ons, i, bar_start, now, hist):
    m = 0
    j = i - 1
    while j >= 0 and ons[j] >= bar_start - 1e-6:
        if hist < 0 or now - ons[j] <= hist:
            m |= 1 << pcs[j]
        j -= 1
    return m


class System(V4):
    def __init__(self, train):
        super().__init__(train)
        tab = {}
        for songs in train.values():
            for s in songs:
                if not s.get('beats'):
                    continue
                z = [b['z'] for b in s['beats']]
                pos = [b['pos'] for b in s['beats']]
                sop = sorted(s['sop'], key=lambda n: n[0])
                f = note_feats(sop, s['tonic'], pos)
                pcs = [x[0] for x in f]
                ons = [n[0] for n in sop]
                for i, (on, off, m) in enumerate(sop):
                    w = np.zeros(7)
                    for k in range(int(on), int(math.ceil(off))):
                        if k < len(z) and z[k] >= 0:
                            w[z[k]] += min(off, k + 1) - max(on, k)
                    if w.sum() <= 0:
                        continue
                    w /= w.sum()
                    pc, k, bp, ob = f[i]
                    bs = k - (pos[k] if k < len(pos) else 0)
                    for key in ((pc, bp, ob), (pc, bp, ob, bar_mask(pcs, ons, i, bs, on, -1))):
                        a = tab.get(key)
                        if a is None:
                            tab[key] = w.copy()
                        else:
                            a += w
        self.tab2 = tab
        tot = sum(v for k, v in tab.items() if len(k) == 3)
        self.marg = tot / tot.sum()

    def chord_dist(self, i, onset, pc, bp, ob):
        p1, sk = super().chord_dist(i, onset, pc, bp, ob)
        k = int(math.floor(onset + 0.02))
        bs = k - (self.pos[k] if 0 <= k < len(self.pos) else 0)
        p = self.prior
        for key in ((pc, bp, ob), (pc, bp, ob, bar_mask(self.pcs, self.ons, i, bs, onset, HIST))):
            c = self.tab2.get(key)
            if c is None:
                break
            tot = c.sum()
            typ = (c > 0.05 * tot).sum()
            lam = tot / (tot + Q['beta2'] * typ)
            p = lam * c / tot + (1 - lam) * p
        if Q['mix'] == 'lin':
            out = (1 - Q['w']) * p1 + Q['w'] * p
        else:
            out = p1 * p / self.marg
            out = out / out.sum()
        return out, sk


def build(train):
    return System(train)
