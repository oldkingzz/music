"""Idea A check of the hear side (run v5rt_1101, round 1). VAL ONLY. Can the melody prior tell when a heard note is
misheard? At every hear() of r1_A_core_v2.PlanA (setting from the command line; '{"min_cov": 1e9, "mu_old": -1}'
reproduces reactive:r2_mix_v2) it computes P(misheard) = 1 - posterior that her true pitch is the heard one (melody
prior for this note from the two heard notes before it, times bench3's 10 % mishearing channel), and bins hear points
by it: how many are really misheard, and how many hear ATTACKS are fresh clashes (harsh against her true note).

  python calib_A_v2.py '<json overrides>' [--sub N]
"""
import argparse
import bisect
import json
import os
import sys
from collections import defaultdict

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'A'))
from diag_A_v1 import B, r2_mix_v2  # noqa: E402
from tune_A_v2 import BASE  # noqa: E402
from rt_systems import r1_A_core_v2 as core  # noqa: E402

BINS = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001])


def p_mis(obj, onset, midi):
    """1 - P(true pitch = heard | the two heard notes before it)."""
    her = obj.her
    if not her:
        return 0.0
    m_b = her[-1][1]
    iv = 11 if len(her) < 2 else int(np.clip(m_b - her[-2][1], -5, 5)) + 5
    pr = obj.pr.mel[(m_b - obj.t) % 12, iv, obj._cls(onset)]
    e = np.concatenate(([0], core.MIS))
    d = np.clip(midi + e - m_b, -12, 12) + 12
    w = pr[d] * np.where(e == 0, 0.9, 0.1 / 11)
    return float(1 - w[0] / w.sum())


class Rec(core.PlanA):
    def start(self, meta):
        super().start(meta)
        self.rec = []

    def hear(self, i, onset, midi, prev_dur):
        pm = p_mis(self, onset, midi)
        old = self.h
        out = super().hear(i, onset, midi, prev_dur)
        self.rec.append((i, onset, midi, pm, old, out))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg')
    ap.add_argument('--sub', type=int, default=0)
    a = ap.parse_args()
    B.bench3.B2.heard = B.HEARD1
    train = B.bench3.train_sets()
    obj = Rec(r2_mix_v2.build(train), core.Prior(), dict(BASE, **json.loads(a.cfg)))
    L = 0.1
    for ds, songs in B.bench3.eval_sets('val').items():
        if a.sub:
            songs = songs[:a.sub]
        tab = defaultdict(lambda: [0, 0, 0, 0])            # hear points, misheard, attacks, fresh clashes
        nn = 0
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            B.simulate(obj, meta, notes, L)
            true = sorted(s['sop'], key=lambda n: n[0])
            z = [b['z'] for b in s['beats']]
            for i, onset, midi, pm, old, out in obj.rec:
                if onset < B.J or int(onset) >= len(z) or z[int(onset)] < 0:
                    continue
                nn += 1
                m_true = true[i][2]
                r = tab[int(np.digitize(pm, BINS))]
                r[0] += 1
                r[1] += m_true != midi
                if out is not None and out >= 0 and out != old:
                    r[2] += 1
                    r[3] += abs(m_true - out) % 12 in B.HARSH
        print(f'{ds}: hear points {nn}', flush=True)
        for b, r in sorted(tab.items()):
            print(f'   P(misheard) [{BINS[b - 1]:.2f},{BINS[b]:.2f}): share {r[0] / nn:.3f} really misheard {r[1] / r[0]:.3f} '
                  f'attacks {r[2]} fresh clashes {r[3]} ({100 * r[3] / nn:.2f} per 100 notes)', flush=True)


if __name__ == '__main__':
    main()
