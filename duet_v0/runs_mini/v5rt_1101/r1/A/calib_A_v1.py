"""Idea A calibration check (run v5rt_1101, round 1). VAL ONLY. Are the prior's probabilities right on val?
At every plan point of r1_A_core_v2.PlanA (setting from the command line) it records the rhythm model's pan / pah and
the melody model's distribution of her next true pitch, then compares with what she really did (her TRUE notes):
  - pan (an attack at q lands on a new note) vs reality, binned;
  - for plan points where a new note really starts within 0.01 beat: predicted P(harsh vs her next note) for every
    alto candidate vs whether it really is harsh, binned (this is the clash risk the plan prices);
  - the same for the held-note posterior (pah case): predicted P(harsh vs her held note) vs reality;
  - mean log-likelihood of her true next pitch, and the share of her next notes inside the model's top 1 / top 3.

  python calib_A_v1.py '<json overrides>' [--sub N]
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

BINS = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0001])


class Rec(core.PlanA):
    def start(self, meta):
        super().start(meta)
        self.rec = []

    def plan(self, q, heard):
        if self.her and heard and heard[-1][0] == self.her[-1][0]:
            on_c, off_c, m_c = heard[-1]
            k = int(q)
            s_b = min(48, int(round((q - on_c) * 12)))
            rest_b = 0 if off_c is None else 1 + min(3, int((q - off_c) / 0.5))
            phase = 0 if q == k else 1
            ioi_b = 49 if len(heard) < 2 else min(48, int(round((on_c - heard[-2][0]) * 12)))
            pan, pah = self.pr.rhy[s_b, rest_b, phase, self._tkey(k), ioi_b][:2]
            pmel = self._mel(heard, len(heard) - 1, q)
            b_new, _ = self._forecast(k, q, off_c)
            pz = pmel[None, :] * self.g[:, (m_c + core.DV - self.t) % 12]
            pz /= pz.sum(1, keepdims=True)
            pm = b_new @ pz
            self.rec.append((q, float(pan), float(pah), m_c, pm, self._hold_post(heard)))
        return super().plan(q, heard)


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
        pan_b = defaultdict(lambda: [0.0, 0.0, 0])
        hn_b = defaultdict(lambda: [0.0, 0.0, 0])
        hh_b = defaultdict(lambda: [0.0, 0.0, 0])
        ll, top1, top3, nn = 0.0, 0, 0, 0
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            B.simulate(obj, meta, notes, L)
            true = sorted(s['sop'], key=lambda n: n[0])
            tons = [n[0] for n in true]
            for q, pan, pah, m_c, pm, pn in obj.rec:
                if q < B.J:
                    continue
                u = bisect.bisect_left(tons, q - 1e-9)
                new = u < len(true) and true[u][0] < q + 0.01
                cur = [n for n in true if n[0] <= q < n[1]]
                newnote = true[u] if new else (cur[-1] if cur and cur[-1][0] > q - L else None)
                r = pan_b[int(np.digitize(pan, BINS))]
                r[0] += pan
                r[1] += newnote is not None
                r[2] += 1
                hv = core.HV
                if newnote is not None:
                    m = newnote[2]
                    d = m - m_c
                    if -12 <= d <= 12:
                        ll += np.log(max(pm[d + 12], 1e-12))
                        order = np.argsort(-pm)
                        top1 += order[0] == d + 12
                        top3 += (d + 12) in order[:3]
                    nn += 1
                    ivm = (m_c + core.DV[:, None] - hv[None, :]) % 12
                    ph = pm @ core.HARSHV[ivm]
                    real = core.HARSHV[(m - hv) % 12]
                    for p, x in zip(ph, real):
                        rr = hn_b[int(np.digitize(p, BINS))]
                        rr[0] += p
                        rr[1] += x
                        rr[2] += 1
                elif cur and cur[-1][0] <= q - L:
                    m = cur[-1][2]
                    ivm = (m_c + core.DV[:, None] - hv[None, :]) % 12
                    ph = pn @ core.HARSHV[ivm]
                    real = core.HARSHV[(m - hv) % 12]
                    for p, x in zip(ph, real):
                        rr = hh_b[int(np.digitize(p, BINS))]
                        rr[0] += p
                        rr[1] += x
                        rr[2] += 1
        print(f'{ds}: next-pitch loglik {ll / max(nn, 1):.3f} top1 {top1 / max(nn, 1):.3f} top3 {top3 / max(nn, 1):.3f} '
              f'(n {nn})', flush=True)
        for name, tab in (('pan', pan_b), ('harsh vs new note', hn_b), ('harsh vs held note', hh_b)):
            print(f'   {name:18s} ' + '  '.join(f'[{BINS[b - 1]:.2f},{BINS[b]:.2f}) pred {r[0] / r[2]:.3f} real '
                                                f'{r[1] / r[2]:.3f} n {r[2]}' for b, r in sorted(tab.items())), flush=True)


if __name__ == '__main__':
    main()
