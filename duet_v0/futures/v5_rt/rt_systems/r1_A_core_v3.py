"""r1_A_core_v3 (run v5rt_1101, round 1, idea A): LIBRARY for the r1_A_* systems (no build(): not a system itself).
r1_A_core_v2's PlanA (read r1_A_core_v1 and v2 for the plan rule) with two additions, found on val (tuned on val):
  1. pan_floor: the clash risk of an attack against a note she has not been heard to start uses
     max(pan, pan_floor): on val the rhythm model's smallest pan values were too small for pop / pop909 (predicted
     0.01, real 0.04).
  2. posterior hear (hear_tau < 1): the melody prior also judges what she sang. At hear() the posterior that her true
     pitch is the heard one is the prior for this note (from the two heard notes before it) times bench3's 10 %
     mishearing channel. When P(misheard) >= hear_tau (about 5 % of her notes on val, 80-95 % of them really
     misheard), Duet does not trust r2_mix_v2's reactive choice (it is consonant with a pitch she probably did not
     sing): it picks, among the alto notes, keeping the sounding note and (if hear_sil) silence, the best
     pfit(h) * P(consonant, not her unison/octave) - lam_h * P(harsh) - w_x * P(Duet at or above her) - m1's voice
     leading costs, all expectations over that posterior, an attack (a note other than the sounding one) paying
     kappa_h * P(harsh) more. pfit = r2_mix_v2's own chord belief at this note (its step() still runs exactly once).
     Below hear_tau the hear rule is v2's.
"""
import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID

from rt_systems import r1_A_core_v2 as V2
from rt_systems.r1_A_core_v2 import DV, GOODV, HARSHV, HV, MIS, Prior  # noqa: F401

E_ALL = np.concatenate(([0], MIS))                        # her true pitch - heard pitch: right, or misheard


class PlanA(V2.PlanA):
    def plan(self, q, heard):
        if self.c['pan_floor'] <= 0:
            return super().plan(q, heard)
        rhy = self.pr.rhy
        self.pr.rhy = self._floored
        try:
            return super().plan(q, heard)
        finally:
            self.pr.rhy = rhy

    def __init__(self, mix, prior, cfg):
        super().__init__(mix, prior, cfg)
        if self.c['pan_floor'] > 0:
            self._floored = prior.rhy.copy()
            self._floored[..., 0] = np.maximum(self._floored[..., 0], self.c['pan_floor'])

    def p_true(self, onset, midi):
        """Posterior over her true pitch (midi + E_ALL) given the heard pitch and the two heard notes before it."""
        her = self.her
        m_b = her[-1][1]
        iv = 11 if len(her) < 2 else int(np.clip(m_b - her[-2][1], -5, 5)) + 5
        pr = self.pr.mel[(m_b - self.t) % 12, iv, self._cls(onset)]
        d = np.clip(midi + E_ALL - m_b, -12, 12) + 12
        w = pr[d] * np.where(E_ALL == 0, 0.9, 0.1 / 11)
        return w / w.sum()

    def hear(self, i, onset, midi, prev_dur):
        c = self.c
        if not self.her or c['hear_tau'] >= 1:
            return super().hear(i, onset, midi, prev_dur)
        post = self.p_true(onset, midi)
        if 1 - post[0] < c['hear_tau']:
            return super().hear(i, onset, midi, prev_dur)
        f = self.f
        prev = f.prev
        f.step(i, onset, midi, prev_dur)                  # r2_mix_v2's step: GRU state and filter, exactly once
        self.her.append((onset, midi))
        self.planned = False
        hs = self.h
        pfit = f.belief(int(onset)) @ M1.CTM
        mt = midi + E_ALL
        ivm = (mt[:, None] - HV[None, :]) % 12
        G, Hh = GOODV[ivm], HARSHV[ivm]
        X = (mt[:, None] <= HV[None, :]).astype(np.float64)
        ph = post @ Hh
        P = M1.P
        d = np.abs(HV - prev) if prev is not None else np.zeros(len(HV))
        s = (pfit[(HV - self.t) % 12] * (post @ G) - c['lam_h'] * ph - c['w_x'] * (post @ X)
             - P['w_step'] * d - P['w_big'] * (d > 7) - P['w_mid'] * np.abs(HV - ALTO_MID))
        att = s - c['kappa_h'] * ph
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        if keep_ok:
            att[hs - ALTO[0]] = s[hs - ALTO[0]]
        j = int(np.argmax(att))
        if c['hear_sil'] and att[j] < 0:
            self.stats['hear_post_sil'] += 1
            self.h = -1
            return -1 if hs >= 0 else None
        h = int(HV[j])
        f.prev = h
        if keep_ok and h == hs:
            self.stats['hear_post_keep'] += 1
            return None
        self.stats['hear_post_attack'] += 1
        self.h = h
        return h


def make(train, cfg):
    return PlanA(r2_mix_v2.build(train), Prior(), cfg)
