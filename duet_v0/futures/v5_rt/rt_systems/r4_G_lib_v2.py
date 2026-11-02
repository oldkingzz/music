"""r4_G_lib_v2 (run v5rt_1101, round 4, idea G 'late-hear policy'): LIBRARY (no build()). rt_systems.r4_G_lib_v1.PlanG
(read its docstring: lag regime from the call order, lag-aware hedge, late override, follow guard; g_on None = W
exactly) plus three options for the LARGE regime only (each off by default; then the decisions are r4_G_lib_v1's):

  g_hmode    'all' (v1: the lag-aware hedge weight for every hear in the large regime) or 'late' (only when her note is
             predicted to end before the answer sounds, d <= lag_hat + g_late_pad; otherwise W's L0 weight, so an
             answer that still sounds on her current note is chosen for it as W chooses).
  g_lam_n    the next-note utility's harsh weight (W: lam_n = 1.0) in the large regime: lower = the hedge seeks hits on
             her next note more and avoids harsh notes less.
  g_free     predicted-late answers (as above) choose among ALL alto notes, not only the notes consonant with and below
             the heard pitch of the note she is leaving: x = argmax U_next(x) - g_free_cur * P(x harsh against the heard
             pitch) (g_free_cur: the risk that she holds the note longer than predicted), then r4_G_lib_v1's late
             override rules (g_late_mu, g_late_hmax, g_late_sil) compare x with the sounding note. W's hear still runs
             first (its state updates exactly once); its answer is replaced.
Val only, tuned on val (no test split). make(train, cfg, reg=None) as r4_G_lib_v1.make, with this class.
"""
import numpy as np

from systems import r2_mix_v2
from theory import ALTO

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_D_lib_v3 as DL
from rt_systems import r3_E_lib_v2 as EL
from rt_systems import r4_G_lib_v1 as G1
from rt_systems.r2_lib_v1 import HARSHV, HV, Prior, load_copy_tab


class PlanG2(G1.PlanG):
    def _late(self, onset, lam):
        return self._ioi_pred(onset) - onset <= lam + self.c.get('g_late_pad', 0.0)

    def _w_next(self, onset):
        lam = self._g_lag()
        if lam is not None and self.c.get('g_hmode', 'all') == 'late' and not self._late(onset, lam):
            return G1.XL.PlanEF3._w_next(self, onset)            # W's L0 weight
        return super()._w_next(onset)

    def _u_next(self, onset, midi, k):
        c = self.c
        if 'g_lam_n' not in c or self._g_lag() is None:
            return super()._u_next(onset, midi, k)
        l0 = c.get('lam_n', 1.0)
        c['lam_n'] = c['g_lam_n']
        try:
            return super()._u_next(onset, midi, k)
        finally:
            c['lam_n'] = l0

    def hear(self, i, onset, midi, prev_dur):
        c = self.c
        if not c.get('g_free', 0):
            return super().hear(i, onset, midi, prev_dur)
        if self.q_last is not None and onset > self.g_last_on + 1e-9:
            self.g_lob.append(self.q_last - onset)
            self.g_last_on = onset
        lam = self._g_lag()
        hs0, att0 = self.h, (self.att_kind, self.att_t, self.att_on)
        q_save = self.q_last
        self.q_last = None                                   # recorded above: G1.hear must not record it again
        try:
            r = G1.XL.PlanEF3.hear(self, i, onset, midi, prev_dur)
        finally:
            self.q_last = q_save
        if lam is None:
            return r
        self.stats['g_large'] += 1
        if not self._late(onset, lam) or not (hs0 >= 0 and ALTO[0] <= hs0 <= ALTO[1]):
            if r is not None and r >= 0:
                self._g_attacked(onset)
            return r
        self.stats['g_late_pred'] += 1
        k = int(onset)
        u = self._u_next(onset, midi, k)
        ph = self._ph_next(onset, midi, k)
        cur = HARSHV[(midi - HV) % 12]
        v = u - c.get('g_free_cur', 0.0) * cur
        x = int(HV[int(np.argmax(v))])
        if x == hs0:
            return self._g_set(hs0, hs0, att0, 'g_free_keep_best')
        if ph[x - ALTO[0]] > c.get('g_late_hmax', 2.0):
            if ph[hs0 - ALTO[0]] >= c.get('g_late_sil', 2.0):
                return self._g_set(-1, hs0, att0, 'g_late_sil')
            return self._g_set(hs0, hs0, att0, 'g_late_keep_harsh')
        if float(v[x - ALTO[0]] - v[hs0 - ALTO[0]]) < c.get('g_late_mu', -1e9):
            return self._g_set(hs0, hs0, att0, 'g_late_keep')
        self.stats['g_free_attack'] += 1
        self.att_kind, self.att_t, self.att_on = 'hear', onset + G1.XL.X2.X1.F4.F3.F2.F1.L0, onset
        self.h = x
        self.f.prev = x
        self.planned = False
        self._g_attacked(onset)
        return x


def make_base(mix, prior, cfg, copy_tab, hm_tab, trk_tab):
    return PlanG2(mix, prior, cfg, copy_tab if cfg.get('copy') else None,
                  hm_tab if cfg.get('hm_mode') == 'lr' else None, trk_tab if cfg.get('trk') else None)


def make(train, cfg, reg=None):
    base = make_base(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab())
    return DL.RegAdapter(base, reg) if reg else base
