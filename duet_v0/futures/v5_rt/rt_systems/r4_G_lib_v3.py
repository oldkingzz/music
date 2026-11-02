"""r4_G_lib_v3 (run v5rt_1101, round 4, idea G 'late-hear policy'): LIBRARY (no build()). rt_systems.r4_G_lib_v2.PlanG2
(read r4_G_lib_v1 and v2) plus two options for the LARGE lag regime only (each off by default; then the decisions are
r4_G_lib_v2's exactly):

  g_guard   F's hear keep guard (W: guard = 1) in the large regime. At bt_cal the plan's half-beat note after her onset
            was chosen before her note was heard and counts as 'recent' at her hear (W measures recency with the
            assumed L0 = 0.1 beat), so the guard keeps that guess whenever it is consonant with the heard pitch and its
            chord-tone belief is close to the new choice's; 0 turns the guard off there.
  g_tau     E's tau_t (W: 0.15; P(misheard) at or above it sends the hear to E's guarded keep / silence / safe-attack
            choice) in the large regime.
Val only, tuned on val (no test split). make(train, cfg, reg=None) as r4_G_lib_v1.make, with this class.
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_D_lib_v3 as DL
from rt_systems import r3_E_lib_v2 as EL
from rt_systems import r4_G_lib_v2 as G2
from rt_systems.r2_lib_v1 import Prior, load_copy_tab


class PlanG3(G2.PlanG2):
    def hear(self, i, onset, midi, prev_dur):
        c = self.c
        sw = {k: c[k2] for k, k2 in (('guard', 'g_guard'), ('tau_t', 'g_tau')) if k2 in c}
        if not sw:
            return super().hear(i, onset, midi, prev_dur)
        rec = self.q_last is not None and onset > self.g_last_on + 1e-9
        n0 = len(self.g_lob)
        if rec:                                              # the lag sample this hear will record (G1.hear)
            self.g_lob.append(self.q_last - onset)
        large = self._g_lag() is not None
        if rec:
            del self.g_lob[n0:]
        if not large:
            return super().hear(i, onset, midi, prev_dur)
        old = {k: c[k] for k in sw}
        c.update(sw)
        try:
            return super().hear(i, onset, midi, prev_dur)
        finally:
            c.update(old)


def make_base(mix, prior, cfg, copy_tab, hm_tab, trk_tab):
    return PlanG3(mix, prior, cfg, copy_tab if cfg.get('copy') else None,
                  hm_tab if cfg.get('hm_mode') == 'lr' else None, trk_tab if cfg.get('trk') else None)


def make(train, cfg, reg=None):
    base = make_base(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab())
    return DL.RegAdapter(base, reg) if reg else base
