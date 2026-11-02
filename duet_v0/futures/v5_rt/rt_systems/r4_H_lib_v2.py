"""r4_H_lib_v2 (run v5rt_1101, round 4, idea H 'plan under information lag'): LIBRARY (no build(): not a system
itself). Val only, tuned on val (this bundle has no test split). rt_systems.r4_H_lib_v1 (read its docstring: LagWrap's
lag bound from the call order, PlanH's lag-aware plan) plus one hear switch, read only in lag mode and off by default
(then every decision is r4_H_lib_v1's exactly):

  late_keep  a hear arrives for her note i (onset o_i) after the plan has already ATTACKED a note at a plan point
             q_a > o_i (in lag mode the plan points after her onset do not know note i yet, and the plan attacks when
             it expects her NEXT note there). Live with AirPods both ways (bt_cal) this hear comes ~0.6 beat after o_i,
             often 50-330 ms after that plan attack, and replacing the plan's note then is a stumble (wrong-then-
             corrected start) for the note the plan aimed at. late_keep = 2: Duet keeps the plan's note (the hear still
             runs in full: tracker posterior, copy model, chord filter and m1's state advance exactly as before; only
             the sounding note and the system's belief about it stay the plan's). late_keep = 1: keep it unless it is
             harsh (2nd, tritone, 7th) against the heard pitch.
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_D_lib_v3 as DL
from rt_systems import r3_E_lib_v2 as EL
from rt_systems import r4_H_lib_v1 as H1
from rt_systems.r2_lib_v1 import Prior, load_copy_tab

HARSH = frozenset({1, 2, 6, 10, 11})
LagWrap = H1.LagWrap


class PlanH2(H1.PlanH):
    def hear(self, i, onset, midi, prev_dur):
        lk = self.c.get('late_keep', 0)
        hs = self.h
        if not (self.lag_on and lk and hs >= 0 and self.att_kind == 'plan' and self.att_t is not None
                and self.att_t > onset + 1e-6):
            return super().hear(i, onset, midi, prev_dur)
        kept = (self.att_kind, self.att_t, self.att_on)
        r = super().hear(i, onset, midi, prev_dur)
        if r is None or r == hs:
            return r
        if lk < 2 and abs(midi - hs) % 12 in HARSH:
            self.stats['late_switch'] += 1
            return r
        self.h = hs                                       # Duet keeps the plan's note; the base's belief follows
        self.f.prev = hs
        self.planned = False
        self.att_kind, self.att_t, self.att_on = kept
        self.stats['late_keep'] += 1
        return None


def make_core(train, cfg):
    return PlanH2(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab() if cfg.get('copy') else None,
                  R3.load_hm_tab() if cfg.get('hm_mode') == 'lr' else None,
                  EL.load_trk_tab() if cfg.get('trk') else None, H1.load_lag_tab())


def make(train, cfg, reg, lag):
    core = make_core(train, cfg)
    inner = DL.RegAdapter(core, reg) if reg else core
    return LagWrap(inner, core, lag)
