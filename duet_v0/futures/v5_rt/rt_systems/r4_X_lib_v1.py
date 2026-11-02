"""r4_X_lib_v1 (run v5rt_1101, round 4, select step): LIBRARY (no build(): not a system itself). Every scored variant
r4_X_<name>_v<k>.py passes its own constants (rule 7). Val only, tuned on val (this bundle has no test split).

Merge of the two round-4 ideas, both built on W (r3_X_regtrkst_v1, rt_systems.r3_X_lib_v3.PlanEF3 inside idea D's
register adapter):
  idea H (rt_systems.r4_H_lib_v2, read r4_H_lib_v1 and v2): LagWrap's lag bound from the API call order (lag mode while
          her grid notes are first known one full beat after their onset), PlanH's lag-aware plan (TRAIN-only rhythm
          table models/H_lagrhy_v1.npz, kappa_L, kappa_st_L), late_keep (a hear for a note she began before the plan's
          latest attack keeps the plan's note) and the optional hedge_L hear weight;
  idea G (rt_systems.r4_G_lib_v4, read r4_G_lib_v1 .. v4): the lag regime from the median q_last - onset of the song's
          answered notes, the lag-aware hedged hear (g_floor, g_wmax, next-note harsh weight g_lam_n), the same hedge in
          E's guarded choice (g_ehedge), the late override (g_late, g_late_hmax) and the keep-guard switch (g_guard).
PlanX = class PlanX(PlanH2, PlanG4): Python's method order is PlanX, PlanH2, PlanH, PlanG4, PlanG3, PlanG2, PlanG,
PlanEF3 (W). So
  plan   in H's lag mode H's lag-aware plan answers; otherwise G's plan (which only records q_last) and then W's plan.
         PlanX.plan records q_last itself first, because H's lag plan does not call G's plan and G's lag samples need
         every plan point;
  hear   H's late_keep is outermost; inside it G's keep-guard switch, G's lag sample and late override, then W's hear
         (whose hedge weight and E's guarded choice use the methods below);
  _w_next  H's hedge_L form when H's lag mode is on and hedge_L > 0, else G's lag-aware form in G's large regime, else
         W's L0 form;
  _u_next / _hear_trk  G's (next-note harsh weight g_lam_n and the E hedge in G's large regime), else W's.
Both regime detectors read nothing but the call order (no time, latency, settings or environment). With neither regime
on (wired, micbt, micbt_cal, v11, bench_rt at L <= 0.3) every decision is W's exactly, as in both ideas.

make(train, cfg, reg, lag) as r4_H_lib_v2.make, with this class.
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_D_lib_v3 as DL
from rt_systems import r3_E_lib_v2 as EL
from rt_systems import r4_G_lib_v4 as G4
from rt_systems import r4_H_lib_v1 as H1
from rt_systems import r4_H_lib_v2 as H2
from rt_systems.r2_lib_v1 import Prior, load_copy_tab

LagWrap = H2.LagWrap


class PlanX(H2.PlanH2, G4.PlanG4):
    def plan(self, q, heard):
        self.q_last = q                                   # G's lag sample (also when H's lag plan answers q)
        return super().plan(q, heard)


def make_core(train, cfg):
    return PlanX(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab() if cfg.get('copy') else None,
                 R3.load_hm_tab() if cfg.get('hm_mode') == 'lr' else None,
                 EL.load_trk_tab() if cfg.get('trk') else None, H1.load_lag_tab())


def make(train, cfg, reg, lag):
    core = make_core(train, cfg)
    inner = DL.RegAdapter(core, reg) if reg else core
    return LagWrap(inner, core, lag)
