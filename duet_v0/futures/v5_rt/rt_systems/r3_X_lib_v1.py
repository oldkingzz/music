"""r3_X_lib_v1 (run v5rt_1101, round 3, combine step): LIBRARY (no build(): not a system itself). Val only, tuned on val
(this bundle has no test split). It merges the three round-3 fixes; every scored variant r3_X_<name>_v<k>.py passes its
own constants (rule 7).

PlanEF = idea E's tracker-aware hearing (rt_systems.r3_E_lib_v2.PlanE, read its docstring: a TRAIN-only table of the
live tracker's errors judges every heard note; when the note is probably misheard or a split fragment, Duet keeps its
note, goes silent or attacks only a safe note below her) plus idea F's stumble guards (rt_systems.r3_F_lib_v4.PlanF4,
read r3_F_lib_v1 .. v4: the plan's onset stumble cost kappa_st and hop guard hop_win / theta_hop / hop_bad, and the
hedged hear's keep guard guard / rec_win / g_mu / g_sil / g_above). Python's method order is
PlanEF, PlanE, PlanF4, PlanF3, PlanF2, PlanF, PlanR2C, PlanR2H, PlanR2, PlanA (r1_A_core_v3), PlanA (r1_A_core_v2), so:
  plan()   F's guarded plan (r3_F_lib_v2.PlanF2.plan); the held-note posterior it uses is E's (_hold_post, hold_trk);
  hear()   E's: the tracker posterior, then E's guarded choice when P(misheard) >= tau_t, else the hedged hear, which
           is F's (keep guard, r3_F_lib_v2.PlanF2._hear_hedge). Every hear decision is recorded for F's guards
           (r3_F_lib_v1.PlanF._mark: who started the sounding note, when, for which of her notes).
  F's cautious posterior hear (post_*) and round 2's posterior hear (hear_tau) are never reached while trk = True:
  E's guarded choice replaces them.
With the F switches off (kappa_st 0, hop_win 0, guard 0) every decision is PlanE's exactly; with trk = False it is
PlanF4's exactly.

make(train, cfg, reg=None): PlanEF built from cfg (r2_mix_v2, A's prior, the copy, harmonic-memory and tracker tables, all
TRAIN only). With reg (constants of rt_systems.r3_D_lib_v3.RegAdapter) idea D's register adapter wraps it: from her first
heard notes an octave shift s puts her melody into the TRAIN soprano register [62, 74); the base hears her pitches + s
and Duet sings the base's note - s folded into 48..79 (s = 0: a pure pass-through, as for every val chorale at her own
pitch). The tracker table is conditioned on intervals and onset classes only, so it is the same in the shifted frame.
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_D_lib_v3 as DL
from rt_systems import r3_E_lib_v2 as EL
from rt_systems import r3_F_lib_v4 as F4
from rt_systems.r2_lib_v1 import Prior, load_copy_tab


class PlanEF(EL.PlanE, F4.PlanF4):
    def hear(self, i, onset, midi, prev_dur):
        """E's hear (tracker posterior, guarded choice or F's hedged hear), recorded for F's plan / keep guards."""
        return self._mark(EL.PlanE.hear(self, i, onset, midi, prev_dur), onset)


def make_base(mix, prior, cfg, copy_tab, hm_tab, trk_tab):
    return PlanEF(mix, prior, cfg, copy_tab if cfg.get('copy') else None, hm_tab if cfg.get('hm_mode') == 'lr' else None,
                  trk_tab if cfg.get('trk') else None)


def make(train, cfg, reg=None):
    base = make_base(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab())
    return DL.RegAdapter(base, reg) if reg else base
