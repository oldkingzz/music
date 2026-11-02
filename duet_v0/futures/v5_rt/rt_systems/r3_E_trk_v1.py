"""r3_E_trk_v1 (run v5rt_1101, round 3, idea E 'tracker-aware hearing'): r2_copyhm_v1 with round 2's posterior hear
(melody prior x bench3's synthetic 10 % uniform mishearing, hear_tau 0.2) replaced by a tracker-aware one; library
rt_systems.r3_E_lib_v2 (read its docstring). At every hear the posterior of her true pitch is a TRAIN-only table of the
live tracker's errors (models/E_trk_tab_v1.npz: heard interval from the previous answered note, onset class x time
since the previous answered onset, return-to-pitch flag; 135 train Bach chorales rendered sung and an octave lower and
run through the live tracker) mixed with 10 % of a flat channel and times the melody / copy prior. When P(misheard)
>= 0.15 Duet keeps its note, goes silent or attacks a note that is below every plausible pitch of hers and harsh with
probability <= 0.2; otherwise round 2's hedged hear. The plan is r2_copyhm_v1's. Copy model models/R2_copy_v1.npz and
harmonic-memory table models/R2_hm_v1.npz (TRAIN only) as round 2. Tuned on val (no test split) with
runs_mini/v5rt_1101/r3/E/tune_E_v1.py (log tune_log_E_v1.jsonl).
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_E_lib_v2 as LIB
from rt_systems.r2_lib_v1 import Prior, load_copy_tab

CFG = dict(
    lam=2.0, kappa=1.0, kappa_hold=0.5,
    mu=0.05, mu_old=-1.0,
    beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0, w_x=0.3, gamma=0.0, min_cov=0.0, silence=False,
    hold_win='beat', post_hold=True, pan_floor=0.0,
    hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False,           # round 2's posterior hear: unused (trk = True)
    w_next=0.0, w_dyn=1.0, lam_n=1.0, hx=0.0, hedge_post=0,
    copy=True, copy_rhy=1.0, copy_p=1.0,
    hm_mode='lr', hm_rho=1.0, hm_min=6, hm_bar=0, hm_lag=8, hm_plan=1,
    trk=True,          # tracker-aware hear
    eps_u=0.1,         # share of a flat channel mixed into the tracker table
    mel_w=1.0,         # exponent of the melody / copy prior (times 1 - P(fragment))
    tau_t=0.15,        # P(misheard) at or above this -> guarded choice
    p_min=0.1,         # posterior mass that makes a true pitch 'plausible' (attacks must lie below all of them)
    harsh_max=0.2,     # an attack in the guarded choice must be harsh with probability <= this
    kappa_t=1.0,       # extra cost of an attack per unit P(harsh)
    lam_t=1.0,         # weight of P(harsh) in the guarded choice
    sil_v=-0.2,        # value of silence in the guarded choice
    above_att=False,   # attacks never at or above a plausible pitch of hers
    free=True,         # any alto note may be attacked (not only the reactive / hedged choice)
    frag_copy=2.0,     # every answered note goes to the copy model
    hold_trk=False)    # the plan's held-note posterior stays round 2's


def build(train):
    return LIB.PlanE(r2_mix_v2.build(train), Prior(), CFG, load_copy_tab(), R3.load_hm_tab(), LIB.load_trk_tab())
