"""r1_A_prior_v1 (run v5rt_1101, round 1, idea A, 'balanced'): melody prior + expected-fit plan + posterior hear.
Library: r1_A_core_v3 (read r1_A_core_v1 / v2 / v3 for the rules); prior runs_mini/v5rt_1101/models/A_prior_v2.npz
fitted on train only by runs_mini/v5rt_1101/r1/A/train_prior_A_v1.py and _v2.py. Wraps r2_mix_v2 (its step() runs
exactly once per heard note). Settings tuned on val (no test split in this bundle) with
runs_mini/v5rt_1101/r1/A/tune_A_v3.py.
"""
from systems import r2_mix_v2

from rt_systems import r1_A_core_v3 as core

CFG = dict(
    lam=2.0,          # plan: weight of P(harsh) against P(hit) per beat of her singing
    kappa=1.0,        # plan: extra cost of an attack per unit P(harsh vs a note she has not been heard to start)
    kappa_hold=0.5,   # plan: the same against her held (heard) note
    mu=0.05,          # hear: a planned note stays if m1's score of the reactive choice beats it by at most this
    mu_old=-1.0,      # hear: an old (not planned) note never stays against a different reactive choice (as reactive)
    beta=1.0,         # chord coupling of the melody prior: P(next pitch | chord) ~ P_mel * (Bc / P(pc)) ** beta
    theta=0.0,        # plan: a new note must beat keeping by this many hit-beats
    theta_sil=0.0,
    nu=1.0,           # plan: weight of m1's voice-leading costs
    w_x=0.3,          # cost per unit P(Duet at or above her note)
    gamma=0.0,        # plan: weight of the next note's first L just after q + 0.5
    min_cov=0.0,
    silence=False,    # plan: never goes silent on purpose
    hold_win='beat',  # plan: a note chosen for her held note counts to the end of the beat
    post_hold=True,   # plan: her held note's true pitch = melody-prior posterior
    pan_floor=0.0,
    hear_tau=0.2,     # hear: P(misheard) at or above this -> posterior choice instead of r2_mix_v2's reactive note
    lam_h=1.0,        # hear (posterior choice): weight of P(harsh)
    kappa_h=1.0,      # hear (posterior choice): extra cost of an attack per unit P(harsh)
    hear_sil=False,
)


def build(train):
    return core.PlanA(r2_mix_v2.build(train), core.Prior(), CFG)
