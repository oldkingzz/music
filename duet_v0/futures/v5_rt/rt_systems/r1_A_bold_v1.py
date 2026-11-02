"""r1_A_bold_v1 (run v5rt_1101, round 1, idea A, 'bold'): r1_A_prior_v1 with half the attack cost, kappa = 0.5, and a
minimum gain theta = 0.01 hit-beats for any change: more notes start on time, at the price of more fresh clashes.
Library: r1_A_core_v3; prior runs_mini/v5rt_1101/models/A_prior_v2.npz (train only). Tuned on val (no test split).
"""
from systems import r2_mix_v2

from rt_systems import r1_A_core_v3 as core

CFG = dict(lam=2.0, kappa=0.5, kappa_hold=0.5, mu=0.05, mu_old=-1.0, beta=1.0, theta=0.01, theta_sil=0.0, nu=1.0,
           w_x=0.3, gamma=0.0, min_cov=0.0, silence=False, hold_win='beat', post_hold=True, pan_floor=0.0,
           hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False)


def build(train):
    return core.PlanA(r2_mix_v2.build(train), core.Prior(), CFG)
