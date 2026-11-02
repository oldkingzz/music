"""r1_A_hearpost_v1 (run v5rt_1101, round 1, idea A, ablation 'posterior hear only'): no plan at all (plan() never
changes Duet's note: min_cov is out of reach), hear() as r1_A_prior_v1: r2_mix_v2's reactive choice, except when the
melody prior says the heard note is probably misheard (P >= 0.2), then the posterior choice of r1_A_core_v3.
Shows what the posterior hear alone does (a module for round 2). Prior runs_mini/v5rt_1101/models/A_prior_v2.npz
(train only). Tuned on val (no test split).
"""
from systems import r2_mix_v2

from rt_systems import r1_A_core_v3 as core

CFG = dict(lam=2.0, kappa=1.0, kappa_hold=0.5, mu=0.05, mu_old=-1.0, beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0,
           w_x=0.3, gamma=0.0, min_cov=1e9, silence=False, hold_win='beat', post_hold=True, pan_floor=0.0,
           hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False)


def build(train):
    return core.PlanA(r2_mix_v2.build(train), core.Prior(), CFG)
