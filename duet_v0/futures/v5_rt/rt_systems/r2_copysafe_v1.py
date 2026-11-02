"""r2_copysafe_v1 (run v5rt_1101, round 2, 'safe'): r2_copy_v1 with a 1.5x attack cost against a note she has not been
heard to start (kappa = 1.5): fewer early guesses, the fewest fresh clashes of the round-2 variants. Library
rt_systems.r2_lib_v1 (copy model calibrated on TRAIN only: models/R2_copy_v1.npz; hedged hear). Settings tuned on val
(no test split) with runs_mini/v5rt_1101/r2/tune_R2_v3.py.
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v1 as LIB

CFG = dict(
    lam=2.0, kappa=1.5, kappa_hold=0.5,
    mu=0.05, mu_old=-1.0,
    beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0, w_x=0.3, gamma=0.0, min_cov=0.0, silence=False,
    hold_win='beat', post_hold=True, pan_floor=0.0,
    hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False,
    w_next=0.0, w_dyn=1.0, lam_n=1.0, hx=0.0, hedge_post=0,
    copy=True, copy_rhy=1.0, copy_p=1.0)


def build(train):
    return LIB.PlanR2(r2_mix_v2.build(train), LIB.Prior(), CFG, LIB.load_copy_tab())
