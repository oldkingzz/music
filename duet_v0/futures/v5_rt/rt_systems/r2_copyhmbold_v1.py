"""r2_copyhmbold_v1 (run v5rt_1101, round 2, 'bold'): r2_copyhm_v1 with a lower attack cost against a note she has not
been heard to start (kappa = 0.75): more notes start on time, the least harsh time of the round-2 variants, at the
price of more fresh clashes. Library rt_systems.r2_lib_v3 (copy model models/R2_copy_v1.npz and harmonic-memory table
models/R2_hm_v1.npz, both TRAIN only). Tuned on val (no test split) with runs_mini/v5rt_1101/r2/tune_R2_v3.py.
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v3 as LIB
from rt_systems.r2_lib_v1 import Prior, load_copy_tab

CFG = dict(
    lam=2.0, kappa=0.75, kappa_hold=0.5,
    mu=0.05, mu_old=-1.0,
    beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0, w_x=0.3, gamma=0.0, min_cov=0.0, silence=False,
    hold_win='beat', post_hold=True, pan_floor=0.0,
    hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False,
    w_next=0.0, w_dyn=1.0, lam_n=1.0, hx=0.0, hedge_post=0,
    copy=True, copy_rhy=1.0, copy_p=1.0,
    hm_mode='lr', hm_rho=1.0, hm_min=6, hm_bar=0, hm_lag=8, hm_plan=1)


def build(train):
    return LIB.PlanR2C(r2_mix_v2.build(train), Prior(), CFG, load_copy_tab(), LIB.load_hm_tab())
