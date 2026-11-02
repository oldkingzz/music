"""r2_copy_v1 (run v5rt_1101, round 2, 'balanced'): round 1's best (r1_A_prior_v1: A's melody prior, expected-value plan
and posterior hear around r2_mix_v2) plus two round-2 parts from rt_systems.r2_lib_v1 (read its docstring):
  - copy model: songs repeat; matches of her last heard notes against earlier passages of the same song predict her
    next pitch and inter-onset interval, with accuracies calibrated on TRAIN only (models/R2_copy_v1.npz, fitted by
    runs_mini/v5rt_1101/r2/train_copy_v1.py). It sharpens the plan's next-note and rhythm forecasts, the posterior
    hear (misheard or not) and the held-note posterior.
  - hedged hear: the hear-point choice also scores the note against her likely next note, because it keeps sounding
    into the first L of that note; weight w_dyn * L0 / (predicted inter-onset interval - L0).
r2_mix_v2's step() runs exactly once per heard note. Settings tuned on val (no test split in this bundle) with
runs_mini/v5rt_1101/r2/tune_R2_v1.py / tune_R2_v2.py.
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v1 as LIB

CFG = dict(
    lam=2.0, kappa=1.0, kappa_hold=0.5,            # plan: harsh weight, attack cost vs an unheard note / her held note
    mu=0.05, mu_old=-1.0,                          # hear: margin for keeping a planned note (old notes never kept)
    beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0, w_x=0.3, gamma=0.0, min_cov=0.0, silence=False,
    hold_win='beat', post_hold=True, pan_floor=0.0,
    hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False,   # posterior hear when P(misheard) >= 0.2
    w_next=0.0, w_dyn=1.0, lam_n=1.0, hx=0.0, hedge_post=0,  # hedged hear, dynamic weight
    copy=True, copy_rhy=1.0, copy_p=1.0)           # copy model for pitch and rhythm, calibrated accuracies as is


def build(train):
    return LIB.PlanR2(r2_mix_v2.build(train), LIB.Prior(), CFG, LIB.load_copy_tab())
