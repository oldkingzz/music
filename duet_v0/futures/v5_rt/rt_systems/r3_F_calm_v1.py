"""r3_F_calm_v1 (run v5rt_1101, round 3, idea F 'stumble guard', fewest stumbles): r3_F_guard_v1 (read it) with the
hop guard as a block (no half-beat move right after a hear attack for her current note, unless the sounding note is
probably not consonant with her held note: hop_bad 0.5) instead of a 0.05 margin, and w_x 0.25. Library
rt_systems.r3_F_lib_v4; tables TRAIN only. Tuned on val (no test split) with runs_mini/v5rt_1101/r3/F/tune_F_v4.py.
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v3 as LIB3
from rt_systems import r3_F_lib_v4 as LIB
from rt_systems.r2_lib_v1 import Prior, load_copy_tab

CFG = dict(
    lam=2.0, kappa=1.0, kappa_hold=0.5,
    mu=0.05, mu_old=-1.0,
    beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0, w_x=0.25, gamma=0.0, min_cov=0.0, silence=False,
    hold_win='beat', post_hold=True, pan_floor=0.0,
    hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False,
    w_next=0.0, w_dyn=1.0, lam_n=1.0, hx=0.0, hedge_post=0,
    copy=True, copy_rhy=1.0, copy_p=1.0,
    hm_mode='lr', hm_rho=1.0, hm_min=6, hm_bar=0, hm_lag=8, hm_plan=1,
    # idea F: stumble guards
    kappa_st=0.3, hop_win=0.55, hop_bad=0.5,
    guard=1, rec_win=0.4, g_mu=0.2, g_sil=0.3, g_above=1,
    post_veto=1, post_below=1, post_nh=1, post_nh_keep=1)


def build(train):
    return LIB.PlanF4(r2_mix_v2.build(train), Prior(), CFG, load_copy_tab(), LIB3.load_hm_tab())
