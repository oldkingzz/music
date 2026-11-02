"""r3_E_trkholdk_v1 (run v5rt_1101, round 3, idea E 'tracker-aware hearing'): r3_E_trkhold_v1 (read it) with half
the plan's attack cost against her held note (kappa_hold 0.25 instead of 0.5): with the tracker posterior of her held
note the plan already avoids notes that are harsh against a plausible pitch, so it may move under a held note a little
more freely. Library rt_systems.r3_E_lib_v2; tables models/E_trk_tab_v1.npz, R2_copy_v1.npz, R2_hm_v1.npz (all TRAIN
only). Tuned on val (no test split) with runs_mini/v5rt_1101/r3/E/tune_E_v1.py.
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_E_lib_v2 as LIB
from rt_systems.r2_lib_v1 import Prior, load_copy_tab

CFG = dict(
    lam=2.0, kappa=1.0, kappa_hold=0.25,
    mu=0.05, mu_old=-1.0,
    beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0, w_x=0.3, gamma=0.0, min_cov=0.0, silence=False,
    hold_win='beat', post_hold=True, pan_floor=0.0,
    hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False,           # round 2's posterior hear: unused (trk = True)
    w_next=0.0, w_dyn=1.0, lam_n=1.0, hx=0.0, hedge_post=0,
    copy=True, copy_rhy=1.0, copy_p=1.0,
    hm_mode='lr', hm_rho=1.0, hm_min=6, hm_bar=0, hm_lag=8, hm_plan=1,
    trk=True, eps_u=0.1, mel_w=1.0, tau_t=0.15, p_min=0.1, harsh_max=0.2, kappa_t=1.0, lam_t=1.0, sil_v=-0.2,
    above_att=False, free=True, frag_copy=2.0,
    hold_trk=True)


def build(train):
    return LIB.PlanE(r2_mix_v2.build(train), Prior(), CFG, load_copy_tab(), R3.load_hm_tab(), LIB.load_trk_tab())
