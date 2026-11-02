"""r3_X_regtrkstsafe_v1 (run v5rt_1101, round 3, combine step): the cautious member of the r3_X family.
r3_X_regtrkst_v1 (read it: D's register adapter + E's tracker-aware hearing + F's stumble guards) with the doubt rule
widened: whenever E's guarded choice fires (P(misheard) >= 0.15) on a heard pitch outside the song's major scale, Duet
attacks no new note (keep the sounding note if E's expected score allows it, else silence). Library
rt_systems.r3_X_lib_v3. Tables TRAIN only (models/A_prior_v2.npz, R2_copy_v1.npz, R2_hm_v1.npz, E_trk_tab_v1.npz).
Tuned on val (no test split) with runs_mini/v5rt_1101/r3/X/dev_X_v2.py.
"""
from rt_systems import r3_X_lib_v3 as XL

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
    above_att=False, free=True, frag_copy=2.0, hold_trk=True,
    kappa_st=0.3, hop_win=0.55, theta_hop=0.05,                      # F: plan stumble cost and hop guard
    guard=1, rec_win=0.4, g_mu=0.1, g_sil=0.3, g_above=1,            # F: hear keep guard
    doubt_key=0.15,      # any doubted (E's guard fires) out-of-key heard pitch: no new attack
    doubt_mode='noatt')  # ... keep the sounding note if E's score allows it, else silence

REG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard=None)


def build(train):
    return XL.make(train, CFG, REG)
