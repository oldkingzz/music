"""r4_G_hedgesafe_v1 (run v5rt_1101, round 4, idea G 'late-hear policy'): r4_G_hedge_v1 (read its docstring) with a more
cautious next-note utility (harsh weight 0.3 instead of 0.2) and a gentler hedge for eighth notes (floor 0.25). Library
rt_systems.r4_G_lib_v2 (read r4_G_lib_v1 and v2). In the small lag regime (wired, micbt, micbt_cal, phone files,
bench_rt L < 0.5) every decision is W's exactly. Val only, tuned on val (no test split) with
runs_mini/v5rt_1101/r4/G/dev_G_v1.py (config 'c1_ln3_f25').
"""
from rt_systems import r4_G_lib_v2 as GL

CFG = dict(
    lam=2.0, kappa=1.0, kappa_hold=0.25,
    mu=0.05, mu_old=-1.0,
    beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0, w_x=0.3, gamma=0.0, min_cov=0.0, silence=False,
    hold_win='beat', post_hold=True, pan_floor=0.0,
    hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False,
    w_next=0.0, w_dyn=1.0, lam_n=1.0, hx=0.0, hedge_post=0,
    copy=True, copy_rhy=1.0, copy_p=1.0,
    hm_mode='lr', hm_rho=1.0, hm_min=6, hm_bar=0, hm_lag=8, hm_plan=1,
    trk=True, eps_u=0.1, mel_w=1.0, tau_t=0.15, p_min=0.1, harsh_max=0.2, kappa_t=1.0, lam_t=1.0, sil_v=-0.2,
    above_att=False, free=True, frag_copy=2.0, hold_trk=True,
    kappa_st=0.3, hop_win=0.55, theta_hop=0.05,
    guard=1, rec_win=0.4, g_mu=0.1, g_sil=0.3, g_above=1,
    doubt_key=0.7, doubt_mode='noatt',
    g_on=0.5, g_n=16, g_min=3, g_add=0.0,                            # G: lag regime from the call order
    g_floor=0.25, g_wmax=4.0, g_lam_n=0.3,                           # G: lag-aware hedge
    g_late=1, g_late_pad=0.0, g_late_mu=0.0, g_late_hmax=0.3)        # G: late override

REG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard=None)


def build(train):
    return GL.make(train, CFG, REG)
