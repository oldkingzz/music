"""r4_X_hdgsafe_v1 (run v5rt_1101, round 4, select step: merge of the two idea picks r4_H_lagk15hd_v1 and
r4_G_ehedgesafe_v1). Val only, tuned on val (this bundle has no test split). Library rt_systems.r4_X_lib_v1 (read its
docstring, and r4_H_lib_v1 / v2, r4_G_lib_v1 .. v4).

r4_X_hgsafe_v1 (read its docstring: W + idea H's lag-aware plan with kappa_L 1.5, kappa_st_L 0, late_keep 2 + idea G's
late-hear policy with next-note harsh weight 0.3, E hedge, late override, keep guard off in G's large regime) plus H's
hedge weight hedge_L = 0.5: while H's lag mode is on, the hedged hear and E's guarded choice weigh her next note by
0.5 x 0.55 / max(d - 0.55, 0.15) (r4_H_lagk15hd_v1's form) in place of G's 0.5 / max(d - 0.5, 0.25); G's next-note
utility (harsh weight 0.3) and the rest of G's policy are unchanged. Outside both regimes every decision is W's exactly.
"""
from rt_systems import r4_X_lib_v1 as XL4

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
    # idea H, read only in H's lag mode
    kappa_L=1.5, kappa_st_L=0.0, late_keep=2, hedge_L=0.5,
    # idea G, read only in G's large regime
    g_on=0.5, g_n=16, g_min=3, g_add=0.0,
    g_floor=0.25, g_wmax=4.0, g_lam_n=0.3,
    g_ehedge=1.0,
    g_late=1, g_late_pad=0.0, g_late_mu=0.0, g_late_hmax=0.3,
    g_guard=0)

REG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard=None)

LAG = dict(n_win=5, n_min=3, n_need=3, d_on=0.9, grid_tol=0.02)


def build(train):
    return XL4.make(train, CFG, REG, LAG)
