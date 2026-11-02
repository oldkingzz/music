"""r4_H_lagk2hd_v1 (run v5rt_1101, round 4, idea H 'plan under information lag'). Val only, tuned on val (this bundle
has no test split). Library rt_systems.r4_H_lib_v2 (read r4_H_lib_v1 and v2).

The most cautious idea-H variant: r4_H_lagk15hd_v1 (read its docstring) with onset attacks paying kappa_L = 2.0 x
P(new) x P(harsh) (1.5 there; W 1.0) and the full lag-aware hedge (hedge_L = 0.5). W = r3_X_regtrkst_v1 (constants
copied unchanged) behind the lag bound from the API call order; in LAG MODE (her grid notes first known one full beat
after their onset: live bt_cal, bench_rt L = 0.5) the plan uses the TRAIN-only lag-aware rhythm table
models/H_lagrhy_v1.npz, no onset stumble cost, and a hear for a note she began before the plan's latest attack keeps
the plan's note (late_keep = 2). Otherwise every decision is W's exactly.
Dev runs: runs_mini/v5rt_1101/r4/H/dev{1,2,3,4}_v1.json (bt_cal cells, val).
"""
from rt_systems import r4_H_lib_v2 as HL

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
    doubt_key=0.7,
    doubt_mode='noatt',
    # idea H, read only in lag mode
    kappa_L=2.0, kappa_st_L=0.0, late_keep=2, hedge_L=0.5)

REG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard=None)

LAG = dict(n_win=5, n_min=3, n_need=3, d_on=0.9, grid_tol=0.02)


def build(train):
    return HL.make(train, CFG, REG, LAG)
