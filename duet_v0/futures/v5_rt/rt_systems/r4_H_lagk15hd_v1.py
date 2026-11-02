"""r4_H_lagk15hd_v1 (run v5rt_1101, round 4, idea H 'plan under information lag'). Val only, tuned on val (this bundle
has no test split). Library rt_systems.r4_H_lib_v2 (read r4_H_lib_v1 and v2).

W = r3_X_regtrkst_v1 (its constants are copied below unchanged: register adapter, tracker-aware hearing, stumble guards,
doubt rule) behind the lag bound from the API call order. While the call order says that a note of hers on the
half-beat grid is first known one full beat after its onset (D_i = 1.0 for at least 3 of her last 5 grid notes: live
bt_cal, i.e. AirPods both ways after calibration, and bench_rt at L = 0.5), the system is in LAG MODE; otherwise (wired,
micbt_cal, v11 at its native latency, bench_rt at L <= 0.3: D_i = 0.5) every decision is W's exactly.
In lag mode:
- the plan uses a TRAIN-only lag-aware rhythm table (models/H_lagrhy_v1.npz: her last KNOWN note is the one with
  onset + 0.7 <= q, and the plan's note covers her new note until the hear answer at onset + 0.55), so at her likely
  onsets it values its guess for her new note about five times more than W (whose table assumed the answer comes
  0.1 beat after her onset), and at points where she may already be on a note it has not heard (att_unh) an attack
  pays for its clash risk against that note;
- attacks at onset points pay kappa_L = 1.5 x P(new note) x P(harsh) (W: 1.0) and no onset stumble cost (kappa_st_L 0:
  the hear of her new note comes ~0.6 beat later, never within the 250 ms stumble window);
- late_keep = 2: a hear for a note she began BEFORE the plan's latest attack keeps the plan's note (the hear still
  updates every model); this removes the stumbles where a late answer replaced the plan's note 50-330 ms after it began;
- hedge_L = 0.5: the hedged hear weighs her next note by 0.5 x 0.55 / max(d - 0.55, 0.15) (d = predicted inter-onset
  interval) in place of W's 0.1 / max(d - 0.1, 0.15): an answer that arrives 0.6 beat late mostly sounds over her next
  note, so it is chosen to fit that note too (fewer fresh clashes from late answers).
Dev runs: runs_mini/v5rt_1101/r4/H/dev{1,2,3,4}_v1.json (dev_H_v1.py / dev_H_v2.py, bt_cal cells, val).
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
    kappa_L=1.5, kappa_st_L=0.0, late_keep=2, hedge_L=0.5)

REG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard=None)

LAG = dict(n_win=5, n_min=3, n_need=3, d_on=0.9, grid_tol=0.02)


def build(train):
    return HL.make(train, CFG, REG, LAG)
