"""r4_G_ehedge_v1 (run v5rt_1101, round 4, idea G 'late-hear policy'): W (r3_X_regtrkst_v1, round 3's page default) with
a lag-aware hear. Library rt_systems.r4_G_lib_v4 (read r4_G_lib_v1 .. v4). Val only, tuned on val (this bundle has no
test split) with runs_mini/v5rt_1101/r4/G/dev_G_v1.py (log dev_log_G_v1.jsonl, config 'e1_f25').

- Lag regime from the call order only: per answered note lob = q_last - onset (q_last = the latest plan point before the
  hear call); LARGE when the median of the song's last 16 values (at least 3) is >= 0.5 beat: the plan for the half beat
  after her onset is usually decided before her note is answered (live bench v2's bt_cal; bench_rt at L = 0.5).
  Otherwise (wired, micbt, micbt_cal, the phone files, bench_rt L < 0.5) every decision is W's exactly.
- Large regime, lag_hat = median lob (0.5 beat), d = her predicted inter-onset interval:
  hedge   W's hedged hear weighs her NEXT note by w = lag_hat / max(d - lag_hat, 0.25) (at most 4) instead of W's
          L0 = 0.1 form, and the next-note utility weighs harsh by 0.2 instead of 1.0: the answer sounds about 0.6 beat
          after her onset, on the rest of her note and, kept by the next plan point, on the start of her next note.
  E hedge idea E's guarded choice (heard pitch doubted by the tracker posterior) adds w * U_next to every note
          candidate's score the same way (silence's next-note value 0).
  late    when her note is predicted to end before the answer sounds (d <= lag_hat), a hear attack is kept back (Duet
          keeps its sounding note) unless it is at least as good for her likely next note and harsh against that note
          with probability <= 0.3.
"""
from rt_systems import r4_G_lib_v4 as GL

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
    g_floor=0.25, g_wmax=4.0, g_lam_n=0.2,                           # G: lag-aware hedge
    g_ehedge=1.0,                                                    # G: the same hedge in E's guarded choice
    g_late=1, g_late_pad=0.0, g_late_mu=0.0, g_late_hmax=0.3)        # G: late override

REG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard=None)


def build(train):
    return GL.make(train, CFG, REG)
