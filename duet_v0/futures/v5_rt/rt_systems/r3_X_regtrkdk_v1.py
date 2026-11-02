"""r3_X_regtrkdk_v1 (run v5rt_1101, round 3, combine step): idea D's register adapter around idea E's tracker-aware
hearing (r3_E_trkholdk_v1's constants) with a key-aware doubt rule; no stumble guards. Library rt_systems.r3_X_lib_v3
(read r3_X_lib_v1 .. v3).

- Register (D, rt_systems.r3_D_lib_v3.RegAdapter, constants of r3_D_reghm_v1): from her first heard notes an octave shift
  s puts the median of her first 4 heard notes into [62, 74), the TRAIN soprano register; the base hears her + s and
  Duet sings the base's note - s folded into 48..79. s = 0 (pass-through) for every val chorale at her own pitch.
- Hearing (E, rt_systems.r3_E_lib_v2.PlanE): a TRAIN-only table of the live tracker's errors judges each heard note;
  when P(misheard) >= 0.15 Duet keeps its note, goes silent or attacks only a note below every plausible pitch of hers
  that is harsh with probability <= 0.2; the plan's held-note posterior is the same table (hold_trk), attack cost on her
  held note 0.25.
- Doubt rule (r3_X_lib_v3): when such a doubted heard pitch is also outside the song's major scale and P(misheard) >=
  0.7, Duet attacks no new note and keeps whatever sounds (doubt_mode 'keep'). Without it the combination failed the
  real-voice gate G3 by one fresh clash (v11: 5 against reactive's 4).
Tables: models/A_prior_v2.npz, R2_copy_v1.npz, R2_hm_v1.npz, E_trk_tab_v1.npz (all TRAIN only). Tuned on val (no test
split) with runs_mini/v5rt_1101/r3/X/dev_X_v2.py (log dev_log_X_v2.jsonl).
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
    doubt_key=0.7,       # out-of-key heard pitch with P(misheard) >= 0.7: no new attack
    doubt_mode='keep')   # ... keep whatever sounds

REG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard=None)


def build(train):
    return XL.make(train, CFG, REG)
