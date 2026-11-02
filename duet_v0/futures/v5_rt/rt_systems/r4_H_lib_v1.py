"""r4_H_lib_v1 (run v5rt_1101, round 4, idea H 'plan under information lag'): LIBRARY (no build(): not a system
itself). Val only, tuned on val (this bundle has no test split). Every scored variant r4_H_<name>_v<k>.py passes its own
constants (rule 7).

Base = W (rt_systems.r3_X_lib_v3.PlanEF3 inside idea D's register adapter, r3_X_regtrkst_v1's constants). Two parts:

1. LagWrap: the lag bound from the API call order alone. At hear() for her note i with onset o_i on the half-beat plan
   grid, let q_last be the most recent plan point; the first plan point that can know note i is q_last + 0.5, so
   D_i = q_last + 0.5 - o_i. Wired, micbt_cal, v11 at its native latency and bench_rt at L <= 0.3: D_i = 0.5 (a note
   is known at the next plan point). bt_cal (AirPods both ways, ~0.65 beat between her onset and the hear call) and
   bench_rt at L = 0.5 (plan before hear on ties): D_i = 1.0, so a note that began half a beat before a plan point is
   not known there. Lag mode is on while at least n_need of her last n_win grid notes had D_i >= d_on (and at least
   n_min were seen). Off: every decision is W's exactly. The wrapper reads nothing but the call order (no time, no
   latency, no settings; rule 7).
2. PlanH.plan in lag mode: W's expected-value plan (r3_F_lib_v2.PlanF2.plan with E's held-note posterior) with a
   LAG-AWARE rhythm table (models/H_lagrhy_v1.npz, TRAIN only, runs_mini/v5rt_1101/r4/H/fit_lagrhy_H_v1.py): the plan
   conditions on her last KNOWN note (onset + LI <= q, LI = 0.7 beat) and counts how long its note covers her NEW note
   before the hear answer replaces it (onset + LS, LS = 0.55 beat; W's table assumed 0.1 for both). A 7th outcome
   att_unh = P(she is on a note that began before q and is not known yet): an attack then pays kappa_unh_L * att_unh *
   P(harsh against that note) (her unknown note's pitch is the next-note prior: melody prior x copy model x chord
   forecast, as W's new-note term). The copy model's predicted next onset is mixed in with the same lags. Knobs (all
   in cfg, read only in lag mode): lam_L (clash weight in the new-note value), kappa_L / kappa_hold_L / kappa_unh_L
   (attack clash costs), kappa_st_L (onset stumble cost), theta_L (attack margin), sil_L (go silent when no attack is
   chosen and P(the kept note clashes with her new or unknown note) >= sil_L), hedge_L (the hedged hear's weight on
   her next note uses LS in place of L0: w = hedge_L * LS / max(d - LS, 0.15), d = her predicted inter-onset interval).
   In bench_rt a note is listed in heard before its hear() runs when they tie; lag mode uses only the notes already
   passed to hear() (as live, where heard and hear() are in step); W's plan returns None at such points.
"""
import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_D_lib_v3 as DL
from rt_systems import r3_E_lib_v2 as EL
from rt_systems import r3_F_lib_v1 as F1
from rt_systems import r3_X_lib_v3 as XL
from rt_systems.r2_lib_v1 import DV, EPS, GOODV, HARSHV, HV, Prior, load_copy_tab

ABS = '/Users/san/claude_sandbox/music/duet_v0'
LAG_TAB = ABS + '/runs_mini/v5rt_1101/models/H_lagrhy_v1.npz'
STEP = 0.5
BLOCK = F1.BLOCK


def load_lag_tab(path=LAG_TAB):
    z = np.load(path)
    return {'rhy': z['rhy'].astype(np.float64), 'LI': float(z['LI']), 'LS': float(z['LS'])}


class PlanH(XL.PlanEF3):
    def __init__(self, mix, prior, cfg, copy_tab=None, hm_tab=None, trk=None, lag_tab=None):
        super().__init__(mix, prior, cfg, copy_tab, hm_tab, trk)
        self.lag_tab = lag_tab
        self.lag_on = False                               # set by LagWrap before every call

    # ------------------------------------------------------------------ hedge (hear) under lag
    def _w_next(self, onset):
        c = self.c
        if self.lag_on and c.get('hedge_L', 0) > 0:
            d = self._ioi_pred(onset) - onset
            ls = self.lag_tab['LS']
            return c['hedge_L'] * ls / max(d - ls, 0.15)
        return super()._w_next(onset)

    # ------------------------------------------------------------------ plan
    def plan(self, q, heard):
        if not self.lag_on:
            return super().plan(q, heard)
        self.stats['lag_plan'] += 1
        return self._plan_lag(q, heard)

    def _plan_lag(self, q, heard):
        her = self.her
        if not her or not heard:
            return None
        if len(heard) > len(her):
            heard = heard[:len(her)]                      # only notes already passed to hear() (as live)
        on_c, off_c, m_c = heard[-1]
        if on_c != her[-1][0]:
            return None
        c = self.c
        T = self.lag_tab
        LI, LS = T['LI'], T['LS']
        k = int(q)
        s_b = min(48, int(round((q - on_c) * 12)))
        known_off = off_c is not None and off_c + LI <= q + 1e-9
        rest_b = 1 + min(3, int((q - off_c) / 0.5)) if known_off else 0
        phase = 0 if q == k else 1
        ioi_b = 49 if len(heard) < 2 else min(48, int(round((on_c - heard[-2][0]) * 12)))
        pan, pah, cn, ch, cx, cb, punh = T['rhy'][s_b, rest_b, phase, self._tkey(k), ioi_b]
        if c['hold_win'] == 'beat':
            ch = cb
        pred = self.pred_next
        if pred is not None and c.get('copy_rhy', 0) > 0:
            tn = on_c + pred['ioi']
            if tn + LI > q + 1e-9:                        # not falsified: she would be known by now
                a = pred['i_acc'] * c['copy_rhy']
                new_at_q = q - 1e-9 <= tn < q + EPS
                unh = tn < q - 1e-9
                sing = off_c is None
                pan_c = 1.0 if new_at_q else 0.0
                punh_c = 1.0 if unh else 0.0
                pah_c = 1.0 if (sing and not new_at_q and not unh) else 0.0
                cn_c = max(0.0, min(tn + LS, q + 0.5) - max(tn, q)) if tn < q + 0.5 else 0.0
                cb_c = max(0.0, min(k + 1, tn) - q) if (sing and not unh) else 0.0
                pan, pah, punh = a * pan_c + (1 - a) * pan, a * pah_c + (1 - a) * pah, a * punh_c + (1 - a) * punh
                cn, ch = a * cn_c + (1 - a) * cn, a * cb_c + (1 - a) * ch
        if cn + ch < c['min_cov']:
            return None
        pmel = self._mel(heard, len(heard) - 1, q)
        pmel = self._mix_copy(pmel, m_c, pred)
        b_new, b_hold = self._forecast(k, q, off_c)
        pz = pmel[None, :] * self.g[:, (m_c + DV - self.t) % 12]
        pz /= pz.sum(1, keepdims=True)
        ivm = (m_c + DV[:, None] - HV[None, :]) % 12
        G, Hh = GOODV[ivm], HARSHV[ivm]
        X = (m_c + DV[:, None] <= HV[None, :]).astype(np.float64)
        pm = b_new @ pz
        ph_new = pm @ Hh
        px_new = pm @ X
        phit = (b_new[:, None] * self.cth * (pz @ G)).sum(0)
        u_new = phit - c.get('lam_L', c['lam']) * ph_new - c['w_x'] * px_new
        pn = self._hold_post(heard)
        ph_hold = pn @ Hh
        pg_hold = pn @ G
        u_hold = (b_hold @ self.cth) * pg_hold - c['lam'] * ph_hold - c['w_x'] * (pn @ X)
        w = cn + ch
        V = cn * u_new + ch * u_hold - w * c['nu'] * self.wmid
        hs = self.h
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        v_keep = V[hs - ALTO[0]] if keep_ok else 0.0
        kap = c.get('kappa_L', c['kappa'])
        att = (kap * pan * ph_new + c.get('kappa_hold_L', c['kappa_hold']) * pah * ph_hold
               + c.get('kappa_unh_L', kap) * punh * ph_new)
        kst = c.get('kappa_st_L', c.get('kappa_st', 0))
        if kst > 0:
            att = att + kst * pan * (1.0 - phit)
        Vc = V - att
        if keep_ok:
            dd = np.abs(HV - hs)
            Vc = Vc - w * c['nu'] * (M1.P['w_step'] * dd + M1.P['w_big'] * (dd > 7))
            Vc[hs - ALTO[0]] = -np.inf
        j = int(np.argmax(Vc))
        base = max(v_keep, 0.0 if c['silence'] else -np.inf) + c.get('theta_L', c['theta'])
        thr = base
        if (c.get('hop_win', 0) > 0 and self.att_kind == 'hear' and self.att_on == on_c
                and q - on_c < c['hop_win']):
            allow = c.get('hop_bad', 0) > 0 and keep_ok and pg_hold[hs - ALTO[0]] < c['hop_bad']
            if not allow:
                th = c.get('theta_hop', BLOCK)
                if th >= BLOCK / 2:
                    thr = np.inf
                else:
                    thr = base + th
        if np.isfinite(Vc[j]) and Vc[j] > thr:
            h = int(HV[j])
            self.stats['lag_attack'] += 1
            self.h = h
            self.planned = True
            self.f.prev = h
            self.att_kind, self.att_t, self.att_on = 'plan', q, None
            return h
        if keep_ok and c.get('sil_L', 2.0) <= 1.0 and (pan + punh) * ph_new[hs - ALTO[0]] >= c['sil_L']:
            self.stats['lag_sil'] += 1
            self.h = -1
            self.planned = False
            self.att_kind = self.att_t = self.att_on = None
            return -1
        return None


class LagWrap:
    """The whole system (register adapter + PlanH) behind the lag bound from the call order (module docstring)."""

    def __init__(self, inner, core, cfg):
        self.inner, self.core, self.c = inner, core, dict(cfg)
        self.songs = []

    def start(self, meta):
        self.q_last = None
        self.dl = []
        self.info = {'lag_on_at': None, 'n_d': 0}
        self.songs.append(self.info)
        self.core.lag_on = False
        self.inner.start(meta)
        self.core.lag_on = False

    def _lag(self):
        c = self.c
        r = self.dl[-c['n_win']:]
        on = len(r) >= c['n_min'] and sum(d >= c['d_on'] for d in r) >= c['n_need']
        if on and self.info['lag_on_at'] is None:
            self.info['lag_on_at'] = len(self.dl)
        return on

    def plan(self, q, heard):
        self.q_last = q
        self.core.lag_on = self._lag()
        return self.inner.plan(q, heard)

    def hear(self, i, onset, midi, prev_dur):
        if self.q_last is not None and abs(onset * 2 - round(onset * 2)) < self.c['grid_tol']:
            self.dl.append(self.q_last + STEP - onset)
            self.info['n_d'] += 1
        self.core.lag_on = self._lag()
        return self.inner.hear(i, onset, midi, prev_dur)


def make_core(train, cfg):
    return PlanH(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab() if cfg.get('copy') else None,
                 R3.load_hm_tab() if cfg.get('hm_mode') == 'lr' else None,
                 EL.load_trk_tab() if cfg.get('trk') else None, load_lag_tab())


def make(train, cfg, reg, lag):
    core = make_core(train, cfg)
    inner = DL.RegAdapter(core, reg) if reg else core
    return LagWrap(inner, core, lag)
