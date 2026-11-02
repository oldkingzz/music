"""r3_F_lib_v2 (run v5rt_1101, round 3, idea F 'stumble guard'): LIBRARY (no build(): not a system itself). Val only,
tuned on val (no test split). rt_systems.r3_F_lib_v1.PlanF (read its docstring: onset stumble cost kappa_st, onset gate
pan_gate / phit_min, hop guard hop_win / theta_hop, hear keep guard guard / rec_win / g_mu / g_sil / g_above, posterior
stumble cost st_post and post_nh) plus, each off by default (then the decisions are r3_F_lib_v1's exactly):
  gate_xmax  at an onset point (pan >= pan_gate) a note may be attacked only if P(Duet at or above her new note) <=
             gate_xmax (low voices: the plan's guesses above her were the harsh ones).
  sil_risk   at a plan point where no note is attacked, Duet goes silent if pan * P(her new note is harsh against the
             sounding note) >= sil_risk (silence, then the hear enters: no ringing clash and no stumble).
  hop_bad    the hop guard lets a half-beat move through when the sounding note is probably not consonant with her
             held note (P(good interval) < hop_bad, e.g. an octave double chosen at hear).
  w_dbl      at hear the hedged chooser pays w_dbl for her unison / octave (r2_mix_v2's own choice is unchanged when
             the hedge is off).
  post_below the posterior hear may only attack notes below the heard pitch (as r2_mix_v2's chooser); w_x_post its own
             'Duet at or above her' weight.
"""
import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID

from rt_systems import r2_lib_v3 as LIB
from rt_systems import r3_F_lib_v1 as F1
from rt_systems.r2_lib_v1 import DV, E_ALL, EPS, GOODV, HARSHV, HV, L0, Prior, load_copy_tab

BLOCK = F1.BLOCK


class PlanF2(F1.PlanF):
    # ------------------------------------------------------------------ plan (r3_F_lib_v1.PlanF.plan + v2 options)
    def plan(self, q, heard):
        if not self.her or not heard:
            return None
        on_c, off_c, m_c = heard[-1]
        if on_c != self.her[-1][0]:
            return None
        c = self.c
        k = int(q)
        s_b = min(48, int(round((q - on_c) * 12)))
        rest_b = 0 if off_c is None else 1 + min(3, int((q - off_c) / 0.5))
        phase = 0 if q == k else 1
        ioi_b = 49 if len(heard) < 2 else min(48, int(round((on_c - heard[-2][0]) * 12)))
        pan, pah, cn, ch, cx, cb = self.pr.rhy[s_b, rest_b, phase, self._tkey(k), ioi_b]
        if c['hold_win'] == 'beat':
            ch = cb
        pred = self.pred_next
        if pred is not None and c.get('copy_rhy', 0) > 0:
            tn = on_c + pred['ioi']
            if tn > q - L0:
                a = pred['i_acc'] * c['copy_rhy']
                new_at_q = tn < q + EPS
                sing = off_c is None
                pan_c = 1.0 if new_at_q else 0.0
                pah_c = 1.0 if (sing and not new_at_q) else 0.0
                cn_c = max(0.0, min(tn + L0, q + 0.5) - max(tn, q)) if tn < q + 0.5 else 0.0
                cb_c = max(0.0, min(k + 1, tn) - q) if sing else 0.0
                pan, pah = a * pan_c + (1 - a) * pan, a * pah_c + (1 - a) * pah
                cn, ch = a * cn_c + (1 - a) * cn, a * cb_c + (1 - a) * ch
        cn = cn + c['gamma'] * cx
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
        u_new = phit - c['lam'] * ph_new - c['w_x'] * px_new
        pn = self._hold_post(heard)
        ph_hold = pn @ Hh
        pg_hold = pn @ G
        u_hold = (b_hold @ self.cth) * pg_hold - c['lam'] * ph_hold - c['w_x'] * (pn @ X)
        w = cn + ch
        V = cn * u_new + ch * u_hold - w * c['nu'] * self.wmid
        hs = self.h
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        v_keep = V[hs - ALTO[0]] if keep_ok else 0.0
        att = c['kappa'] * pan * ph_new + c['kappa_hold'] * pah * ph_hold
        if c.get('kappa_st', 0) > 0:
            att = att + c['kappa_st'] * pan * (1.0 - phit)
        Vc = V - att
        if c.get('pan_gate', 2.0) <= 1.0 and pan >= c['pan_gate']:
            ok = phit >= c.get('phit_min', 0.0)
            if c.get('gate_xmax', 2.0) <= 1.0:
                ok = ok & (px_new <= c['gate_xmax'])
            Vc = np.where(ok, Vc, -np.inf)
        if keep_ok:
            dd = np.abs(HV - hs)
            Vc = Vc - w * c['nu'] * (M1.P['w_step'] * dd + M1.P['w_big'] * (dd > 7))
            Vc[hs - ALTO[0]] = -np.inf
        j = int(np.argmax(Vc))
        base = max(v_keep, 0.0 if c['silence'] else -np.inf) + c['theta']
        thr = base
        if (c.get('hop_win', 0) > 0 and self.att_kind == 'hear' and self.att_on == on_c
                and q - on_c < c['hop_win']):
            allow = c.get('hop_bad', 0) > 0 and keep_ok and pg_hold[hs - ALTO[0]] < c['hop_bad']
            if not allow:
                th = c.get('theta_hop', BLOCK)
                if th >= BLOCK / 2:
                    if np.isfinite(Vc[j]) and Vc[j] > base:
                        self.stats['hop_blocked'] += 1
                    thr = np.inf
                else:
                    thr = base + th
        if np.isfinite(Vc[j]) and Vc[j] > thr:
            h = int(HV[j])
            self.stats['plan_attack'] += 1
            self.h = h
            self.planned = True
            self.f.prev = h
            self.att_kind, self.att_t, self.att_on = 'plan', q, None
            return h
        if keep_ok and c['silence'] and 0.0 > v_keep + c['theta_sil']:
            self.stats['plan_silence'] += 1
            self.h = -1
            self.planned = False
            self.att_kind = self.att_t = self.att_on = None
            return -1
        if keep_ok and c.get('sil_risk', 2.0) <= 1.0 and pan * ph_new[hs - ALTO[0]] >= c['sil_risk']:
            self.stats['plan_sil_risk'] += 1
            self.h = -1
            self.planned = False
            self.att_kind = self.att_t = self.att_on = None
            return -1
        return None

    # ------------------------------------------------------------------ hear
    def _hear_hedge(self, i, onset, midi, prev_dur):
        """r3_F_lib_v1.PlanF._hear_hedge plus w_dbl."""
        f, c = self.f, self.c
        prev, prev_m = f.prev, f.prev_m
        h_r = int(f.step(i, onset, midi, prev_dur))
        first = not self.her
        self.her.append((onset, midi))
        hs, planned = self.h, self.planned
        self.planned = False
        k = int(onset)
        w = self._w_next(onset)
        v = None
        pfit = None
        h_c = h_r
        if (w > 0 or (self._hm_on() and self.shift is not None)) and h_r >= 0:
            pfit = self._belief(k) @ M1.CTM
            v = self._m1vec(midi, pfit, prev, prev_m)
            if w > 0:
                v = v + w * self._u_next(onset, midi, k)
            if c.get('w_dbl', 0) > 0:
                v = v - c['w_dbl'] * (np.abs(midi - HV) % 12 == 0)
            j = int(np.argmax(v))
            if np.isfinite(v[j]):
                h_c = int(HV[j])
                if h_c != h_r:
                    self.stats['hedge_moved'] += 1
                f.prev = h_c
        if not first and hs >= 0 and hs == h_c:
            self.stats['hear_same'] += 1
            return None
        if c.get('guard', 0) and not first and hs >= 0 and self._recent(onset):
            if pfit is None:
                pfit = self._belief(k) @ M1.CTM
            if self._keep_ok(hs, h_c, midi, pfit):
                f.prev = hs
                self.stats['guard_keep_sil' if h_c < 0 else 'guard_keep'] += 1
                return None
        if first or hs < 0 or h_c < 0:
            self.stats['hear_enter' if hs < 0 else 'hear_switch'] += 1
            self.h = h_c
            return h_c
        mu = c['mu'] if planned else c['mu_old']
        if mu >= 0:
            if v is not None:
                sr = v[h_c - ALTO[0]]
                ss = v[hs - ALTO[0]] if ALTO[0] <= hs <= ALTO[1] else -np.inf
                ok = np.isfinite(ss) and sr - ss <= mu
            else:
                pf = f.belief(k) @ M1.CTM
                sr = self._m1score(h_r, midi, pf, prev, prev_m)
                ss = self._m1score(hs, midi, pf, prev, prev_m)
                ok = ss is not None and sr is not None and sr - ss <= mu
            if ok:
                f.prev = hs
                self.stats['hear_kept_plan' if planned else 'hear_kept_old'] += 1
                return None
        self.stats['hear_switch'] += 1
        self.h = h_c
        return h_c

    def _hear_post(self, i, onset, midi, prev_dur, post):
        """r3_F_lib_v1.PlanF._hear_post plus post_below and w_x_post."""
        c, f = self.c, self.f
        prev = f.prev
        f.step(i, onset, midi, prev_dur)
        self.her.append((onset, midi))
        self.planned = False
        hs = self.h
        k = int(onset)
        pfit = self._belief(k) @ M1.CTM
        mt = midi + E_ALL
        ivm = (mt[:, None] - HV[None, :]) % 12
        G, Hh = GOODV[ivm], HARSHV[ivm]
        X = (mt[:, None] <= HV[None, :]).astype(np.float64)
        ph = post @ Hh
        P = M1.P
        d = np.abs(HV - prev) if prev is not None else np.zeros(len(HV))
        s = (pfit[(HV - self.t) % 12] * (post @ G) - c['lam_h'] * ph - c.get('w_x_post', c['w_x']) * (post @ X)
             - P['w_step'] * d - P['w_big'] * (d > 7) - P['w_mid'] * np.abs(HV - ALTO_MID))
        if c.get('hedge_post', 0):
            w = self._w_next(onset)
            if w > 0:
                s = s + w * self._u_next(onset, midi, k)
        att = s - c['kappa_h'] * ph
        if c.get('post_nh', 0):
            att = np.where(HARSHV[(midi - HV) % 12] > 0, -np.inf, att)
        if c.get('post_below', 0):
            att = np.where(HV >= midi, -np.inf, att)
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        if keep_ok and c.get('st_post', 0) > 0 and self._recent(onset):
            att = att - c['st_post']
        if keep_ok:
            att[hs - ALTO[0]] = s[hs - ALTO[0]]
        j = int(np.argmax(att))
        if c['hear_sil'] and att[j] < 0:
            self.stats['hear_post_sil'] += 1
            self.h = -1
            return -1 if hs >= 0 else None
        if not np.isfinite(att[j]):
            self.stats['hear_post_none'] += 1
            self.h = -1
            return -1 if hs >= 0 else None
        h = int(HV[j])
        f.prev = h
        if keep_ok and h == hs:
            self.stats['hear_post_keep'] += 1
            return None
        self.stats['hear_post_attack'] += 1
        self.h = h
        return h


def make(train, cfg):
    tab = load_copy_tab() if cfg.get('copy') else None
    hm = LIB.load_hm_tab() if cfg.get('hm_mode') == 'lr' else None
    return PlanF2(r2_mix_v2.build(train), Prior(), cfg, tab, hm)
