"""r2_lib_v1 (run v5rt_1101, round 2): LIBRARY for the r2_* systems (no build(): not a system itself). Every scored
variant r2_<name>_v<k>.py passes its own constants (rule 7). Val only, tuned on val (this bundle has no test split).

Base: round 1's best, r1_A_core_v3.PlanA (A's melody prior + expected-value plan + posterior hear around r2_mix_v2;
r2_mix_v2's step() still runs exactly once per heard note). Two additions, each off when its switch is off (then the
decisions are r1_A_core_v3's exactly):

1. HEDGED HEAR (w_next > 0 or w_dyn > 0). Live, the note Duet picks at a hear point keeps sounding into the first L of
   her NEXT note (the bleed: about 13 % of her singing time, reactive hits only 30 % of it and is harsh 26 % of it).
   bench3's chooser only scores her current note. Here the chooser score of every allowed note (m1_filter_v5lo's own
   score, recomputed exactly: w = 0 gives r2_mix_v2's choice) gets  + w * U_next(h), U_next = P(hit) - lam_n P(harsh)
   - hx P(Duet at or above her) against her likely next note (A's melody prior, sharpened by the copy model below;
   chord = r2_mix_v2's belief at this beat, pushed by m1's transition table to the beat of her predicted next onset).
   w = w_next, or (w_dyn > 0) w_dyn * L0 / max(d - L0, 0.15) with d her predicted inter-onset interval (the share of
   the note's time that is bleed). No extra attack: the note is still started at the hear point and must be consonant
   with what was heard, so a fresh clash still needs a mishearing.
2. COPY MODEL (copy = True). Songs repeat. After every heard note, the earlier heard notes of the same song are searched
   for passages whose last n notes match the last n heard notes (same heard pitch and inter-onset interval, at most one
   pitch mismatch since 10 % of heard pitches are wrong, n >= 3). Their continuations vote for her next pitch and
   inter-onset interval. How often the top vote is right was counted on TRAIN only (runs_mini/v5rt_1101/r2/
   train_copy_v1.py -> models/R2_copy_v1.npz; by match length, vote share, number of matches and A's prior for that
   pitch). The calibrated P(top is right) replaces A's prior on that pitch (the rest of A's prior is scaled to the
   remaining mass) in: the plan's next-note prediction, the hedge, the posterior hear (is the heard note misheard?) and
   the held-note posterior. The interval vote, weighted by its own calibrated accuracy times copy_rhy, is mixed into the
   plan's rhythm numbers (P(a new note at q), P(she holds), the coverage windows) in place of A's rhythm table alone.
L0 = 0.1 beat is the latency the systems assume (as round 1's priors); bench_rt's L is never read.
"""
import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID, CONSONANT

from rt_systems import r1_A_core_v3 as A3
from rt_systems.r1_A_core_v2 import DV, GOODV, HARSHV, HV, Prior

ABS = '/Users/san/claude_sandbox/music/duet_v0'
COPY_TAB = ABS + '/runs_mini/v5rt_1101/models/R2_copy_v1.npz'
L0 = 0.1                                                  # assumed hearing latency (beats)
EPS = 0.01                                                # a note 'starts at q' if it starts in [q, q + EPS)
TOL_IOI = 0.01                                            # inter-onset intervals match within this (beats)
CONSV = np.array([float(i in CONSONANT) for i in range(12)])
E_ALL = A3.E_ALL                                          # true - heard: 0 (right) or a mishearing offset
CH_ALL = np.where(E_ALL == 0, 0.9, 0.1 / 11)              # bench3's channel P(heard | true) on those offsets


def len_b(n):
    return 0 if n < 5 else 1 if n < 8 else 2 if n < 12 else 3


def share_b(s):
    return 0 if s < 0.5 else 1 if s < 0.75 else 2 if s < 0.95 else 3


def nsup_b(n):
    return 0 if n <= 1 else 1 if n <= 3 else 2


def ngram_b(p):
    return 0 if p < 0.1 else 1 if p < 0.25 else 2 if p < 0.5 else 3


class Copy:
    """Within-song repeat model over HEARD notes (onset, heard midi), fed once per note in time order."""

    def __init__(self, tab=None, nmin=3, cap=8192):
        self.tab, self.nmin, self.cap = tab, nmin, cap

    def start(self):
        self.n = 0
        self.hm = np.zeros(self.cap, dtype=np.int64)
        self.on = np.zeros(self.cap)
        self.io = np.zeros(self.cap)
        self.r0 = np.zeros(0, dtype=np.int64)
        self.r1 = np.zeros(0, dtype=np.int64)
        self.pred = None

    def add(self, onset, midi):
        a = self.n
        if a >= len(self.hm):
            self.hm = np.concatenate((self.hm, np.zeros(self.cap, dtype=np.int64)))
            self.on = np.concatenate((self.on, np.zeros(self.cap)))
            self.io = np.concatenate((self.io, np.zeros(self.cap)))
        ioi = onset - self.on[a - 1] if a else -1.0
        if a:
            ep = self.hm[:a] == midi
            er = np.abs(self.io[:a] - ioi) < TOL_IOI
            p0 = np.concatenate(([0], self.r0[:a - 1]))
            p1 = np.concatenate(([0], self.r1[:a - 1]))
            self.r0 = np.where(ep & er, p0 + 1, 0)
            self.r1 = np.where(er, np.where(ep, p1 + 1, p0 + 1), 0)
        self.hm[a], self.on[a], self.io[a] = midi, onset, ioi
        self.n = a + 1
        self.pred = self._predict()

    def _predict(self):
        a = self.n - 1
        r = self.r1
        if a < 1 or len(r) == 0:
            return None
        best = int(r.max())
        if best < self.nmin:
            return None
        idx = np.nonzero(r >= max(self.nmin, best - 2))[0]          # earlier passage ends b; continuation b + 1 <= a
        w = 2.0 ** (r[idx] - best) * (1.0 + 1e-6 * idx / max(a, 1))  # ties: the most recent passage
        cp = self.hm[idx + 1]
        up, inv = np.unique(cp, return_inverse=True)
        vp = np.bincount(inv, weights=w)
        jp = int(np.argmax(vp))
        ci = self.io[idx + 1]
        ki = np.round(ci * 96).astype(np.int64)
        uk, invk = np.unique(ki, return_inverse=True)
        vi = np.bincount(invk, weights=w)
        ji = int(np.argmax(vi))
        sel = invk == ji
        ioi = float(ci[sel][np.argmax(w[sel])])
        tw = float(w.sum())
        out = dict(top=int(up[jp]), p_share=float(vp[jp] / tw), ioi=ioi, i_share=float(vi[ji] / tw), best=best,
                   nsup=len(idx), lb=len_b(best), nb=nsup_b(len(idx)))
        out['pb'], out['ib'] = share_b(out['p_share']), share_b(out['i_share'])
        if self.tab is not None:
            out['p_row'] = self.tab['acc_p'][out['lb'], out['pb'], out['nb']]       # over A-prior buckets
            out['i_acc'] = float(self.tab['acc_i'][out['lb'], out['ib'], out['nb']])
        return out


def load_copy_tab(path=COPY_TAB):
    z = np.load(path)
    return {'acc_p': z['acc_p'].astype(np.float64), 'acc_i': z['acc_i'].astype(np.float64)}


class PlanR2(A3.PlanA):
    def __init__(self, mix, prior, cfg, copy_tab=None):
        super().__init__(mix, prior, cfg)
        self.copy = Copy(copy_tab) if self.c.get('copy') else None
        self.ctrans = mix.m.A

    def start(self, meta):
        super().start(meta)
        self.pred_cur = self.pred_next = None
        if self.copy is not None:
            self.copy.start()

    # ------------------------------------------------------------------ copy-sharpened priors
    def _mix_copy(self, pmel, m_ref, pred):
        """pmel over DV (true pitch - m_ref): the copy model's calibrated P(top is right) replaces A's prior there."""
        if pred is None:
            return pmel
        d = pred['top'] - m_ref
        if abs(d) > 12:
            return pmel
        acc = float(pred['p_row'][ngram_b(float(pmel[d + 12]))]) * self.c.get('copy_p', 1.0)
        out = pmel.copy()
        out[d + 12] = 0.0
        s = out.sum()
        if s <= 0:
            return pmel
        out *= (1.0 - acc) / s
        out[d + 12] += acc
        return out

    def p_true(self, onset, midi):
        her = self.her
        m_b = her[-1][1]
        iv = 11 if len(her) < 2 else int(np.clip(m_b - her[-2][1], -5, 5)) + 5
        pr = self.pr.mel[(m_b - self.t) % 12, iv, self._cls(onset)]
        pr = self._mix_copy(pr, m_b, self.pred_cur)
        d = np.clip(midi + E_ALL - m_b, -12, 12) + 12
        w = pr[d] * CH_ALL
        return w / w.sum()

    def _hold_post(self, heard):
        if not self.c['post_hold'] or len(heard) < 2:
            return super()._hold_post(heard)
        m_c, m_b = heard[-1][2], heard[-2][2]
        pr = self._mel(heard, len(heard) - 2, heard[-1][0])
        pr = self._mix_copy(pr, m_b, self.pred_cur)
        d = np.clip(m_c + E_ALL - m_b, -12, 12) + 12
        w = pr[d] * CH_ALL
        out = np.zeros(25)
        out[12 + E_ALL] = w / w.sum()
        return out

    # ------------------------------------------------------------------ plan (r1_A_core_v2.plan + copy rhythm/pitch)
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
            if tn > q - L0:                                     # not already falsified (she would have been heard)
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
        u_new = (b_new[:, None] * self.cth * (pz @ G)).sum(0) - c['lam'] * ph_new - c['w_x'] * (pm @ X)
        pn = self._hold_post(heard)
        ph_hold = pn @ Hh
        u_hold = (b_hold @ self.cth) * (pn @ G) - c['lam'] * ph_hold - c['w_x'] * (pn @ X)
        w = cn + ch
        V = cn * u_new + ch * u_hold - w * c['nu'] * self.wmid
        hs = self.h
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        v_keep = V[hs - ALTO[0]] if keep_ok else 0.0
        att = c['kappa'] * pan * ph_new + c['kappa_hold'] * pah * ph_hold
        Vc = V - att
        if keep_ok:
            dd = np.abs(HV - hs)
            Vc = Vc - w * c['nu'] * (M1.P['w_step'] * dd + M1.P['w_big'] * (dd > 7))
            Vc[hs - ALTO[0]] = -np.inf
        j = int(np.argmax(Vc))
        if Vc[j] > max(v_keep, 0.0 if c['silence'] else -np.inf) + c['theta']:
            h = int(HV[j])
            self.stats['plan_attack'] += 1
            self.h = h
            self.planned = True
            self.f.prev = h
            return h
        if keep_ok and c['silence'] and 0.0 > v_keep + c['theta_sil']:
            self.stats['plan_silence'] += 1
            self.h = -1
            self.planned = False
            return -1
        return None

    # ------------------------------------------------------------------ hear
    def hear(self, i, onset, midi, prev_dur):
        if self.copy is not None:
            self.pred_cur = self.copy.pred                     # prediction for this note (made one note ago)
            self.copy.add(onset, midi)
            self.pred_next = self.copy.pred                    # prediction for her next note
        c = self.c
        if self.her and c['hear_tau'] < 1:
            post = self.p_true(onset, midi)
            if 1 - post[0] >= c['hear_tau']:
                return self._hear_post(i, onset, midi, prev_dur, post)
        return self._hear_hedge(i, onset, midi, prev_dur)

    def _w_next(self, onset):
        c = self.c
        if c.get('w_dyn', 0) > 0:
            d = self._ioi_pred(onset) - onset
            return c['w_dyn'] * L0 / max(d - L0, 0.15)
        return c.get('w_next', 0.0)

    def _ioi_pred(self, onset):
        """Predicted onset of her next note (after the current note, already in self.her)."""
        pred = self.pred_next
        if pred is not None and pred.get('i_acc', 0) >= 0.5:
            return onset + pred['ioi']
        if len(self.her) >= 2:
            return onset + max(self.her[-1][0] - self.her[-2][0], 0.25)
        return onset + 1.0

    def _u_next(self, onset, midi, k):
        """U(h) against her next note: P(hit) - lam_n P(harsh) - hx P(Duet at or above her), over HV."""
        c = self.c
        her = self.her
        iv = 11 if len(her) < 2 else int(np.clip(midi - her[-2][1], -5, 5)) + 5
        tn = self._ioi_pred(onset)
        pmel = self.pr.mel[(midi - self.t) % 12, iv, self._cls(tn)]
        pmel = self._mix_copy(pmel, midi, self.pred_next)
        b = np.asarray(self.f.belief(k), dtype=np.float64)
        for j in range(k + 1, min(int(tn + 1e-9), k + 4) + 1):
            b = b @ self.ctrans[self._tkey(j)]
        pz = pmel[None, :] * self.g[:, (midi + DV - self.t) % 12]
        pz /= pz.sum(1, keepdims=True)
        ivm = (midi + DV[:, None] - HV[None, :]) % 12
        G, Hh = GOODV[ivm], HARSHV[ivm]
        pm = b @ pz
        u = (b[:, None] * self.cth * (pz @ G)).sum(0) - c.get('lam_n', 1.0) * (pm @ Hh)
        if c.get('hx', 0):
            u = u - c['hx'] * (pm @ (midi + DV[:, None] <= HV[None, :]).astype(np.float64))
        return u

    def _m1vec(self, midi, pfit, prev, prev_m):
        """m1_filter_v5lo's chooser score for every alto note (-inf where m1 would not allow it), same arithmetic."""
        P = M1.P
        s = np.full(len(HV), -np.inf)
        hi = min(ALTO[1], midi - 1)
        if hi < ALTO[0]:
            return s
        hv = HV[:hi - ALTO[0] + 1]
        iv = np.abs(midi - hv) % 12
        sc = pfit[(hv - self.t) % 12].copy()
        sc = sc - np.where(iv == 0, P['w_oct'], 0.0)
        d = np.abs(hv - prev) if prev is not None else np.zeros(len(hv), dtype=np.int64)
        sc = sc - (P['w_step'] * d + P['w_big'] * (d > 7) + P['w_mid'] * np.abs(hv - ALTO_MID))
        if prev is not None and prev_m is not None:
            dm, dh = midi - prev_m, hv - prev
            i0 = abs(prev_m - prev) % 12
            par = (dm != 0) & (dh != 0) & ((dh > 0) == (dm > 0)) & ((iv == 0) | (iv == 7)) & (iv == i0)
            sc = sc - np.where(par, P['w_par'], 0.0)
        s[:len(hv)] = np.where(CONSV[iv] > 0, sc, -np.inf)
        return s

    def _hear_hedge(self, i, onset, midi, prev_dur):
        """r1_A_core_v2.hear with the hedged choice h_c in place of r2_mix_v2's h_r (identical when w = 0)."""
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
        h_c = h_r
        if w > 0 and h_r >= 0:
            pfit = f.belief(k) @ M1.CTM
            v = self._m1vec(midi, pfit, prev, prev_m) + w * self._u_next(onset, midi, k)
            j = int(np.argmax(v))
            if np.isfinite(v[j]):
                h_c = int(HV[j])
                if h_c != h_r:
                    self.stats['hedge_moved'] += 1
                f.prev = h_c
        if not first and hs >= 0 and hs == h_c:
            self.stats['hear_same'] += 1
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
                pfit = f.belief(k) @ M1.CTM
                sr = self._m1score(h_r, midi, pfit, prev, prev_m)
                ss = self._m1score(hs, midi, pfit, prev, prev_m)
                ok = ss is not None and sr is not None and sr - ss <= mu
            if ok:
                f.prev = hs
                self.stats['hear_kept_plan' if planned else 'hear_kept_old'] += 1
                return None
        self.stats['hear_switch'] += 1
        self.h = h_c
        return h_c

    def _hear_post(self, i, onset, midi, prev_dur, post):
        """r1_A_core_v3's posterior choice (heard note probably misheard), plus the hedge term if hedge_post."""
        c, f = self.c, self.f
        prev = f.prev
        f.step(i, onset, midi, prev_dur)                  # r2_mix_v2's step: GRU state and filter, exactly once
        self.her.append((onset, midi))
        self.planned = False
        hs = self.h
        k = int(onset)
        pfit = f.belief(k) @ M1.CTM
        mt = midi + E_ALL
        ivm = (mt[:, None] - HV[None, :]) % 12
        G, Hh = GOODV[ivm], HARSHV[ivm]
        X = (mt[:, None] <= HV[None, :]).astype(np.float64)
        ph = post @ Hh
        P = M1.P
        d = np.abs(HV - prev) if prev is not None else np.zeros(len(HV))
        s = (pfit[(HV - self.t) % 12] * (post @ G) - c['lam_h'] * ph - c['w_x'] * (post @ X)
             - P['w_step'] * d - P['w_big'] * (d > 7) - P['w_mid'] * np.abs(HV - ALTO_MID))
        if c.get('hedge_post', 0):
            w = self._w_next(onset)
            if w > 0:
                s = s + w * self._u_next(onset, midi, k)
        att = s - c['kappa_h'] * ph
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        if keep_ok:
            att[hs - ALTO[0]] = s[hs - ALTO[0]]
        j = int(np.argmax(att))
        if c['hear_sil'] and att[j] < 0:
            self.stats['hear_post_sil'] += 1
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
    return PlanR2(r2_mix_v2.build(train), Prior(), cfg, tab)
