"""r2_lib_v2 (run v5rt_1101, round 2): LIBRARY (no build()). r2_lib_v1.PlanR2 (A's plan + posterior hear, hedged hear,
copy model; read its docstring) plus HARMONIC MEMORY (hm_rho > 0; off = r2_lib_v1's decisions exactly):

When her current note ends a passage that repeats an earlier passage of the same song (the copy model's longest match,
at least hm_min notes, the time shift a whole number of beats, and with hm_bar a whole number of bars), the chord of
the current beat k is probably the chord Duet already heard at beat k' = k - shift. m1_filter_v5lo's HMM evidence of
the finished beats (its own per-beat log-likelihoods, m1's transition table) gives the smoothed posterior of beat k'
(forward message times a backward message over at most hm_lag later finished beats). Duet's belief for beat k becomes
  b_k  ~  b_k * gamma_k'^hm_rho      (log-linear),
used wherever this library reads a chord belief: the hedged hear's chooser score (so the choice at a hear point can
differ from r2_mix_v2's even with w = 0), the posterior hear, the next-note utility and the plan's forecasts.
r2_mix_v2's own step() still runs exactly once per note and its state (GRU, filter) is unchanged by this.
Val only, tuned on val (no test split).
"""
import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID

from rt_systems import r2_lib_v1 as R1
from rt_systems.r2_lib_v1 import DV, E_ALL, GOODV, HARSHV, HV, Prior, load_copy_tab


class PlanR2H(R1.PlanR2):
    def start(self, meta):
        super().start(meta)
        self.fw = []                                       # m1 HMM forward posteriors of the finished beats
        self.shift = None                                  # beats back to the aligned earlier passage
        self.gcache = {}

    # ------------------------------------------------------------------ harmonic memory
    def _hm_on(self):
        return self.c.get('hm_rho', 0) > 0 and self.copy is not None

    def _fw_extend(self):
        f = self.f
        while len(self.fw) < f.done:
            j = len(self.fw)
            ll = f.ev[j]
            pr = f.m.pi if j == 0 else self.fw[-1] @ f.m.A[M1.tkey(f._pos(j), f.M)]
            w = pr * np.exp(ll - ll.max())
            self.fw.append(w / w.sum())

    def _gamma(self, kp):
        f = self.f
        if kp < 0 or kp >= f.done:
            return None
        key = (kp, f.done)
        g = self.gcache.get(key)
        if g is not None:
            return g
        self._fw_extend()
        beta = np.ones(M1.S)
        for j in range(min(f.done - 1, kp + int(self.c.get('hm_lag', 8))), kp, -1):
            ll = f.ev[j]
            beta = f.m.A[M1.tkey(f._pos(j), f.M)] @ (np.exp(ll - ll.max()) * beta)
            beta /= beta.sum()
        g = self.fw[kp] * beta
        g = g / g.sum()
        self.gcache[key] = g
        return g

    def _aug(self, b, k):
        if not self._hm_on() or self.shift is None:
            return b
        g = self._gamma(k - self.shift)
        if g is None:
            return b
        x = np.log(np.maximum(b, 1e-300)) + self.c['hm_rho'] * np.log(np.maximum(g, 1e-300))
        x = np.exp(x - x.max())
        return x / x.sum()

    def _belief(self, k):
        return self._aug(np.asarray(self.f.belief(k), dtype=np.float64), k)

    def _align(self):
        cp = self.copy
        self.shift = None
        r = cp.r1
        if len(r) == 0:
            return
        best = int(r.max())
        if best < self.c.get('hm_min', 4):
            return
        a = cp.n - 1
        b = int(np.nonzero(r == best)[0][-1])
        d = cp.on[a] - cp.on[b]
        s = int(round(d))
        if s < 1 or abs(d - s) > 0.02 or (self.c.get('hm_bar', 0) and s % self.M):
            return
        self.shift = s

    def _forecast(self, k, q, off_c):
        b_new, b_hold = super()._forecast(k, q, off_c)
        if self._hm_on() and self.c.get('hm_plan', 1):
            return self._aug(b_new, k), self._aug(b_hold, k)
        return b_new, b_hold

    # ------------------------------------------------------------------ hear
    def hear(self, i, onset, midi, prev_dur):
        if self.copy is not None:
            self.pred_cur = self.copy.pred
            self.copy.add(onset, midi)
            self.pred_next = self.copy.pred
            if self._hm_on():
                self._align()
        c = self.c
        if self.her and c['hear_tau'] < 1:
            post = self.p_true(onset, midi)
            if 1 - post[0] >= c['hear_tau']:
                return self._hear_post(i, onset, midi, prev_dur, post)
        return self._hear_hedge(i, onset, midi, prev_dur)

    def _u_next(self, onset, midi, k):
        c = self.c
        her = self.her
        iv = 11 if len(her) < 2 else int(np.clip(midi - her[-2][1], -5, 5)) + 5
        tn = self._ioi_pred(onset)
        pmel = self.pr.mel[(midi - self.t) % 12, iv, self._cls(tn)]
        pmel = self._mix_copy(pmel, midi, self.pred_next)
        b = self._belief(k)
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

    def _hear_hedge(self, i, onset, midi, prev_dur):
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
        if (w > 0 or (self._hm_on() and self.shift is not None)) and h_r >= 0:
            pfit = self._belief(k) @ M1.CTM
            v = self._m1vec(midi, pfit, prev, prev_m)
            if w > 0:
                v = v + w * self._u_next(onset, midi, k)
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
    return PlanR2H(r2_mix_v2.build(train), Prior(), cfg, tab)
