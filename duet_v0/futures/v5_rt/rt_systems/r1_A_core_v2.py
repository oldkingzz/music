"""r1_A_core_v2 (run v5rt_1101, round 1, idea A): LIBRARY for the r1_A_* systems (no build(): not a system itself).
Same decision rule as r1_A_core_v1 (read its docstring), with four changes found on val (tuned on val):
  1. prior v2 (runs_mini/v5rt_1101/models/A_prior_v2.npz, train only): with hold_win = 'beat' the held-note window is
     [q, min(end of beat int(q), her next hear point)) instead of [q, q + 0.5): a note chosen for her held note stays
     to the end of the beat, and the plan at the half beat has nothing new to add.
  2. post_hold: her held note's TRUE pitch is the melody prior's posterior given what was heard (the prior for her
     note from the two notes before it, times bench3's 10 % mishearing channel) instead of a flat 0.9 / 0.1.
  3. kappa_hold: the clash penalty for an attack on her held note has its own weight (v1 used kappa for both).
  4. mu_old: the hear margin for a sounding note the plan did NOT start for this note (an old note); mu stays the
     margin for a planned note. v1 used mu for both.
"""
from collections import Counter

import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID, CONSONANT

PRIOR = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/models/A_prior_v2.npz'
HV = np.arange(ALTO[0], ALTO[1] + 1)
DV = np.arange(-12, 13)
GOODV = np.array([float(i in (3, 4, 7, 8, 9)) for i in range(12)])
HARSHV = np.array([float(i in (1, 2, 6, 10, 11)) for i in range(12)])
FLOOR = 1e-300
MIS = np.array([e for e in range(-5, 7) if e != 0])       # true - heard when misheard (heard = true + d, d in -6..5)


def _flat_post():
    p = np.zeros(25)
    p[12] = 0.9
    p[12 + MIS] += 0.1 / 11
    return p


PN = _flat_post()


class Prior:
    def __init__(self, path=PRIOR):
        z = np.load(path)
        self.mel = z['mel'].astype(np.float64)            # [pc, iv bucket, metric class, 25]
        self.rhy = z['rhy'].astype(np.float64)            # [s, rest, phase, bar class, ioi, 6]


class PlanA:
    def __init__(self, mix, prior, cfg):
        self.f, self.pr, self.c = mix, prior, dict(cfg)
        m = mix.m
        pm = m.stat @ m.Bc
        self.g = (m.Bc / pm[None, :]) ** self.c['beta']
        self.stats = Counter()

    def start(self, meta):
        self.f.start(meta)
        self.t, self.M, self.pos, self.K = meta['tonic'], meta['M'], meta['pos'], meta['K']
        self.h = -1
        self.planned = False                              # the plan started the sounding note since the last hear
        self.her = []
        self.cth = M1.CTM[:, (HV - self.t) % 12]
        self.wmid = M1.P['w_mid'] * np.abs(HV - ALTO_MID)

    def _pos(self, k):
        return self.pos[k] if k < len(self.pos) else k % self.M

    def _tkey(self, k):
        return M1.tkey(self._pos(k), self.M)

    def _cls(self, x):
        k = int(x)
        if x - k < 0.01:
            return 0 if int(self._pos(k)) % self.M == 0 else 1
        return 2

    def _mel(self, heard, j, x):
        """P(true pitch of the note after heard[j] - heard[j] pitch) for a note starting at x (over DV)."""
        m_c = heard[j][2]
        iv = 11 if j < 1 else int(np.clip(m_c - heard[j - 1][2], -5, 5)) + 5
        return self.pr.mel[(m_c - self.t) % 12, iv, self._cls(x)]

    def _hold_post(self, heard):
        if not self.c['post_hold'] or len(heard) < 2:
            return PN
        m_c, m_b = heard[-1][2], heard[-2][2]
        pr = self._mel(heard, len(heard) - 2, heard[-1][0])   # her held note given the notes before it
        e = np.concatenate(([0], MIS))
        d = np.clip(m_c + e - m_b, -12, 12) + 12
        w = pr[d] * np.where(e == 0, 0.9, 0.1 / 11)
        out = np.zeros(25)
        out[12 + e] = w / w.sum()
        return out

    # ------------------------------------------------------------------ plan
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
        cn = cn + c['gamma'] * cx
        if cn + ch < c['min_cov']:
            return None
        pmel = self._mel(heard, len(heard) - 1, q)
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
            d = np.abs(HV - hs)
            Vc = Vc - w * c['nu'] * (M1.P['w_step'] * d + M1.P['w_big'] * (d > 7))
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

    def _forecast(self, k, q, off_c):
        f = self.f
        a = f.alpha
        for j in range(f.done, k):
            a = M1.Filter._advance(f, a, j, f._beat_ll(j, cur_end=(j + 1 if off_c is None else off_c)))
        e_now = q if off_c is None else min(q, off_c)
        ll_new = f._beat_ll(k, cur_end=e_now)
        ll_hold = f._beat_ll(k, cur_end=k + 1) if off_c is None else ll_new
        bh_new = M1.Filter._advance(f, a, k, ll_new)
        bh_hold = M1.Filter._advance(f, a, k, ll_hold)
        if f._p is None:
            return bh_new, bh_hold
        bg = np.asarray(f._p, dtype=np.float64)
        for j in range(int(self.her[-1][0]) + 1, k + 1):
            bg = bg @ f.m.A[self._tkey(j)]
        lg = f.a * np.log(np.maximum(bg, FLOOR))
        out = []
        for bh in (bh_new, bh_hold):
            x = (1 - f.a) * np.log(np.maximum(bh, FLOOR)) + lg
            x = np.exp(x - x.max())
            out.append(x / x.sum())
        return out

    # ------------------------------------------------------------------ hear
    def hear(self, i, onset, midi, prev_dur):
        f = self.f
        prev, prev_m = f.prev, f.prev_m
        h_r = int(f.step(i, onset, midi, prev_dur))
        first = not self.her
        self.her.append((onset, midi))
        hs, planned = self.h, self.planned
        self.planned = False
        if not first and hs >= 0 and hs == h_r:
            self.stats['hear_same'] += 1
            return None
        if first or hs < 0 or h_r < 0:
            self.stats['hear_enter' if hs < 0 else 'hear_switch'] += 1
            self.h = h_r
            return h_r
        mu = self.c['mu'] if planned else self.c['mu_old']
        if mu >= 0:
            pfit = f.belief(int(onset)) @ M1.CTM
            sr = self._m1score(h_r, midi, pfit, prev, prev_m)
            ss = self._m1score(hs, midi, pfit, prev, prev_m)
            if ss is not None and sr is not None and sr - ss <= mu:
                f.prev = hs
                self.stats['hear_kept_plan' if planned else 'hear_kept_old'] += 1
                return None
        self.stats['hear_switch'] += 1
        self.h = h_r
        return h_r

    def _m1score(self, h, midi, pfit, prev, prev_m):
        """m1_filter_v5lo's chooser score of note h for her heard note midi (None if m1 would not allow it)."""
        P = M1.P
        if h < ALTO[0] or h > min(ALTO[1], midi - 1):
            return None
        iv = abs(midi - h) % 12
        if iv not in CONSONANT or iv == 0:
            return None
        s = pfit[(h - self.t) % 12]
        d = abs(h - prev) if prev is not None else 0
        s -= P['w_step'] * d + P['w_big'] * (d > 7) + P['w_mid'] * abs(h - ALTO_MID)
        if prev is not None and prev_m is not None:
            dm, dh = midi - prev_m, h - prev
            i0 = abs(prev_m - prev) % 12
            if dm != 0 and dh != 0 and (dm > 0) == (dh > 0) and iv in (0, 7) and i0 == iv:
                s -= P['w_par']
        return s


def make(train, cfg):
    return PlanA(r2_mix_v2.build(train), Prior(), cfg)
