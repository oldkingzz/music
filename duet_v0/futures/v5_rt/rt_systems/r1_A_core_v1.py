"""r1_A_core_v1 (run v5rt_1101, round 1, idea A): LIBRARY for the r1_A_* systems; it has no build(), so it is not a
system itself. The scored variants are r1_A_<name>_v<k>.py, each with its settings as constants.

Idea A: a melody prior (what she sings next, learned from train only) and an expected-fit choice at plan points.
Prior: runs_mini/v5rt_1101/models/A_prior_v1.npz, fitted by runs_mini/v5rt_1101/r1/A/train_prior_A_v1.py from
bench3.train_sets() (pitch n-gram with mishearing on the context; rhythm statistics of the plan-point view at L = 0.1).

plan(q, heard), after her first note was answered:
  rhythm  context (time since her last heard onset, rest or not, beat or half beat, bar class, last inter-onset
          interval) -> pan = P(an attack at q lands on a note not heard yet), pah = P(it lands on her held note),
          cn = expected beats of a new note's first L inside [q, q + 0.5), ch = expected beats of her held note
          inside [q, next decision), cx = first-L beats of a new note just after q + 0.5 (counted with weight gamma).
  pitch   P(her next true pitch | last heard pitch as a scale degree, last heard interval, metric class of q) times
          (Bc[z, pc] / P(pc))^beta for each chord z (m1's P(true pc | chord)), normalised per chord.
  chord   r2_mix_v2's own belief pushed forward (live_predict.PredictDuet._forecast): m1's HMM over the beats heard,
          advanced to beat k = int(q) by its transition table (evidence of beat k: her notes so far in [k, q) for the
          new-note case; her held note to the end of beat k for the hold case), the GRU ensemble's last output pushed
          by the same table once per beat since her last answered note, log-linear 0.3/0.7.
  value   V(h) = (cn + gamma cx) U_new(h) + ch U_hold(h) - voice leading, U = P(hit) - lam P(harsh) - w_x P(Duet at
          or above her); an attack (a note other than the sounding one) also pays kappa (pan P(harsh vs the new
          note) + pah P(harsh vs the held note)): a fresh clash costs more than a note ringing on. Keep the sounding
          note (no attack term) unless a new note beats it by theta; go silent (V = 0) only if 'silence' is on and
          keeping is worse than silence by theta_sil.
hear(i, onset, midi, prev_dur): r2_mix_v2's step() (exactly once per note, the GRU state needs it) = the reactive
  choice h_r. The sounding note stays if it is in the alto range below her heard note, consonant with it and not her
  unison/octave, and m1's chooser score of h_r beats it by at most mu (same belief, same previous note); m1's voice
  leading then follows the kept note. Otherwise Duet switches to h_r.
"""
from collections import Counter

import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID, CONSONANT

PRIOR = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/models/A_prior_v1.npz'
HV = np.arange(ALTO[0], ALTO[1] + 1)                      # Duet's candidate notes
DV = np.arange(-12, 13)                                   # her next pitch - her last heard pitch
GOODV = np.array([float(i in (3, 4, 7, 8, 9)) for i in range(12)])
HARSHV = np.array([float(i in (1, 2, 6, 10, 11)) for i in range(12)])
FLOOR = 1e-300


def _noise_post():
    """P(true - heard) for a heard note under bench3's mishearing with a flat prior: 0.9 right, else one of the 11
    other pitch classes at the nearest octave (heard = true + d, d in -6..-1, 1..5)."""
    p = np.zeros(25)
    p[12] = 0.9
    for d in list(range(-6, 0)) + list(range(1, 6)):
        p[12 - d] += 0.1 / 11
    return p


PN = _noise_post()


class Prior:
    def __init__(self, path=PRIOR):
        z = np.load(path)
        self.mel = z['mel'].astype(np.float64)            # [pc, iv bucket, metric class, 25]
        self.rhy = z['rhy'].astype(np.float64)            # [s, rest, phase, bar class, ioi, 5]


class PlanA:
    def __init__(self, mix, prior, cfg):
        self.f, self.pr, self.c = mix, prior, dict(cfg)
        m = mix.m
        pm = m.stat @ m.Bc                                # P(pc) under the stationary chord mix
        self.g = (m.Bc / pm[None, :]) ** self.c['beta']
        self.stats = Counter()

    # ------------------------------------------------------------------ song
    def start(self, meta):
        self.f.start(meta)
        self.t, self.M, self.pos, self.K = meta['tonic'], meta['M'], meta['pos'], meta['K']
        self.h = -1
        self.her = []                                     # (onset, heard midi) of every note answered
        self.cth = M1.CTM[:, (HV - self.t) % 12]          # [chord, candidate]: candidate is a chord tone
        self.wmid = M1.P['w_mid'] * np.abs(HV - ALTO_MID)

    def _pos(self, k):
        return self.pos[k] if k < len(self.pos) else k % self.M

    def _tkey(self, k):
        return M1.tkey(self._pos(k), self.M)

    # ------------------------------------------------------------------ plan
    def plan(self, q, heard):
        if not self.her or not heard:
            return None
        on_c, off_c, m_c = heard[-1]
        if on_c != self.her[-1][0]:
            return None                                   # heard but not answered yet: leave it to hear()
        c = self.c
        k = int(q)
        s_b = min(48, int(round((q - on_c) * 12)))
        rest_b = 0 if off_c is None else 1 + min(3, int((q - off_c) / 0.5))
        phase = 0 if q == k else 1
        ioi_b = 49 if len(heard) < 2 else min(48, int(round((on_c - heard[-2][0]) * 12)))
        pan, pah, cn, ch, cx = self.pr.rhy[s_b, rest_b, phase, self._tkey(k), ioi_b]
        cn = cn + c['gamma'] * cx
        if cn + ch < c['min_cov']:
            return None
        m_p = heard[-2][2] if len(heard) >= 2 else None
        pc = (m_c - self.t) % 12
        iv = 11 if m_p is None else int(np.clip(m_c - m_p, -5, 5)) + 5
        cls = (0 if int(self._pos(k)) % self.M == 0 else 1) if phase == 0 else 2
        pmel = self.pr.mel[pc, iv, cls]
        b_new, b_hold = self._forecast(k, q, off_c)
        pz = pmel[None, :] * self.g[:, (m_c + DV - self.t) % 12]
        pz /= pz.sum(1, keepdims=True)                    # P(next pitch | chord)
        ivm = (m_c + DV[:, None] - HV[None, :]) % 12
        G, Hh = GOODV[ivm], HARSHV[ivm]
        X = (m_c + DV[:, None] <= HV[None, :]).astype(np.float64)
        pm = b_new @ pz
        ph_new = pm @ Hh
        u_new = (b_new[:, None] * self.cth * (pz @ G)).sum(0) - c['lam'] * ph_new - c['w_x'] * (pm @ X)
        ph_hold = PN @ Hh
        u_hold = (b_hold @ self.cth) * (PN @ G) - c['lam'] * ph_hold - c['w_x'] * (PN @ X)
        w = cn + ch
        V = cn * u_new + ch * u_hold - w * c['nu'] * self.wmid
        hs = self.h
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        v_keep = V[hs - ALTO[0]] if keep_ok else 0.0
        att = c['kappa'] * (pan * ph_new + pah * ph_hold)
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
            self.f.prev = h                               # m1's voice leading follows what Duet sings
            return h
        if keep_ok and c['silence'] and 0.0 > v_keep + c['theta_sil']:
            self.stats['plan_silence'] += 1
            self.h = -1
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
        h_r = int(f.step(i, onset, midi, prev_dur))       # the reactive choice; also advances the GRU state
        first = not self.her
        self.her.append((onset, midi))
        hs = self.h
        if not first and hs >= 0 and hs == h_r:
            self.stats['hear_same'] += 1
            return None
        if first or hs < 0 or h_r < 0:
            self.stats['hear_enter' if hs < 0 else 'hear_switch'] += 1
            self.h = h_r
            return h_r
        pfit = f.belief(int(onset)) @ M1.CTM
        sr = self._m1score(h_r, midi, pfit, prev, prev_m)
        ss = self._m1score(hs, midi, pfit, prev, prev_m)
        if ss is not None and sr is not None and sr - ss <= self.c['mu']:
            f.prev = hs
            self.stats['hear_kept'] += 1
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
