"""Idea B draft library (round 1, run v5rt_1101): priors counted from TRAIN only, and the plan-point probe.

Priors (bench3.train_sets(), clean songs, each dataset weighted equally by song count, like m1's Model):
  onset prior  P(a new note of hers starts in [q, q + ONSET_EPS) | what is heard at plan point q with L_PRIOR):
               key = (state none/singing/rest, time since her last heard onset in 1/4 beats, q on a beat or half beat,
               downbeat, median of her last 6 inter-onset intervals bucketed), backed off to coarser keys.
  pitch prior  P(next - current in semitones, -12..12 | current tonic-relative pc, previous move category,
               next onset on a beat / half beat / elsewhere), backed off; mixed NU with a key profile (her current
               note may be misheard).
Probe (no state change in the reactive system): the chord of the beat of q, as r2_mix_v2 would believe it
  (m1 HMM run on over the beats not finished yet, GRU belief pushed one beat when her last note is in an earlier beat,
  log-linear 0.3 / 0.7), once for 'she starts a new note at q' (her sounding note cut at q) and once for 'she holds it'
  (her sounding note counts to the end of the beat). Then for every alto note g:
  Phit[g]   = P(g is a hit against her next note | onset at q): sum_n P(n) good(n, g) sum_z P(z | n) CT(z, g),
              P(z | n) ~ b_on(z) Bc[z, pc(n)];
  Pharsh[g] = P(g is a 2nd / tritone / 7th against her next note);
  hitc[g]   = hit against her sounding (heard) note under the 'hold' belief; harshc[g] = harsh against it.
"""
import bisect
import math

import numpy as np

from systems import m1_filter_v5lo as M1
from theory import ALTO, CONSONANT, MAJOR

HARSH = frozenset({1, 2, 6, 10, 11})
GOOD = CONSONANT - {0}
G = np.arange(ALTO[0], ALTO[1] + 1)          # candidate Duet notes F3..F5
G0, NG = int(G[0]), len(G)
DMAX = 12
DS = np.arange(-DMAX, DMAX + 1)              # her next interval in semitones
ND = len(DS)
OFF = 300
_IV = np.array([abs(d - OFF) % 12 for d in range(2 * OFF + 1)])
GOODV = np.isin(_IV, sorted(GOOD)).astype(np.float64)
HARSHV = np.isin(_IV, sorted(HARSH)).astype(np.float64)
ONSET_EPS = 0.01                             # pop909 onsets sit 0.0002-0.0003 beat after the grid
L_PRIOR = 0.1


# ---------------------------------------------------------------------------------------------------- features
def ioi_bucket(x):
    if x is None:
        return 0
    return 1 if x <= 0.3 else 2 if x <= 0.6 else 3 if x <= 1.1 else 4


def onset_keys(q, last_on, last_off_known, ioi_med, pos_q):
    """Keys from fine to coarse. last_on None = nothing heard yet."""
    ph = 0 if abs(q - round(q)) < 1e-9 else 1
    if last_on is None:
        return [(0, 0, ph, int(pos_q == 0), 0), (0, 0, ph, 0), (0, 0, ph), (0, ph)]
    st = 2 if last_off_known else 1
    dq = min(16, int(round((q - last_on) / 0.25)))
    ib = ioi_bucket(ioi_med)
    return [(st, dq, ph, int(pos_q == 0), ib), (st, dq, ph, ib), (st, dq, ph), (st, ph)]


def move_cat(a, b):
    """Category of her move a -> b (midi): 0 none, 1 repeat, 2 step up, 3 step down, 4 leap up, 5 leap down."""
    if a is None:
        return 0
    d = b - a
    return 1 if d == 0 else 2 if 0 < d <= 2 else 3 if -2 <= d < 0 else 4 if d > 0 else 5


def phase_cat(on):
    f = on - math.floor(on)
    if f < ONSET_EPS:
        return 0
    if abs(f - 0.5) < ONSET_EPS:
        return 1
    return 2


class Backoff:
    """Weighted counts per key level with recursive smoothing toward the coarser level."""

    def __init__(self, nlev, size, alpha):
        self.c = [dict() for _ in range(nlev)]
        self.size, self.alpha = size, alpha
        self.glob = np.zeros(size)

    def add(self, keys, idx, w):
        for lev, k in enumerate(keys):
            v = self.c[lev].get(k)
            if v is None:
                v = self.c[lev][k] = np.zeros(self.size)
            v[idx] += w
        self.glob[idx] += w

    def finish(self, floor):
        g = self.glob + floor
        self.g = g / g.sum()

    def get(self, keys):
        p = self.g
        for lev in range(len(keys) - 1, -1, -1):
            v = self.c[lev].get(keys[lev])
            if v is not None:
                p = (v + self.alpha * p) / (v.sum() + self.alpha)
        return p


class Priors:
    def __init__(self, train, alpha=5.0, nu=0.1):
        self.nu = nu
        n_all = sum(len(v) for v in train.values())
        self.on = Backoff(4, 2, alpha)
        self.dl = Backoff(3, ND, alpha)
        kp = np.zeros(12)
        for songs in train.values():
            if not songs:
                continue
            w = (n_all / len(train)) / len(songs)          # every dataset weighs the same in total
            for s in songs:
                self._song(s, w, kp)
        self.on.finish(1.0)
        self.dl.finish(1.0)
        self.key = (kp + 1.0) / (kp + 1.0).sum()           # P(tonic-relative pc) of her notes

    def _song(self, s, w, kp):
        t, M, K = s['tonic'], s['M'], s['K']
        pos = [b['pos'] for b in s['beats']]
        notes = sorted(s['sop'], key=lambda n: n[0])
        ons = [n[0] for n in notes]
        med = []
        for i in range(len(notes)):
            iois = [ons[j] - ons[j - 1] for j in range(max(1, i - 5), i + 1)]
            med.append(float(np.median(iois)) if iois else None)
        for i, (on, off, m) in enumerate(notes):
            kp[(m - t) % 12] += w * (off - on)
            if i >= 1:
                c = notes[i - 1]
                pm = notes[i - 2][2] if i >= 2 else None
                d = int(np.clip(m - c[2], -DMAX, DMAX))
                self.dl.add(self._dkeys((c[2] - t) % 12, move_cat(pm, c[2]), phase_cat(on)), d + DMAX, w)
        j, q = 0, 0.0
        while q < K:
            while j < len(notes) and notes[j][0] + L_PRIOR <= q:
                j += 1
            a = bisect.bisect_left(ons, q)
            y = int(a < len(ons) and ons[a] < q + ONSET_EPS)
            k = int(q)
            pq = pos[k] if k < len(pos) else k % M
            if j == 0:
                keys = onset_keys(q, None, False, None, pq)
            else:
                lo, loff, _ = notes[j - 1]
                keys = onset_keys(q, lo, loff + L_PRIOR <= q, med[j - 1], pq)
            self.on.add(keys, y, w)
            q += 0.5

    @staticmethod
    def _dkeys(pc, mc, ph):
        return [(pc, mc, ph), (pc, ph), (ph,)]

    def p_onset(self, q, heard, pos_q):
        if not heard:
            return float(self.on.get(onset_keys(q, None, False, None, pos_q))[1])
        lo, loff, _ = heard[-1]
        ons = [h[0] for h in heard[-7:]]
        iois = [ons[i] - ons[i - 1] for i in range(1, len(ons))]
        med = float(np.median(iois)) if iois else None
        return float(self.on.get(onset_keys(q, lo, loff is not None, med, pos_q))[1])

    def p_next(self, heard, t, ph):
        """P(her next note = heard[-1] + DS) (length ND), mixed NU with the key profile placed within +-6 semitones."""
        m = heard[-1][2]
        pm = heard[-2][2] if len(heard) >= 2 else None
        p = self.dl.get(self._dkeys((m - t) % 12, move_cat(pm, m), ph))
        if self.nu:
            kq = np.where(np.abs(DS) <= 6, self.key[(m + DS - t) % 12], 0.0)
            kq[DS == 6] *= 0.5
            kq[DS == -6] *= 0.5
            p = (1 - self.nu) * p + self.nu * kq / kq.sum()
        return p


# ---------------------------------------------------------------------------------------------------- probe
class Probe:
    """Candidate table at a plan point, from an r2_mix_v2.Mix (m1 Filter subclass) without changing its state."""

    def __init__(self, pri, a=0.7):
        self.pri, self.a = pri, a
        self.lam = M1.P['lam']

    def start(self, meta):
        self.t, self.M, self.pos, self.K = meta['tonic'], meta['M'], meta['pos'], meta['K']
        self.CTG = M1.CTM[:, (G - self.t) % 12]            # [z, g]: g is a tone of chord z

    def _pos(self, k):
        return self.pos[k] if k < len(self.pos) else k % self.M

    def _ll(self, f, heard, j, cut):
        v = np.zeros(12)
        for on, off, m in reversed(heard):
            e = off if off is not None else cut
            if e <= j:
                break
            ov = min(e, j + 1) - max(on, j)
            if ov > 0:
                v[(m - self.t) % 12] += ov
        if v.sum() <= 0:
            return np.zeros(M1.S)
        return self.lam * (f.m.logB @ v)

    def beliefs(self, f, q, heard):
        k = int(math.floor(q + 1e-9))
        a = f.alpha
        for j in range(f.done, k):
            a = M1.Filter._advance(f, a, j, self._ll(f, heard, j, q))
        bh_on = M1.Filter._advance(f, a, k, self._ll(f, heard, k, q))
        bh_hold = M1.Filter._advance(f, a, k, self._ll(f, heard, k, k + 1))
        p = getattr(f, '_p', None)
        if p is None:
            return bh_on, bh_hold
        bg = np.maximum(np.asarray(p, dtype=np.float64), 1e-300)
        if f.notes and int(math.floor(f.notes[-1][0] + 1e-9)) < k:
            bg = bg @ f.m.A[M1.tkey(self._pos(k), self.M)]
        out = []
        for bh in (bh_on, bh_hold):
            lg = (1 - self.a) * np.log(np.maximum(bh, 1e-300)) + self.a * np.log(np.maximum(bg, 1e-300))
            w = np.exp(lg - lg.max())
            out.append(w / w.sum())
        return out

    def table(self, f, q, heard):
        """(p_on, singing, m_c, Phit, Pharsh, hitc, harshc) or None before her first heard note."""
        if not heard:
            return None
        on_c, off_c, m_c = heard[-1]
        k = int(math.floor(q + 1e-9))
        p_on = self.pri.p_onset(q, heard, self._pos(k))
        ph = 0 if abs(q - k) < 1e-9 else 1
        pD = self.pri.p_next(heard, self.t, ph)
        b_on, b_hold = self.beliefs(f, q, heard)
        n = m_c + DS
        W = b_on[None, :] * f.m.Bc[:, (n - self.t) % 12].T          # [n, z]
        W /= W.sum(1, keepdims=True)
        ect = W @ self.CTG                                           # [n, g]
        d = n[:, None] - G[None, :] + OFF
        Phit = pD @ (GOODV[d] * ect)
        Pharsh = pD @ HARSHV[d]
        dc = m_c - G + OFF
        hitc = GOODV[dc] * (b_hold @ self.CTG)
        harshc = HARSHV[dc]
        return p_on, off_c is None, int(m_c), Phit, Pharsh, hitc, harshc
