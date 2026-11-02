"""Idea B draft library v3 (round 1, run v5rt_1101). Differences from blib_d2:
- Probe(beta): her next pitch is chord-aware: P(n | her current note, chord z) ~ P_delta(n | current) *
  (Bc[z, pc n] / key[pc n]) ** beta, normalised over n for every z, and P(n, z) = b(z) P(n | current, z)
  (beta = 0 gives blib_d2's independent prior and P(z | n) ~ b(z) Bc[z, n]);
- Probe(adapt): the onset prior adapts to the song: every plan point's context (state, time since her last onset,
  beat / half beat) is remembered, its outcome (did a note start there) is read from the heard list one plan point
  later, and p_on = (song count on + adapt * p) / (song count + adapt), first on the coarse then on the fine context
  (adapt = 0: off).
Everything is counted from TRAIN (bench3.train_sets()) or from what the system heard in this song; no song data, no
os.environ.
"""
import bisect
import math

import numpy as np

from systems import m1_filter_v5lo as M1
from theory import ALTO, CONSONANT

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
NOISE = 0.10                                 # bench3's mishearing rate (B2.NOISE)
STEP = 0.5                                   # plan points every half beat


def ioi_bucket(x):
    if x is None:
        return 0
    return 1 if x <= 0.3 else 2 if x <= 0.6 else 3 if x <= 1.1 else 4


def onset_keys(q, last_on, last_off_known, ioi_med, pos_q):
    ph = 0 if abs(q - round(q)) < 1e-9 else 1
    if last_on is None:
        return [(0, 0, ph, int(pos_q == 0), 0), (0, 0, ph, 0), (0, 0, ph), (0, ph)]
    st = 2 if last_off_known else 1
    dq = min(16, int(round((q - last_on) / 0.25)))
    ib = ioi_bucket(ioi_med)
    return [(st, dq, ph, int(pos_q == 0), ib), (st, dq, ph, ib), (st, dq, ph), (st, ph)]


def move_cat(a, b):
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
        sets = {k: v for k, v in train.items() if v}
        npp, nnp = {}, {}
        for ds, songs in sets.items():
            npp[ds] = sum(int(math.ceil(s['K'] / STEP)) for s in songs)
            nnp[ds] = sum(max(len(s['sop']) - 1, 0) for s in songs)
        mpp = sum(npp.values()) / len(sets)
        mnp = sum(nnp.values()) / len(sets)
        self.on = Backoff(4, 2, alpha)
        self.dl = Backoff(3, ND, alpha)
        kp = np.zeros(12)
        for ds, songs in sets.items():
            for s in songs:
                self._song(s, mpp / npp[ds], mnp / max(nnp[ds], 1), kp)
        self.on.finish(1.0)
        self.dl.finish(1.0)
        self.key = (kp + 1.0) / (kp + 1.0).sum()

    def _song(self, s, wo, wd, kp):
        t, M, K = s['tonic'], s['M'], s['K']
        pos = [b['pos'] for b in s['beats']]
        notes = sorted(s['sop'], key=lambda n: n[0])
        ons = [n[0] for n in notes]
        med = []
        for i in range(len(notes)):
            iois = [ons[j] - ons[j - 1] for j in range(max(1, i - 5), i + 1)]
            med.append(float(np.median(iois)) if iois else None)
        for i, (on, off, m) in enumerate(notes):
            kp[(m - t) % 12] += wd * (off - on)
            if i >= 1:
                c = notes[i - 1]
                pm = notes[i - 2][2] if i >= 2 else None
                d = int(np.clip(m - c[2], -DMAX, DMAX))
                self.dl.add(self._dkeys((c[2] - t) % 12, move_cat(pm, c[2]), phase_cat(on)), d + DMAX, wd)
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
            self.on.add(keys, y, wo)
            q += STEP

    @staticmethod
    def _dkeys(pc, mc, ph):
        return [(pc, mc, ph), (pc, ph), (ph,)]

    def onset_keys_heard(self, q, heard, pos_q):
        if not heard:
            return onset_keys(q, None, False, None, pos_q)
        lo, loff, _ = heard[-1]
        ons = [h[0] for h in heard[-7:]]
        iois = [ons[i] - ons[i - 1] for i in range(1, len(ons))]
        med = float(np.median(iois)) if iois else None
        return onset_keys(q, lo, loff is not None, med, pos_q)

    def p_next(self, heard, t, ph):
        """P(her next note = heard[-1] + DS), mixed NU with the key profile within +-6 semitones."""
        m = heard[-1][2]
        pm = heard[-2][2] if len(heard) >= 2 else None
        p = self.dl.get(self._dkeys((m - t) % 12, move_cat(pm, m), ph))
        if self.nu:
            kq = np.where(np.abs(DS) <= 6, self.key[(m + DS - t) % 12], 0.0)
            kq[DS == 6] *= 0.5
            kq[DS == -6] *= 0.5
            p = (1 - self.nu) * p + self.nu * kq / kq.sum()
        return p


def heard_lik(x, n):
    """P(heard midi x | true midi n) under bench3's mishearing (vector over n)."""
    d = x - n
    same = d == 0
    other = (np.abs(d) <= 6) & (d % 12 != 0) & (d != 6)   # a tritone is misheard downwards (d = -6 comes first)
    return np.where(same, 1 - NOISE, np.where(other, NOISE / 11, 0.0))


class Probe:
    """Candidate tables from an r2_mix_v2.Mix (m1 Filter subclass) without changing its state."""

    def __init__(self, pri, a=0.7, beta=0.0, adapt=0.0):
        self.pri, self.a, self.beta, self.adapt = pri, a, beta, adapt
        self.lam = M1.P['lam']

    def start(self, meta):
        self.t, self.M, self.pos, self.K = meta['tonic'], meta['M'], meta['pos'], meta['K']
        self.CTG = M1.CTM[:, (G - self.t) % 12]
        self.pending, self.cnt = [], {}

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

    def beliefs(self, f, q, heard, hold=True):
        k = int(math.floor(q + 1e-9))
        a = f.alpha
        for j in range(f.done, k):
            a = M1.Filter._advance(f, a, j, self._ll(f, heard, j, q))
        bhs = [M1.Filter._advance(f, a, k, self._ll(f, heard, k, q))]
        if hold:
            bhs.append(M1.Filter._advance(f, a, k, self._ll(f, heard, k, k + 1)))
        p = getattr(f, '_p', None)
        if p is None:
            return bhs
        bg = np.maximum(np.asarray(p, dtype=np.float64), 1e-300)
        if f.notes and int(math.floor(f.notes[-1][0] + 1e-9)) < k:
            bg = bg @ f.m.A[M1.tkey(self._pos(k), self.M)]
        out = []
        for bh in bhs:
            lg = (1 - self.a) * np.log(np.maximum(bh, 1e-300)) + self.a * np.log(np.maximum(bg, 1e-300))
            w = np.exp(lg - lg.max())
            out.append(w / w.sum())
        return out

    def joint(self, f, b, pD, n):
        """P(n, z) for her next note n (vector over c + DS) and the chord z of its beat."""
        pcs = (n - self.t) % 12
        if self.beta:
            r = (f.m.Bc[:, pcs] / self.pri.key[pcs][None, :]) ** self.beta     # [z, n]
            cond = pD[None, :] * r
            cond /= cond.sum(1, keepdims=True)
            return (b[:, None] * cond).T                                       # [n, z]
        W = b[None, :] * f.m.Bc[:, pcs].T
        W /= W.sum(1, keepdims=True)
        return pD[:, None] * W

    def _tables(self, J, n):
        """Phit[g], Pharsh[g] from the joint J[n, z]."""
        d = n[:, None] - G[None, :] + OFF
        pn = J.sum(1)
        ect = J @ self.CTG                                                     # [n, g] = sum_z P(n, z) CT(z, g)
        return (GOODV[d] * ect).sum(0), pn @ HARSHV[d]

    def p_onset(self, q, heard, pos_q):
        keys = self.pri.onset_keys_heard(q, heard, pos_q)
        p = float(self.pri.on.get(keys)[1])
        if not self.adapt:
            return p
        if heard:                                                              # outcomes of earlier plan points
            ons = None
            keep = []
            for q0, kc, kf in self.pending:
                if q0 <= q - STEP + 1e-9:
                    if ons is None:
                        ons = [h[0] for h in heard]
                    a = bisect.bisect_left(ons, q0)
                    y = int(a < len(ons) and ons[a] < q0 + ONSET_EPS)
                    for kk in (kc, kf):
                        c = self.cnt.setdefault(kk, [0, 0])
                        c[0] += y
                        c[1] += 1
                else:
                    keep.append((q0, kc, kf))
            self.pending = keep
        kc, kf = ('c',) + keys[-1], ('f',) + keys[-2]
        self.pending.append((q, kc, kf))
        for kk in (kc, kf):
            c = self.cnt.get(kk)
            if c:
                p = (c[0] + self.adapt * p) / (c[1] + self.adapt)
        return p

    def table(self, f, q, heard):
        """Plan point: (p_on, singing, m_c, Phit, Pharsh, hitc, harshc) or None before her first heard note."""
        k = int(math.floor(q + 1e-9))
        if not heard:
            if self.adapt:
                self.p_onset(q, heard, self._pos(k))
            return None
        on_c, off_c, m_c = heard[-1]
        p_on = self.p_onset(q, heard, self._pos(k))
        ph = 0 if abs(q - k) < 1e-9 else 1
        pD = self.pri.p_next(heard, self.t, ph)
        b_on, b_hold = self.beliefs(f, q, heard)
        n = m_c + DS
        Phit, Pharsh = self._tables(self.joint(f, b_on, pD, n), n)
        dc = m_c - G + OFF
        hitc = GOODV[dc] * (b_hold @ self.CTG)
        harshc = HARSHV[dc]
        return p_on, off_c is None, int(m_c), Phit, Pharsh, hitc, harshc

    def hear_table(self, f, onset, midi, before):
        """Hear point, before step(): (P(heard is right), E_hit[g], E_harsh[g]) against her TRUE note; None if her
        previous note is unknown or the heard note is more than DMAX from it."""
        if not before:
            return None
        c = before[-1][2]
        if abs(midi - c) > DMAX:
            return None
        pD = self.pri.p_next(before, self.t, phase_cat(onset))
        n = c + DS
        b_on = self.beliefs(f, onset, before, hold=False)[0]
        J = self.joint(f, b_on, pD, n) * heard_lik(midi, n)[:, None]
        s = J.sum()
        if s <= 0:
            return None
        J /= s
        eh, ex = self._tables(J, n)
        return float(J[midi - c + DMAX].sum()), eh, ex
