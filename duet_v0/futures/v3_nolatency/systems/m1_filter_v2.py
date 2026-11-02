"""m1_filter: chord belief filter (HMM forward filter over the 7 diatonic chords).

History use: everything she sang before is compressed into one running probability over the 7 chords.
Each finished beat updates it with the notes she held in that beat (weighted by how long); between beats
the belief moves by a transition table learned from train labels, separate per bar position (chords
change mostly on beat 1). Her CURRENT note (zero latency) is applied as evidence for the current beat
before Duet picks. Duet then picks the alto note that maximises P(chord tone of current chord) while
consonant with her current note, minus voice-leading costs.

DUET_HIST = H (beats): the filter is rerun over only the last H beats (uniform-ish prior before that).
Duet's own previous note (for voice leading) is always used; it is not her melody.
"""
import os

import numpy as np

from theory import ALTO, ALTO_MID, CONSONANT, CT, MAJOR

S = 7
CTM = np.zeros((S, 12))
for _z in range(S):
    for _p in CT[_z]:
        CTM[_z, _p] = 1.0

P = dict(noise=0.1, robust=1, lam=1.5, kappa=0.5, w_step=0.01, w_big=0.4, w_par=0.6, w_mid=0.004, w_oct=0.6, w_unis=1.0,
         order2=0, cur_w=1.0, below=1, w_cons=10.0)
for _k in list(P):
    if os.environ.get('M1_' + _k.upper()):
        P[_k] = float(os.environ['M1_' + _k.upper()])


def tkey(pos, M):
    return int(pos) % 4 if M == 4 else (4 if int(pos) % M == 0 else 5)


def beat_obs(sop, K):
    """12-vector per beat: how long (in beats) each relative pc sounded in that beat."""
    obs = np.zeros((K, 12))
    for on, off, m in sop:
        k0, k1 = int(on), int(np.ceil(off))
        for k in range(max(k0, 0), min(k1, K)):
            ov = min(off, k + 1) - max(on, k)
            if ov > 0:
                obs[k, m % 12] += ov
    return obs


class Model:
    def __init__(self, train):
        kap = P['kappa']
        A = np.zeros((6, S, S))
        A2 = np.zeros((6, S, S, S))
        B = np.zeros((S, 12))
        pi = np.zeros(S)
        for songs in train.values():
            for s in songs:
                t, M = s['tonic'], s['M']
                z = [b['z'] for b in s['beats']]
                K = len(z)
                rel = [[on, off, (m - t) % 12] for on, off, m in s['sop']]
                obs = beat_obs(rel, K)
                first = True
                for k in range(K):
                    if z[k] < 0:
                        continue
                    B[z[k]] += obs[k]
                    if first:
                        pi[z[k]] += 1
                        first = False
                    if k >= 1 and z[k - 1] >= 0:
                        A[tkey(s['beats'][k]['pos'], M), z[k - 1], z[k]] += 1
                        if k >= 2 and z[k - 2] >= 0:
                            A2[tkey(s['beats'][k]['pos'], M), z[k - 2], z[k - 1], z[k]] += 1
        self.A = (A + kap) / (A + kap).sum(-1, keepdims=True)
        # 2nd order backed off toward 1st order
        self.A2 = (A2 + 5.0 * self.A[:, None, :, :]) / (A2.sum(-1, keepdims=True) + 5.0)
        Bc = (B + 1.0) / (B + 1.0).sum(-1, keepdims=True)
        self.Bc = Bc                                           # P(true pc | chord), clean
        self.logB = np.log((1 - P['noise']) * Bc + P['noise'] / 12)   # heard pc: 10 % misheard
        nz = P['noise']
        self.N = np.full((12, 12), nz / 11)                   # N[t, heard]
        np.fill_diagonal(self.N, 1 - nz)
        self.pi = (pi + 1) / (pi + 1).sum()
        tot = sum(self.A[:, i, :].sum(0) for i in range(S))
        self.stat = tot / tot.sum()


class Filter:
    def __init__(self, model):
        self.m = model
        self.H = int(os.environ.get('DUET_HIST', '-1'))

    def start(self, meta):
        self.t, self.M, self.pos, self.K = meta['tonic'], meta['M'], meta['pos'], meta['K']
        self.notes = []            # [onset, offset or None, relpc]
        self.ev = []               # per finished beat: loglik vector
        self.alpha = None          # filtered posterior after last finished beat (full history)
        self.alpha2 = None         # joint (z_{k-1}, z_k) for order2
        self.done = 0              # number of finished beats
        self.prev = None
        self.prev_m = None

    # ---------------------------------------------------------------- evidence
    def _beat_ll(self, k, cur_end=None):
        v = np.zeros(12)
        for on, off, pc in reversed(self.notes):
            e = off if off is not None else (k + 1 if cur_end is None else cur_end)
            if e <= k:
                break
            ov = min(e, k + 1) - max(on, k)
            if ov > 0:
                v[pc] += ov
        if v.sum() <= 0:
            return np.zeros(S)
        return P['lam'] * (self.m.logB @ v)

    def _pos(self, k):
        return self.pos[k] if k < len(self.pos) else k % self.M

    def _advance(self, alpha, k, ll, a2=None):
        """alpha over beat k-1 -> alpha over beat k with evidence ll."""
        if alpha is None:
            pr = self.m.pi.copy()
        else:
            pr = alpha @ self.m.A[tkey(self._pos(k), self.M)]
        w = pr * np.exp(ll - ll.max())
        return w / w.sum()

    def _advance2(self, j, k, ll):
        """2nd-order: j is joint P(z_{k-2}, z_{k-1}) -> joint P(z_{k-1}, z_k)."""
        if j is None:
            pr = self.m.pi.copy()
            w = pr * np.exp(ll - ll.max())
            w /= w.sum()
            return np.outer(self.m.stat, w) / 1.0, w
        T = self.m.A2[tkey(self._pos(k), self.M)]          # [z2, z1, z]
        jn = np.einsum('ab,abc->bc', j, T) * np.exp(ll - ll.max())[None, :]
        jn /= jn.sum()
        return jn, jn.sum(0)

    def _finish_beats(self, upto):
        while self.done < upto:
            k = self.done
            ll = self._beat_ll(k)
            self.ev.append(ll)
            if P['order2']:
                self.alpha2, self.alpha = self._advance2(self.alpha2, k, ll)
            else:
                self.alpha = self._advance(self.alpha, k, ll)
            self.done += 1

    def belief(self, k):
        """Posterior over the chord of beat k, given all notes up to now (current one included)."""
        ll = self._beat_ll(k) * P['cur_w']
        if self.H < 0:
            if P['order2']:
                return self._advance2(self.alpha2, k, ll)[1]
            return self._advance(self.alpha, k, ll)
        a = None
        k0 = max(0, k - self.H)
        for j in range(k0, k):
            a = self._advance(a, j, self.ev[j])
        return self._advance(a, k, ll)

    # ---------------------------------------------------------------- choice
    def step(self, i, onset, midi, prev_dur):
        if self.notes and prev_dur is not None:
            self.notes[-1][1] = self.notes[-1][0] + prev_dur
        k = int(onset)
        self._finish_beats(k)
        self.notes.append([onset, None, (midi - self.t) % 12])
        b = self.belief(k)
        pfit = b @ CTM                                        # P(pc is a chord tone), relative pcs
        hp = (midi - self.t) % 12
        if P['robust']:
            # joint over (chord z, her TRUE pc t) given what was heard
            pt = self.m.Bc * self.m.N[:, hp][None, :]          # [z, t]
            pt /= pt.sum(1, keepdims=True)
            jz = b[:, None] * pt                               # [z, t]
            # true midi for each t: nearest to heard within +-6
            tm = np.array([midi + ((t - hp + 6) % 12) - 6 for t in range(12)])
        ref = self.prev if self.prev is not None else ALTO_MID
        hi = min(ALTO[1], midi - 1) if P['below'] else ALTO[1]
        best, bs = -1, -1e9
        for h in range(ALTO[0], hi + 1):
            iv = abs(midi - h) % 12
            if P['robust']:
                cv = np.fromiter(((abs(x - h) % 12) in CONSONANT for x in tm), float, 12)
                s = P['w_cons'] * 0 + float(CTM[:, (h - self.t) % 12] @ (jz @ cv))
                if s <= 0:
                    continue
            else:
                if iv not in CONSONANT:
                    continue
                s = pfit[(h - self.t) % 12]
            if iv == 0:
                s -= P['w_oct'] if h != midi else P['w_unis']
            d = abs(h - ref) if self.prev is not None else 0
            s -= P['w_step'] * d + P['w_big'] * (d > 7) + P['w_mid'] * abs(h - ALTO_MID)
            if self.prev is not None and self.prev_m is not None:
                dm, dh = midi - self.prev_m, h - self.prev
                i0 = abs(self.prev_m - self.prev) % 12
                if dm != 0 and dh != 0 and (dm > 0) == (dh > 0) and iv in (0, 7) and i0 == iv:
                    s -= P['w_par']
            if s > bs:
                best, bs = h, s
        if best >= 0:
            self.prev = best
        self.prev_m = midi
        return best


_CACHE = {}


def build(train):
    key = tuple(sorted((k, len(v)) for k, v in train.items())) + tuple(sorted(P.items()))
    if key not in _CACHE:
        _CACHE[key] = Model(train)
    return Filter(_CACHE[key])
