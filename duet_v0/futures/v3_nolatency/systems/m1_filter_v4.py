"""m1_filter_v4: chord belief filter with a STYLE mixture (factorial HMM forward filter).

History use: everything she sang so far is compressed into one running table of probabilities over
(style, chord): 4 styles (bach, folk, pop, pop909; each has its own chord-transition and note tables
learned from labelled train songs) x the 7 diatonic chords. Every finished beat, the table is pushed
forward by each style's transition table (keyed by bar position: chords change mostly on the downbeat)
and reweighted by how well the notes she held in that beat fit each chord. Because a style that keeps
predicting her notes badly loses weight, the past also tells Duet WHICH kind of song this is.
Her current note (zero latency) is applied as evidence for the current beat before Duet chooses.
Duet then picks the alto note maximising P(it is a tone of the current chord) among notes consonant
with what it heard, minus voice-leading costs (step size, leaps > 7, parallel 5ths/8ves, octave/unison
doubling, distance from mid-range).

DUET_HIST = H beats: the filter is rerun over only the last H beats of her melody (fresh prior before
that). Duet's own previous note (used for voice leading) is always kept: it is not her melody.
"""
import os

import numpy as np

from theory import ALTO, ALTO_MID, CONSONANT, CT

S = 7
CTM = np.zeros((S, 12))
for _z in range(S):
    for _p in CT[_z]:
        CTM[_z, _p] = 1.0

P = dict(noise=0.1, lam=1.5, kappa=0.5, back=200.0, eps=0.0, styles=1, w_step=0.01, w_big=0.4,
         w_par=0.6, w_mid=0.004, w_oct=0.6, w_unis=1.0, cur_w=1.0)
for _k in list(P):
    if os.environ.get('M1_' + _k.upper()):
        P[_k] = float(os.environ['M1_' + _k.upper()])


def tkey(pos, M):
    return int(pos) % 4 if M == 4 else (4 if int(pos) % M == 0 else 5)


def counts(songs):
    A = np.zeros((6, S, S))
    B = np.zeros((S, 12))
    pi = np.zeros(S)
    for s in songs:
        t, M = s['tonic'], s['M']
        z = [b['z'] for b in s['beats']]
        K = len(z)
        obs = np.zeros((K, 12))
        for on, off, m in s['sop']:
            for k in range(max(int(on), 0), min(int(np.ceil(off)), K)):
                ov = min(off, k + 1) - max(on, k)
                if ov > 0:
                    obs[k, (m - t) % 12] += ov
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
    return A, B, pi


def tables(A, B, pi, base=None):
    kap, nz = P['kappa'], P['noise']
    if base is not None:                       # back off a style's counts toward the pooled tables
        bA, bB, bpi = base
        A = A + P['back'] * bA
        B = B + P['back'] * bB * 10
        pi = pi + 5 * bpi
    A = (A + kap) / (A + kap).sum(-1, keepdims=True)
    Bc = (B + 1.0) / (B + 1.0).sum(-1, keepdims=True)
    logB = np.log((1 - nz) * Bc + nz / 12)     # heard pc: 10 % of notes are misheard
    pi = (pi + 1) / (pi + 1).sum()
    return A, logB, pi


class Model:
    def __init__(self, train):
        per = {ds: counts(v) for ds, v in train.items()}
        pooled = tuple(sum(c[i] for c in per.values()) for i in range(3))
        pA, pB, ppi = tables(*pooled)
        base = (pA, np.exp(pB) / np.exp(pB).sum(-1, keepdims=True), ppi)
        if P['styles']:
            ts = [tables(*per[ds], base=base) for ds in sorted(per)]
        else:
            ts = [(pA, pB, ppi)]
        self.A = np.stack([t[0] for t in ts])        # [style, key, z, z']
        self.logB = np.stack([t[1] for t in ts])     # [style, z, pc]
        self.pi = np.stack([t[2] for t in ts])       # [style, z]
        self.nst = len(ts)


class Filter:
    def __init__(self, model):
        self.m = model
        self.H = int(os.environ.get('DUET_HIST', '-1'))

    def start(self, meta):
        self.t, self.M, self.pos, self.K = meta['tonic'], meta['M'], meta['pos'], meta['K']
        self.notes = []            # [onset, offset or None, relpc]
        self.ev = []               # per finished beat: loglik [style, z]
        self.alpha = None          # joint posterior [style, z] after the last finished beat
        self.done = 0
        self.prev = None
        self.prev_m = None

    def _beat_ll(self, k):
        v = np.zeros(12)
        for on, off, pc in reversed(self.notes):
            e = off if off is not None else k + 1
            if e <= k:
                break
            ov = min(e, k + 1) - max(on, k)
            if ov > 0:
                v[pc] += ov
        if v.sum() <= 0:
            return np.zeros((self.m.nst, S))
        return P['lam'] * (self.m.logB @ v)

    def _pos(self, k):
        return self.pos[k] if k < len(self.pos) else k % self.M

    def _advance(self, alpha, k, ll):
        if alpha is None:
            pr = self.m.pi / self.m.nst
        else:
            key = tkey(self._pos(k), self.M)
            pr = np.einsum('sz,szy->sy', alpha, self.m.A[:, key])
            if P['eps'] > 0:
                ps = pr.sum(1, keepdims=True)
                pr = pr / ps * ((1 - P['eps']) * ps + P['eps'] / self.m.nst)
        w = pr * np.exp(ll - ll.max())
        return w / w.sum()

    def _finish_beats(self, upto):
        while self.done < upto:
            ll = self._beat_ll(self.done)
            self.ev.append(ll)
            if self.H < 0:
                self.alpha = self._advance(self.alpha, self.done, ll)
            self.done += 1

    def belief(self, k):
        """Joint posterior [style, chord] for beat k given all notes heard so far (current included)."""
        ll = self._beat_ll(k) * P['cur_w']
        if self.H < 0:
            return self._advance(self.alpha, k, ll)
        a = None
        for j in range(max(0, k - self.H), k):
            a = self._advance(a, j, self.ev[j])
        return self._advance(a, k, ll)

    def step(self, i, onset, midi, prev_dur):
        if self.notes and prev_dur is not None:
            self.notes[-1][1] = self.notes[-1][0] + prev_dur
        k = int(onset)
        self._finish_beats(k)
        self.notes.append([onset, None, (midi - self.t) % 12])
        self.last_joint = self.belief(k)
        b = self.last_joint.sum(0)
        pfit = b @ CTM
        ref = self.prev if self.prev is not None else ALTO_MID
        best, bs = -1, -1e9
        for h in range(ALTO[0], min(ALTO[1], midi - 1) + 1):
            iv = abs(midi - h) % 12
            if iv not in CONSONANT:
                continue
            s = pfit[(h - self.t) % 12]
            if iv == 0:
                s -= P['w_oct']
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


def build(train):
    return Filter(Model(train))
