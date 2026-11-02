"""m4_retrieval_v1: hashed n-gram RETRIEVAL of the chord + explicit VOICE-LEADING PLANNER.

History use (explicit): at her note i the query is
    (her current pitch class, bar position of its beat, on/off-beat) + her previous n pitch classes,
    n = NMAX .. 0 (only notes that started within the last DUET_HIST beats; unset/-1 = no limit).
Every note of every training song is indexed under the same keys with the chord(s) sounding under
it (duration weighted). At step time we look up all n and blend them with interpolated backoff
(Witten-Bell style: a context seen often and with few different chords is trusted more), giving
p(chord | history). A second history term: the song's own memory -- what chord distribution the
system itself inferred the last times this exact (pc, previous pc, bar position) occurred in this
song (verses repeat), mixed in with weight W_SONG. Both are cut by DUET_HIST.
Planner: Duet's note h (alto range, below her) maximizes
    sum_z p(z) * [h consonant with her and h a tone of z]                (expected 'music')
    - leap cost - big-leap cost - parallel 5th/8ve cost - unison/octave cost - distance from mid range
    + contrary/oblique motion bonus.
"""
import json
import math
import os
import random

import numpy as np

from theory import ALTO, ALTO_MID, CONSONANT, CT

P = dict(nmax=3, beta=10.0, w_song=0.9, leap=0.02, big=0.3, par=0.5, uni=0.1, mid=0.004,
         contrary=0.03, noise=1, below=1)
P.update(json.loads(os.environ.get('DUET_M4', '{}')))
NMAX = int(P['nmax'])
HIST = float(os.environ.get('DUET_HIST', '-1'))
CTM = np.zeros((7, 12))
for _z in range(7):
    CTM[_z, list(CT[_z])] = 1.0
CANDS = np.arange(ALTO[0], ALTO[1] + 1)


def note_feats(sop, tonic, pos):
    """Per note: (pc rel tonic, beat index, bar pos, onbeat)."""
    out = []
    for on, off, m in sop:
        k = int(math.floor(on + 0.02))
        bp = pos[k] % 8 if 0 <= k < len(pos) else 0
        out.append(((m - tonic) % 12, k, bp, int(abs(on + 0.02 - k) < 0.06)))
    return out


def keys(pcs, i, onsets, pc, bp, ob, now=None, hist=-1.0):
    """Lookup keys from most general (n=0) to most specific, respecting DUET_HIST."""
    base = (pc, bp, ob)
    ks = [base]
    ctx = ()
    for n in range(1, NMAX + 1):
        j = i - n
        if j < 0 or (hist >= 0 and now - onsets[j] > hist):
            break
        ctx = (pcs[j],) + ctx
        ks.append(base + ctx)
    return ks


class System:
    def __init__(self, train):
        tab = {}
        rng = random.Random(5160)
        for songs in train.values():
            for s in songs:
                if not s.get('beats'):
                    continue
                z = [b['z'] for b in s['beats']]
                pos = [b['pos'] for b in s['beats']]
                sop = sorted(s['sop'], key=lambda n: n[0])
                for copy in range(1 + int(P['noise'])):
                    if copy == 0:
                        sp = sop
                    else:                                      # 10 % mishearing, like the bench
                        sp = [[a, b, (m + rng.randint(1, 11)) if rng.random() < 0.1 else m] for a, b, m in sop]
                    f = note_feats(sp, s['tonic'], pos)
                    pcs = [x[0] for x in f]
                    ons = [n[0] for n in sp]
                    for i, (on, off, m) in enumerate(sp):
                        w = np.zeros(7)
                        for k in range(int(on), int(math.ceil(off))):
                            if k < len(z) and z[k] >= 0:
                                w[z[k]] += min(off, k + 1) - max(on, k)
                        if w.sum() <= 0:
                            continue
                        w /= w.sum()
                        pc, _, bp, ob = f[i]
                        for key in keys(pcs, i, ons, pc, bp, ob):
                            a = tab.get(key)
                            if a is None:
                                tab[key] = w.copy()
                            else:
                                a += w
        self.tab = tab
        self.prior = np.full(7, 1 / 7)

    # ---------------------------------------------------------------- streaming
    def start(self, meta):
        self.t, self.pos = meta['tonic'], meta['pos']
        self.pcs, self.ons, self.mid = [], [], []
        self.prev_h, self.prev_m = None, None
        self.mem = {}                                       # song memory: key -> (sum p, last onset list)

    def chord_dist(self, i, onset, pc, bp, ob):
        p = self.prior
        for key in keys(self.pcs, i, self.ons, pc, bp, ob, onset, HIST):
            c = self.tab.get(key)
            if c is None:
                break
            tot = c.sum()
            typ = (c > 0.05 * tot).sum()
            lam = tot / (tot + P['beta'] * typ)
            p = lam * c / tot + (1 - lam) * p
        # song memory: the same short context earlier in this song
        sk = (pc, self.pcs[i - 1] if i and (HIST < 0 or onset - self.ons[i - 1] <= HIST) else -1, bp)
        e = self.mem.get(sk)
        if e is not None and P['w_song'] > 0:
            ent = [(o, q) for o, q in e if HIST < 0 or onset - o <= HIST]
            if ent:
                q = sum(x for _, x in ent) / len(ent)
                a = P['w_song'] * len(ent) / (len(ent) + 1)
                p = (1 - a) * p + a * q
        return p, sk

    def plan(self, midi, p):
        h = CANDS
        iv = np.abs(midi - h) % 12
        cons = np.isin(iv, list(CONSONANT)).astype(float)
        fit = p @ CTM[:, (h - self.t) % 12]                    # P(h is a chord tone)
        s = cons * fit
        s = s - P['uni'] * (iv == 0) - P['mid'] * np.abs(h - ALTO_MID)
        if P['below']:
            s = s - 1.0 * (h >= midi)
        if self.prev_h is not None:
            d = h - self.prev_h
            s = s - P['leap'] * np.abs(d) - P['big'] * (np.abs(d) > 7)
            dm = midi - self.prev_m
            i0 = abs(self.prev_m - self.prev_h) % 12
            if dm != 0:
                same = (np.sign(d) == np.sign(dm)) & (d != 0)
                s = s - P['par'] * (same & np.isin(iv, (0, 7)) & (iv == i0))
                s = s + P['contrary'] * ((np.sign(d) == -np.sign(dm)) | (d == 0))
        return int(h[int(np.argmax(s))])

    def step(self, i, onset, midi, prev_dur):
        k = int(math.floor(onset + 0.02))
        bp = self.pos[k] % 8 if 0 <= k < len(self.pos) else 0
        ob = int(abs(onset + 0.02 - k) < 0.06)
        pc = (midi - self.t) % 12
        p, sk = self.chord_dist(i, onset, pc, bp, ob)
        self.mem.setdefault(sk, []).append((onset, p))
        self.pcs.append(pc)
        self.ons.append(onset)
        h = self.plan(midi, p)
        self.prev_h, self.prev_m = h, midi
        return h


def build(train):
    return System(train)
