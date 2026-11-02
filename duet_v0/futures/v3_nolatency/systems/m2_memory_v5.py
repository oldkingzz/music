"""m2_memory_v5 = v3 + emission also conditioned on her previous pitch class (backoff to v3 emission).
m2_memory_v3 = v1 + metric-aware chord model (emission depends on whether the note starts a bar,
a beat or falls off the beat; transitions depend on the bar position of the new beat).

m2_memory: chord-belief filter + SONG MEMORY of repeated melodic contexts.

History use, step by step at her note i:
 1. Chord filter (note-level HMM, trained on corpus counts): belief over the 7 diatonic chords from
    every past note (or only the last DUET_HIST beats), transition depends on whether a beat / a
    bar line was crossed since her previous note.
 2. Hindsight: once the next L notes have been heard, the chord at an earlier note j is re-estimated
    with fixed-lag smoothing (it uses notes j..j+L, all in the past by now). That smoothed belief is
    written to the song memory under note j's context key (last n tonic-relative pitch classes +
    bar position of the onset, n = 4, 3, 2 backoff).
 3. When the current context matches a stored key, the stored hindsight belief is multiplied into
    the current belief (weight grows with key length and number of earlier matches), so a repeated
    phrase gets the chord we figured out for it last time -> consistent, better-informed harmony.
 4. Online adaptation: this song's smoothed chord transitions are added to the transition counts.
Duet then sings the alto note below her maximizing P(chord tone) * consonant - voice-leading costs.
Env: DUET_HIST (beats of past melody usable; -1/unset = all), DUET_M2 (json overrides of P).
"""
import json
import math
import os

import numpy as np

from theory import ALTO, ALTO_MID, CONSONANT, CT, MAJOR

P = dict(L=3, ns=(4, 3, 2), wmem=(1.0, 0.7, 0.4), memcap=3.0, adapt=0.5, vl=0.02, vc=0.004,
         big=0.15, hrep=0.0, par=0.3, dbl=0.25, outkey=0.1, kappa=0.5, use_mem=1, use_adapt=1, below=1)
P.update(json.loads(os.environ.get('DUET_M2', '{}')))
HIST = float(os.environ.get('DUET_HIST', '-1'))
NZ = 7
FIT = np.array([[1.0 if pc in CT[z] else 0.0 for z in range(NZ)] for pc in range(12)])  # [pc, z]
STATS = {'notes': 0, 'fired': {n: 0 for n in P['ns']}}


NT = 10


def ttype(prev_on, on, pos_of):
    """0 = same beat; 1 = crossed a bar line; else 2 + bar position of the new beat (capped)."""
    if prev_on is None:
        return 1
    k0, k1 = int(prev_on), int(on)
    if k1 == k0:
        return 0
    for k in range(k0 + 1, k1 + 1):
        if pos_of(k) == 0:
            return 1
    p = pos_of(k1)
    return 2 + min(max(p, 0), NT - 3)


def strength(on, pos_of):
    if on - int(on) > 1e-6:
        return 2
    return 0 if pos_of(int(on)) == 0 else 1


class Model:
    def __init__(self, train):
        E = np.full((3, 12, NZ), 0.5, dtype=float)
        E2 = np.zeros((3, 13, 12, NZ))           # emission counts pc | z (duration weighted)
        T = np.full((NT, NZ, NZ), float(P['kappa']))
        pri = np.ones(NZ)
        for songs in train.values():
            for s in songs:
                t, beats = s['tonic'], s['beats']
                posf = lambda k: beats[k]['pos'] if k < len(beats) else -1
                prev, pon, ppc = None, None, 12
                for on, off, m in sorted(s['sop'], key=lambda n: n[0]):
                    cpc = (m - t) % 12
                    if True:
                        kk = int(on)
                        zz = beats[kk]['z'] if kk < len(beats) else -1
                        if zz >= 0:
                            E2[strength(on, posf), ppc, cpc, zz] += max(off - on, 0.25)
                        ppc = cpc
                    k = int(on)
                    z = beats[k]['z'] if k < len(beats) else -1
                    if z < 0:
                        prev, pon = None, None
                        continue
                    E[strength(on, posf), (m - t) % 12, z] += max(off - on, 0.25)
                    if prev is None:
                        pri[z] += 1
                    else:
                        T[ttype(pon, on, posf), prev, z] += 1
                    prev, pon = z, on
        self.E1 = E / E.sum(1, keepdims=True)
        b = P.get('bo', 5.0)
        E2 = E2 + b * self.E1[:, None, :, :]
        self.E = E2 / E2.sum(2, keepdims=True)        # [strength, prev pc (12 = none), pc, z]
        self.Tc = T
        self.pri = pri / pri.sum()


class Duet:
    def __init__(self, model):
        self.m = model

    def start(self, meta):
        self.t, self.pos = meta['tonic'], meta['pos']
        self.M = meta['M']
        self.notes = []          # (on, midi, rel pc, metric strength)
        self.alpha = []          # filtered posterior per note
        self.tt = []             # transition type into each note
        self.mem = {}            # key -> [sum of smoothed beliefs, count, last onset]
        self.Tsong = np.zeros_like(self.m.Tc)
        self.T = self.m.Tc / self.m.Tc.sum(2, keepdims=True)
        self.h = []
        self._sm_prev = None
        self.hmem = {}           # context key -> interval (semitones below her) Duet sang there last time
        self.mem_used = []       # per note: (n or 0, mem belief or None, filter belief)

    def posf(self, k):
        return self.pos[k] if k < len(self.pos) else -1

    def key(self, j, n):
        if j - n + 1 < 0:
            return None
        on = self.notes[j][0]
        bp = (self.posf(int(on)), round((on - int(on)) * 4))
        return (n, bp) + tuple(self.notes[q][2] for q in range(j - n + 1, j + 1))

    def trans(self, typ):
        return self.T[typ]

    def _refresh_T(self):
        Tn = self.m.Tc / self.m.Tc.sum(2, keepdims=True)
        if not P['use_adapt']:
            self.T = Tn
        elif P.get('beta'):          # Dirichlet: this song's soft counts with a corpus prior of strength beta
            self.T = (self.Tsong + P['beta'] * Tn) / (self.Tsong.sum(2, keepdims=True) + P['beta'])
        else:
            Tc = self.m.Tc + P['adapt'] * self.Tsong
            self.T = Tc / Tc.sum(2, keepdims=True)

    def step(self, i, onset, midi, prev_dur):
        pc = (midi - self.t) % 12
        typ = ttype(self.notes[-1][0] if self.notes else None, onset, self.posf)
        self.notes.append((onset, midi, pc, strength(onset, self.posf), self.notes[-1][2] if self.notes else 12))
        self.tt.append(typ)
        e = self.m.E[self.notes[-1][3], self.notes[-1][4], pc]
        # 1. forward filter (optionally restricted to last HIST beats)
        if HIST >= 0:
            j0 = i
            while j0 > 0 and self.notes[j0 - 1][0] >= onset - HIST:
                j0 -= 1
            a = self.m.pri * self.m.E[self.notes[j0][3], 12 if j0 == 0 or HIST >= 0 else self.notes[j0][4], self.notes[j0][2]]
            a /= a.sum()
            for q in range(j0 + 1, i + 1):
                a = (a @ self.trans(self.tt[q])) * self.m.E[self.notes[q][3], self.notes[q][4], self.notes[q][2]]
                a /= a.sum()
        else:
            a = self.m.pri * e if i == 0 else (self.alpha[-1] @ self.trans(typ)) * e
            a /= a.sum()
        self.alpha.append(a)
        # 2. hindsight: smooth note j = i - L with notes j..i, store in memory
        L = P['L']
        j = i - L
        if j >= 0:
            beta = np.ones(NZ)
            for q in range(i, j, -1):
                beta = self.trans(self.tt[q]) @ (self.m.E[self.notes[q][3], self.notes[q][4], self.notes[q][2]] * beta)
            sm = self.alpha[j] * beta
            sm /= sm.sum()
            for n in P['ns']:
                k = self.key(j, n)
                if k is not None:
                    r = self.mem.setdefault(k, [np.zeros(NZ), 0, 0.0])
                    r[0] += sm
                    r[1] += 1
                    r[2] = self.notes[j][0]
            # 4. online transition adaptation from smoothed pair (j-1 -> j)
            if P['use_adapt'] and j >= 1:
                self.Tsong[self.tt[j]] += np.outer(self._sm_prev if self._sm_prev is not None else self.alpha[j - 1], sm)
                self._refresh_T()
            self._sm_prev = sm
        # 3. memory lookup for the current context
        b = a
        fired = 0
        mb = None
        if P['use_mem']:
            for n, w in zip(P['ns'], P['wmem']):
                k = self.key(i, n)
                r = self.mem.get(k) if k is not None else None
                if r is not None and (HIST < 0 or r[2] >= onset - HIST):
                    mb = r[0] / r[1]
                    wt = w * min(r[1], P['memcap'])
                    b = a * (mb + 1e-3) ** wt
                    b /= b.sum()
                    fired = n
                    break
        STATS['notes'] += 1
        if fired:
            STATS['fired'][fired] += 1
        self.mem_used.append((fired, int(a.argmax()), int(b.argmax()), int(mb.argmax()) if fired else -1))
        self.belief = b
        kh = self.key(i, P['ns'][0])
        rep = self.hmem.get(kh) if (kh is not None and P.get('hrep')) else None
        h = self.choose(midi, b, rep)
        if kh is not None and h >= 0:
            self.hmem[kh] = midi - h
        self.h.append(h)
        return h

    def choose(self, midi, b, rep=None):
        pf = FIT @ b                                  # P(chord tone) per pc
        prev = self.h[-1] if self.h else None
        pm = self.notes[-2][1] if len(self.notes) >= 2 else None
        best, bs = -1, -1e9
        hi = min(ALTO[1], midi - 1) if P['below'] else ALTO[1]
        for h in range(ALTO[0], hi + 1):
            iv = abs(midi - h) % 12
            if iv not in CONSONANT:
                continue
            hp = (h - self.t) % 12
            s = pf[hp] - P['vc'] * abs(h - ALTO_MID)
            if iv == 0:
                s -= P['dbl']
            if hp not in MAJOR:
                s -= P['outkey']
            if rep is not None and midi - h == rep:
                s += P['hrep']             # sing the same harmony as last time this phrase came
            if prev is not None and prev >= 0:
                d = abs(h - prev)
                s -= P['vl'] * d
                if d > 7:
                    s -= P['big']
                if pm is not None:
                    dm, dh = midi - pm, h - prev
                    i0 = abs(pm - prev) % 12
                    if dm and dh and (dm > 0) == (dh > 0) and iv in (0, 7) and i0 == iv:
                        s -= P['par']
            if s > bs:
                best, bs = h, s
        return best


_MODEL = None


def build(train):
    global _MODEL
    _MODEL = Model(train)
    import atexit

    def dump():
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, 'hb', 'm2_memory_stats.json'), 'w') as f:
            json.dump({'P': {k: v for k, v in P.items()}, 'HIST': HIST,
                       'notes': STATS['notes'], 'fired': STATS['fired']}, f, default=str)
    atexit.register(dump)
    return Duet(_MODEL)
