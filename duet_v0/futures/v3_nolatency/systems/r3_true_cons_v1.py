"""r3_true_cons_v1 (v4 round 3, idea E5): r2_mix_v2's chord belief, chooser = m1_filter_v5lo's step in its robust form: every
note below her heard note is scored by P(chord tone AND consonant with her TRUE note | heard note, belief), 10 % mishearing."""
import numpy as np
import torch

from systems import m1_filter_v5lo
from systems.m3_seq_v4 import V3, Ens, _load
from systems.r2_mix_v2 import Mix
from theory import ALTO, ALTO_MID, CONSONANT

CTM = m1_filter_v5lo.CTM   # [chord, relative pc] = 1 for a chord tone (read only)
W_STEP = 0.01              # m1_filter_v5lo's 6 chooser weights, unchanged
W_BIG = 0.4
W_PAR = 0.6
W_MID = 0.004
W_OCT = 0.3
W_UNIS = 1.0               # inert: Duet stays strictly below her heard note
MODE = 'all'               # E5 scoring on every note; 'off' = the champion's hard rule (faithfulness check only)
TRUE_LO = -6               # her true note is placed at heard + ((t - heard + 6) % 12) - 6, i.e. -6..+5 (m1's robust branch)


class TrueCons(Mix):
    """r2_mix_v2's belief; chooser = a copy of m1_filter_v5lo.Filter.step (look 0, below 1) whose robust branch is E5."""
    mode = MODE
    true_lo = TRUE_LO

    def use_e5(self, hp):
        """Score this note by expected consonance with her true note? (hp = heard pitch class relative to the tonic)"""
        return self.mode == 'all'

    def step(self, i, onset, midi, prev_dur):
        self._p = self.ens.chord_probs(onset, midi, prev_dur)   # exactly once per note: advances the GRU state
        if self.notes and prev_dur is not None:
            self.notes[-1][1] = self.notes[-1][0] + prev_dur
        k = int(onset)
        self._finish_beats(k)
        self.notes.append([onset, None, (midi - self.t) % 12])
        b = self.belief(k)                                     # r2_mix_v2: b_HMM^0.3 * b_GRU^0.7, normalized
        pfit = b @ CTM                                         # P(pc is a chord tone), relative pcs
        hp = (midi - self.t) % 12
        robust = self.mode != 'off' and self.use_e5(hp)
        if robust:
            # joint over (chord z, her TRUE pc t) given what was heard: P(t | z, heard) ~ P(t | z) * P(heard | t)
            pt = self.m.Bc * self.m.N[:, hp][None, :]          # [z, t]
            pt /= pt.sum(1, keepdims=True)
            jz = b[:, None] * pt                               # [z, t]
            lo = self.true_lo
            tm = np.array([midi + ((t - hp - lo) % 12) + lo for t in range(12)])   # true midi for each t, near the heard one
        ref = self.prev if self.prev is not None else ALTO_MID
        hi = min(ALTO[1], midi - 1)                            # below her heard note
        best, bs = -1, -1e9
        for h in range(ALTO[0], hi + 1):
            iv = abs(midi - h) % 12
            if robust:
                cv = np.fromiter(((abs(x - h) % 12) in CONSONANT for x in tm), float, 12)
                s = float(CTM[:, (h - self.t) % 12] @ (jz @ cv))
                if s <= 0:
                    continue
            else:
                if iv not in CONSONANT:
                    continue
                s = pfit[(h - self.t) % 12]
            if iv == 0:
                s -= W_OCT if h != midi else W_UNIS
            d = abs(h - ref) if self.prev is not None else 0
            s -= W_STEP * d + W_BIG * (d > 7) + W_MID * abs(h - ALTO_MID)
            if self.prev is not None and self.prev_m is not None:
                dm, dh = midi - self.prev_m, h - self.prev
                i0 = abs(self.prev_m - self.prev) % 12
                if dm != 0 and dh != 0 and (dm > 0) == (dh > 0) and iv in (0, 7) and i0 == iv:
                    s -= W_PAR
            if s > bs:
                best, bs = h, s
        if best >= 0:
            self.prev = best
        self.prev_m = midi
        return best


def make(cls, train):
    """r2_mix_v2's build with this file's class: m1's model and the 3-GRU average (full 96, big 128, s1 96)."""
    torch.set_num_threads(1)
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0, dict(V3, step=0.01))
    return cls(m1_filter_v5lo.build(train).m, ens)


def build(train):
    return make(TrueCons, train)
