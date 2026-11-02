"""r2_mix_v4 (v4 round 2, idea E2, a = 0.85, leap penalty W_BIG 0.8): r2_mix_v3's mixed chord belief b_HMM^(1-a) * b_GRU^a,
with its own copy of m1_filter_v5lo's chooser (its 6 weights as file constants; only W_BIG changed, 0.4 -> 0.8)."""
import numpy as np
import torch

from systems import m1_filter_v5lo
from systems.m3_seq_v4 import V3, Ens, _load
from theory import ALTO, ALTO_MID, CONSONANT

ALPHA = 0.85     # weight of the GRU belief in the log-linear mix (0 = m1_filter_v5lo, 1 = r1_gru_chooser_v1)
FLOOR = 1e-300   # keeps the logs finite if a float32 GRU probability underflows to 0
W_STEP = 0.01    # per semitone Duet moves
W_BIG = 0.8      # extra for a leap of more than 7 semitones (m1_filter_v5lo: 0.4)
W_PAR = 0.6      # parallel 5th / octave with her
W_MID = 0.004    # per semitone away from the middle of the alto range
W_OCT = 0.3      # an octave from her note
W_UNIS = 1.0     # unison with her (inert: Duet always sings below her)
CTM = m1_filter_v5lo.CTM


class Mix(m1_filter_v5lo.Filter):
    """m1's model and real-time filter unchanged; belief(k) = normalized b_HMM(k)^(1-a) * b_GRU^a; m1's chooser, own weights."""

    def __init__(self, model, ens):
        super().__init__(model)
        self.ens = ens
        self.a = ALPHA
        self.w_step, self.w_big, self.w_par, self.w_mid, self.w_oct, self.w_unis = W_STEP, W_BIG, W_PAR, W_MID, W_OCT, W_UNIS
        self._p = None

    def start(self, meta):
        super().start(meta)
        self.ens.start(meta)
        self._p = None

    def belief(self, k):
        bh = np.maximum(np.asarray(super().belief(k), dtype=np.float64), FLOOR)   # m1's HMM, current note included
        bg = np.maximum(np.asarray(self._p, dtype=np.float64), FLOOR)             # 3-GRU average at her current note
        lg = (1.0 - self.a) * np.log(bh) + self.a * np.log(bg)
        w = np.exp(lg - lg.max())
        return w / w.sum()

    def step(self, i, onset, midi, prev_dur):
        """Copy of m1_filter_v5lo.Filter.step on its robust = 0, look = 0, below = 1, order2 = 0 path."""
        self._p = self.ens.chord_probs(onset, midi, prev_dur)   # exactly once per note: advances the GRU state
        if self.notes and prev_dur is not None:
            self.notes[-1][1] = self.notes[-1][0] + prev_dur
        k = int(onset)
        self._finish_beats(k)
        self.notes.append([onset, None, (midi - self.t) % 12])
        b = self.belief(k)
        pfit = b @ CTM                                        # P(pc is a chord tone), relative pcs
        ref = self.prev if self.prev is not None else ALTO_MID
        hi = min(ALTO[1], midi - 1)                           # below = 1: only notes under hers
        best, bs = -1, -1e9
        for h in range(ALTO[0], hi + 1):
            iv = abs(midi - h) % 12
            if iv not in CONSONANT:
                continue
            s = pfit[(h - self.t) % 12]
            if iv == 0:
                s -= self.w_oct if h != midi else self.w_unis
            d = abs(h - ref) if self.prev is not None else 0
            s -= self.w_step * d + self.w_big * (d > 7) + self.w_mid * abs(h - ALTO_MID)
            if self.prev is not None and self.prev_m is not None:
                dm, dh = midi - self.prev_m, h - self.prev
                i0 = abs(self.prev_m - self.prev) % 12
                if dm != 0 and dh != 0 and (dm > 0) == (dh > 0) and iv in (0, 7) and i0 == iv:
                    s -= self.w_par
            if s > bs:
                best, bs = h, s
        if best >= 0:
            self.prev = best
        self.prev_m = midi
        return best


def build(train):
    torch.set_num_threads(1)
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0, dict(V3, step=0.01))
    return Mix(m1_filter_v5lo.build(train).m, ens)
