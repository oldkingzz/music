"""r2_gru_chooser_v1 (v4 round 2, E1 attempt 2): r1_gru_chooser_v1 (3-GRU average chord guess + m1_filter_v5lo's chooser,
copied in here) with one chooser weight value changed: the leap penalty W_BIG 0.4 -> 0.6."""
import torch

from systems import m1_filter_v5lo
from systems.m3_seq_v4 import V3, Ens, _load
from theory import ALTO, ALTO_MID, CONSONANT

# m1_filter_v5lo's six chooser weights (its P['w_*']); only W_BIG differs from m1 (0.4 there)
W_STEP = 0.01     # cost per semitone that Duet's line moves
W_BIG = 0.6       # extra cost of a leap of more than 7 semitones
W_PAR = 0.6       # cost of a parallel 5th/8ve (both voices move the same way into the same interval)
W_MID = 0.004     # cost per semitone away from the middle of the alto range
W_OCT = 0.3       # cost of an octave with her note
W_UNIS = 1.0      # cost of a unison with her note (inert: Duet stays strictly below her)


class G(m1_filter_v5lo.Filter):
    """belief = the GRU ensemble's chord probabilities at her current note; m1_filter_v5lo's chooser (its code path:
    robust 0, look 0, below 1, order2 0) with the weights above; m1's bookkeeping kept as in m1's step."""
    w_step, w_big, w_par, w_mid, w_oct, w_unis = W_STEP, W_BIG, W_PAR, W_MID, W_OCT, W_UNIS

    def __init__(self, model, ens):
        super().__init__(model)
        self.ens = ens
        self._p = None

    def start(self, meta):
        super().start(meta)
        self.ens.start(meta)
        self._p = None

    def belief(self, k):
        return self._p

    def step(self, i, onset, midi, prev_dur):
        self._p = self.ens.chord_probs(onset, midi, prev_dur)   # exactly once per note: advances the GRU state
        if self.notes and prev_dur is not None:
            self.notes[-1][1] = self.notes[-1][0] + prev_dur
        k = int(onset)
        self._finish_beats(k)
        self.notes.append([onset, None, (midi - self.t) % 12])
        b = self.belief(k)
        pfit = b @ m1_filter_v5lo.CTM                          # P(pc is a chord tone), relative pcs
        ref = self.prev if self.prev is not None else ALTO_MID
        hi = min(ALTO[1], midi - 1)                            # below = 1: strictly under her heard note
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
    return G(m1_filter_v5lo.build(train).m, ens)
