"""r2_mix_v1 (v4 round 2, idea E2, a = 0.5): m1_filter_v5lo's chooser unchanged, on the chord belief b(k) ~ b_HMM(k)^(1-a) * b_GRU^a,
the normalized log-linear mix of m1's own real-time HMM filter and r1_gru_chooser_v1's 3-GRU average at her current note."""
import numpy as np
import torch

from systems import m1_filter_v5lo
from systems.m3_seq_v4 import V3, Ens, _load

ALPHA = 0.5      # weight of the GRU belief in the log-linear mix (0 = m1_filter_v5lo, 1 = r1_gru_chooser_v1)
FLOOR = 1e-300   # keeps the logs finite if a float32 GRU probability underflows to 0


class Mix(m1_filter_v5lo.Filter):
    """m1's model, real-time filter and chooser unchanged; belief(k) = normalized b_HMM(k)^(1-a) * b_GRU^a."""

    def __init__(self, model, ens):
        super().__init__(model)
        self.ens = ens
        self.a = ALPHA
        self._p = None

    def start(self, meta):
        super().start(meta)
        self.ens.start(meta)
        self._p = None

    def step(self, i, onset, midi, prev_dur):
        self._p = self.ens.chord_probs(onset, midi, prev_dur)   # exactly once per note: advances the GRU state
        return super().step(i, onset, midi, prev_dur)          # m1's chooser; it calls belief(k) once

    def belief(self, k):
        bh = np.maximum(np.asarray(super().belief(k), dtype=np.float64), FLOOR)   # m1's HMM, current note included
        bg = np.maximum(np.asarray(self._p, dtype=np.float64), FLOOR)             # 3-GRU average at her current note
        lg = (1.0 - self.a) * np.log(bh) + self.a * np.log(bg)
        w = np.exp(lg - lg.max())
        return w / w.sum()


def build(train):
    torch.set_num_threads(1)
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0, dict(V3, step=0.01))
    return Mix(m1_filter_v5lo.build(train).m, ens)
