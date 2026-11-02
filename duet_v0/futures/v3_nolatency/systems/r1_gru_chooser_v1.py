"""r1_gru_chooser_v1 (v4 round 1, idea E1): m1_filter_v5lo's no-copy chooser, but the real-time chord guess comes
from m3_seq_v5's three GRUs (full 96, big 128, s1 96; chord probabilities averaged) instead of the HMM filter."""
import torch

from systems import m1_filter_v5lo
from systems.m3_seq_v4 import V3, Ens, _load


class G(m1_filter_v5lo.Filter):
    """m1's chooser unchanged; belief(k) = the GRU ensemble's chord probabilities at her current note."""

    def __init__(self, model, ens):
        super().__init__(model)
        self.ens = ens
        self._p = None

    def start(self, meta):
        super().start(meta)
        self.ens.start(meta)
        self._p = None

    def step(self, i, onset, midi, prev_dur):
        self._p = self.ens.chord_probs(onset, midi, prev_dur)   # exactly once per note: advances the GRU state
        return super().step(i, onset, midi, prev_dur)

    def belief(self, k):
        return self._p


def build(train):
    torch.set_num_threads(1)
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0, dict(V3, step=0.01))
    return G(m1_filter_v5lo.build(train).m, ens)
