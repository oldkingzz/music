"""r3_style_mix_v4 (v4 round 3, idea E2 x E3, a = 0.85): r2_mix_v2's log-linear mix with r2_style_hmm_v1's style-aware HMM
(pooled + pop + pop909) in place of m1's pooled HMM, b(k) ~ b_style(k)^(1-a) * b_GRU^a, then m1_filter_v5lo's chooser."""
import numpy as np
import torch

from systems import r2_style_hmm_v1 as sty
from systems.m3_seq_v4 import V3, Ens, _load

ALPHA = 0.85     # weight of the GRU belief in the log-linear mix (0 = r2_style_hmm_v1, 1 = r1_gru_chooser_v1)
FLOOR = 1e-300   # keeps the logs finite if a float32 GRU probability underflows to 0


class StyleMix(sty.StyleFilter):
    """Style-aware HMM filter and m1's chooser unchanged; belief(k) = normalized b_style(k)^(1-a) * b_GRU^a."""

    def __init__(self, models, lams, rh, mprior, ens):
        super().__init__(models, lams, rh, mprior)
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
        bh = np.maximum(np.asarray(super().belief(k), dtype=np.float64), FLOOR)   # style HMM, current note included
        bg = np.maximum(np.asarray(self._p, dtype=np.float64), FLOOR)             # 3-GRU average at her current note
        lg = (1.0 - self.a) * np.log(bh) + self.a * np.log(bg)
        w = np.exp(lg - lg.max())
        return w / w.sum()


def build(train):
    torch.set_num_threads(1)
    base = sty.build(train)
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0, dict(V3, step=0.01))
    return StyleMix(base.models, base.lams, base.rh, base.mprior, ens)
