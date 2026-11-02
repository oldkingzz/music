"""m3_seq_v4: ensemble of the two trained GRUs (96 units 'full' + 128 units 'big'); chord probabilities are
averaged, then the same val-tuned voice-leading choice as m3_seq_v3. DUET_HIST respected by both GRUs."""
import os

import numpy as np
import torch

from systems import m3_seq_v1 as S

V3 = dict(dbl=0.2, step=0.015, big=0.1, par=0.3, above=0.0)


class Ens(S.Duet):
    def __init__(self, nets, hist, cfg):
        super().__init__(None, hist, cfg)
        self.subs = [S.Duet(n, hist, cfg) for n in nets]

    def start(self, meta):
        super().start(meta)
        for d in self.subs:
            d.start(meta)

    def chord_probs(self, onset, midi, prev_dur):
        return np.mean([d.chord_probs(onset, midi, prev_dur) for d in self.subs], axis=0)


def _load(tag, hid):
    net = S.Net(hid)
    net.load_state_dict(torch.load(os.path.join(S.HERE, f'm3_seq_{tag}.pt')))
    net.eval()
    return net


def build(train):
    torch.set_num_threads(1)
    hist = float(os.environ.get('DUET_HIST', '-1'))
    return Ens([_load('full', 96), _load('big', 128)], hist, dict(V3))
