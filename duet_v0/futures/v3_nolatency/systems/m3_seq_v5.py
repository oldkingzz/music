"""m3_seq_v5: ensemble of three trained GRUs (96 'full', 128 'big', 96 seed-1 's1'); averaged chord
probabilities, then the val-tuned voice-leading choice of m3_seq_v3 but step cost 0.01 (re-tuned on val for the ensemble). DUET_HIST respected by all GRUs."""
import os

import torch

from systems.m3_seq_v4 import V3, Ens, _load


def build(train):
    torch.set_num_threads(1)
    hist = float(os.environ.get('DUET_HIST', '-1'))
    return Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], hist, dict(V3, step=0.01))
