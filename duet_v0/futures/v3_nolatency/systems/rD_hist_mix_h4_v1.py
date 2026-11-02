"""rD_hist_mix_h4_v1 (Phase D history ablation): r2_mix_v2 with both chord guessers limited to her last 4 beats (HMM H = 4, GRU hist = 4)."""
import torch

from systems import m1_filter_v5lo
from systems.m3_seq_v4 import V3, Ens, _load
from systems.r2_mix_v2 import Mix

H = 4   # beats of her past melody both chord guessers may see (r2_mix_v2 itself = the whole song so far)


def build(train):
    torch.set_num_threads(1)
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], float(H), dict(V3, step=0.01))
    mix = Mix(m1_filter_v5lo.build(train).m, ens)
    mix.H = H   # m1_filter_v5lo.Filter.belief then reruns the HMM filter over only the last H beats (what DUET_HIST did)
    return mix
