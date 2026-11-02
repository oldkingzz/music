"""r3_cover_fit_v4 (v4 round 3, idea E4): the beats her note will cover are forecast for the HMM part only (the GRUs were
trained on the chords of the covered beats already), then mixed with the GRU as in r2_mix_v2; lengths: her last 16 notes."""
import torch

from systems import m1_filter_v5lo
from systems.m3_seq_v4 import V3, Ens, _load
from systems.r3_cover_fit_v1 import Cover

# m1_filter_v5lo's six chooser weights (its P['w_*'] values), all unchanged
W_STEP = 0.01     # cost per semitone that Duet's line moves
W_BIG = 0.4       # extra cost of a leap of more than 7 semitones
W_PAR = 0.6       # cost of a parallel 5th/8ve (both voices move the same way into the same interval)
W_MID = 0.004     # cost per semitone away from the middle of the alto range
W_OCT = 0.3       # cost of an octave with her note
W_UNIS = 1.0      # cost of a unison with her note (inert: Duet stays strictly below her)

# E4 settings
AHEAD = 8         # beats after her onset beat that are forecast with m1's chord-change tables
NDUR = 16         # her last NDUR finished note lengths, each counted equally, form the length estimate
LENGTH = 'emp'    # empirical distribution of those lengths (not their median)
SIDE = 'hmm'      # forecast only m1's HMM belief, then the log-linear mix with the GRU (a = 0.7 as in r2_mix_v2)
EVID = 0          # her held note is not reused as evidence on the forecast beats


class CoverHmm(Cover):
    """r3_cover_fit_v1's class with the constants above."""
    w_step, w_big, w_par, w_mid, w_oct, w_unis = W_STEP, W_BIG, W_PAR, W_MID, W_OCT, W_UNIS
    ahead, ndur, length, side, evid = AHEAD, NDUR, LENGTH, SIDE, EVID


def build(train):
    torch.set_num_threads(1)
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0, dict(V3, step=0.01))
    return CoverHmm(m1_filter_v5lo.build(train).m, ens)
