"""r3_cover_fit_v3 (v4 round 3, idea E4): r3_cover_fit_v1 (expected chord-tone share over the beats her note will cover) with
the length estimated from her last 32 finished notes that started like this one: on the beat, or off it (from her last 128)."""
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
NDUR = 32         # at most NDUR lengths of her finished notes that started like the current one (on / off the beat)
WINDOW = 128      # ... taken from her last WINDOW finished notes; fewer than 2 such notes -> her last NDUR notes of any kind
LENGTH = 'emp'    # empirical distribution of those lengths (not their median)
SIDE = 'mix'      # forecast the champion's mixed belief
EVID = 0          # her held note is not reused as evidence on the forecast beats
FIRST_LEN = 1.0   # length (beats) assumed before she has finished any note


class CoverPhase(Cover):
    """r3_cover_fit_v1's class with the constants above and the on/off-beat length estimate."""
    w_step, w_big, w_par, w_mid, w_oct, w_unis = W_STEP, W_BIG, W_PAR, W_MID, W_OCT, W_UNIS
    ahead, ndur, length, side, evid = AHEAD, NDUR, LENGTH, SIDE, EVID
    window = WINDOW

    def _lengths(self):
        cur = self.notes[-1][0]                                   # her current note's onset (in beats)
        off_beat = (cur - int(cur)) > 1e-6
        prev = [n for n in self.notes[-(self.window + 1):-1] if n[1] is not None]
        same = [n[1] - n[0] for n in prev if ((n[0] - int(n[0])) > 1e-6) == off_beat][-self.ndur:]
        if len(same) >= 2:
            return same
        durs = [n[1] - n[0] for n in prev][-self.ndur:]
        return durs or [FIRST_LEN]


def build(train):
    torch.set_num_threads(1)
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0, dict(V3, step=0.01))
    return CoverPhase(m1_filter_v5lo.build(train).m, ens)
