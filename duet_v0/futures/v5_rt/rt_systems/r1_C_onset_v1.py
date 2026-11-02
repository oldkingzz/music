"""r1_C_onset_v1 (run v5rt_1101, round 1, idea C, variant 1: the plain learned joint forecaster at her onsets).

r2_mix_v2 (reactive at hear) + the learned joint forecaster r1_C_fnet_v1 (causal GRU over her heard notes, trained on
train only: runs_mini/v5rt_1101/models/C_fnet_v1s.pt) used at every plan point q (r1_C_core_v3.plan, onset path):
P(a note of hers starts at q) and the joint P(chord, pc) of that note, its chord marginal replaced by r2_mix_v2's own
forecast for beat int(q) (W_R2 = 1); Duet attacks the alto note with the best expected hit minus LAM * expected harsh
(also against her held note if no new note comes) when P(onset) >= 0.7, its expected harsh <= 0.05 and it beats
keeping the sounding note by 0.02; at hear, the planned note stays if consonant, below her and within KEEP_MARGIN of
the reactive choice's P(chord tone), else r2_mix_v2's step(). No re-voicing, no wary ear. Settings tuned on val.
"""
from rt_systems import r1_C_core_v3 as core

CKPT = 'C_fnet_v1s.pt'   # forecaster weights (train only), runs_mini/v5rt_1101/models/
Q_MIN = 1.0              # no plans before beat 1
T_PC = 1.0               # temper P(pc | chord): power 1 / T_PC (1 = as trained)
EPS_PC = 0.0             # floor P(pc | chord) with this share of uniform
W_R2 = 1.0               # chord marginal at an onset plan = r2_mix_v2's forecast (log-linear weight)
ATT_ON = 1               # onset attacks on
P_ON_MIN = 0.7           # attack only if P(a note starts at q) >= this
LAM = 4.0                # expected-harsh penalty of an attack
LAM_KEEP = 1.0           # expected-harsh penalty of keeping the sounding note
HARSH_MAX = 0.05         # never attack a note whose expected harsh exceeds this
GAIN_MIN = 0.02          # an attack must beat keeping the sounding note by this
W_ABOVE = 0.5            # penalty * P(Duet at or above her next note)
W_STEP = 0.005           # voice-leading cost per semitone from the sounding note
HOLD = 1                 # count the case "she keeps holding her heard note" with weight 1 - P(onset)
KEEP_MARGIN = 0.1        # hear: keep the planned note if its P(chord tone) >= that of the reactive choice - this
REV = 0                  # held-note re-voicing off
REV_PON_MAX = 0.7
REV_GAIN = 0.05
REV_HARSH_MAX = 0.08
REV_HALF = 0
W_REV_FC = 0.0
SUS_POST = 0.0           # wary ear off

CFG = dict(CKPT=CKPT, Q_MIN=Q_MIN, T_PC=T_PC, EPS_PC=EPS_PC, W_R2=W_R2, ATT_ON=ATT_ON, P_ON_MIN=P_ON_MIN, LAM=LAM,
           LAM_KEEP=LAM_KEEP, HARSH_MAX=HARSH_MAX, GAIN_MIN=GAIN_MIN, W_ABOVE=W_ABOVE, W_STEP=W_STEP, HOLD=HOLD,
           KEEP_MARGIN=KEEP_MARGIN, REV=REV, REV_PON_MAX=REV_PON_MAX, REV_GAIN=REV_GAIN, REV_HARSH_MAX=REV_HARSH_MAX,
           REV_HALF=REV_HALF, W_REV_FC=W_REV_FC, SUS_POST=SUS_POST)


def build(train):
    return core.build_with(train, CFG)
