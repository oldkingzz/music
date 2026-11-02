"""r1_C_revoice_v1 (run v5rt_1101, round 1, idea C, variant 2: forecaster-gated re-voicing of held notes + wary ear).

r2_mix_v2 answers every note reactively at hear (its step(), once per note). The learned joint forecaster
r1_C_fnet_v1 (causal GRU over her heard notes, trained on train only: runs_mini/v5rt_1101/models/C_fnet_v1s.pt) is used
(r1_C_core_v3) for two things, never to guess her next note blind:
- Re-voicing a HELD note at a beat boundary q = k (or at k + 0.5 if Duet did not move at k): when her last heard note
  started before beat k and its end is not heard yet, and the forecaster says a new note at q is not likely
  (P(onset at q) <= 0.7), Duet moves to the alto note below her held note, consonant with it (not unison/octave), with
  the highest P(chord tone of beat k) under r2_mix_v2's belief for beat k (its HMM with her held note as evidence, its
  GRU belief pushed on by the HMM table), if that beats the sounding note by >= 0.05 and its risk
  P(onset) * P(harsh vs a new note, forecast) + (1 - P(onset)) * P(harsh vs her TRUE held note, posterior from the
  forecaster's prior and the 10 % mishearing model) <= 0.08.
- Wary ear: at hear, when the posterior P(her true pc = the heard pc) < 0.3 and Duet is sounding a note, Duet does not
  attack r2_mix_v2's new choice unless it has the lower posterior P(harsh).
No onset attacks (they raised fresh clashes, see r1_C_onset_v1). Settings tuned on val.
"""
from rt_systems import r1_C_core_v3 as core

CKPT = 'C_fnet_v1s.pt'   # forecaster weights (train only), runs_mini/v5rt_1101/models/
Q_MIN = 1.0              # no plans before beat 1
T_PC = 1.5               # temper P(pc | chord): power 1 / T_PC (the trained net is over-confident)
EPS_PC = 0.05            # floor P(pc | chord) with this share of uniform
W_R2 = 1.0               # (onset path only, unused here)
ATT_ON = 0               # onset attacks off
P_ON_MIN = 0.7
LAM = 4.0
LAM_KEEP = 1.0
HARSH_MAX = 0.05
GAIN_MIN = 0.02
W_ABOVE = 0.5
W_STEP = 0.005           # voice-leading cost per semitone from the sounding note (re-voicing too)
HOLD = 1
KEEP_MARGIN = 0.1
REV = 1                  # held-note re-voicing on
REV_PON_MAX = 0.7        # re-voice only if P(a new note starts at q) <= this
REV_GAIN = 0.05          # ... and P(chord tone) gains at least this over the sounding note
REV_HARSH_MAX = 0.08     # ... and its risk of a harsh interval is at most this
REV_HALF = 1             # also at the half beat k + 0.5
W_REV_FC = 0.0           # re-voicing chord belief = r2_mix_v2's only
SUS_POST = 0.3           # wary ear: P(true pc = heard) below this -> no new attack unless it is the safer note

CFG = dict(CKPT=CKPT, Q_MIN=Q_MIN, T_PC=T_PC, EPS_PC=EPS_PC, W_R2=W_R2, ATT_ON=ATT_ON, P_ON_MIN=P_ON_MIN, LAM=LAM,
           LAM_KEEP=LAM_KEEP, HARSH_MAX=HARSH_MAX, GAIN_MIN=GAIN_MIN, W_ABOVE=W_ABOVE, W_STEP=W_STEP, HOLD=HOLD,
           KEEP_MARGIN=KEEP_MARGIN, REV=REV, REV_PON_MAX=REV_PON_MAX, REV_GAIN=REV_GAIN, REV_HARSH_MAX=REV_HARSH_MAX,
           REV_HALF=REV_HALF, W_REV_FC=W_REV_FC, SUS_POST=SUS_POST)


def build(train):
    return core.build_with(train, CFG)
