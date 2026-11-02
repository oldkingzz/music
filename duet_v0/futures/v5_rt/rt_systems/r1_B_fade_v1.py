"""r1_B_fade_v1 (round 1, idea B, run v5rt_1101) = r1_B_cons_v1 plus fading to silence at plan points: when the note
Duet is sounding has a negative expected value over the coming window (P(hit) - LAM * P(harsh) against her likely
next note, or against her held note if she holds it) and silence is better by more than KS = 0.02, Duet stops at the
plan point and enters at the next hear point with r2_mix_v2's choice (a late but clean entry). Trades rt_nd for much less
harsh time. Everything else is r1_B_cons_v1's; built by rt_systems.r1_B_lib_v1. Val only, tuned on val (offline replay
runs_mini/v5rt_1101/r1/B/off_d5_ablate_b0a4.json).
"""
from rt_systems import r1_B_lib_v1 as LIB

BETA = 0.0
ADAPT = 4.0
PREV_ACTUAL = False
RULE = dict(
    LW=0.1, WH=0.4,
    LAM=1.0,
    MU=2.0, KAP=0.05,
    RHO=0.08, TAU=0.5, PMIN=0.0,
    BELOW=1, WS=0.0,
    SIL=1, KS=0.02,                    # fade to silence when that beats keeping the sounding note by KS
    SUSP=0.7, LAMH=1.0, MUH=1.0,
    KSH=0.0, SILH=1, ROBH=1, MARGH=0.0,
    PREV_ACTUAL=PREV_ACTUAL)


def build(train):
    return LIB.build(train, RULE, BETA, ADAPT)
