"""r1_B_cons_v2 (round 1, idea B, run v5rt_1101) = r1_B_cons_v1 with PREV_ACTUAL = True: after every hear point
r2_mix_v2's voice-leading memory (m1's prev) is set to the note Duet really sounds (an early-moved note it kept, or the
hear rule's replacement for a probably misheard note) instead of r2_mix_v2's own choice. Everything else (priors, probe,
plan and hear rules, constants) is r1_B_cons_v1's; built by rt_systems.r1_B_lib_v1. Val only, tuned on val.
"""
from rt_systems import r1_B_lib_v1 as LIB

BETA = 0.0
ADAPT = 4.0
PREV_ACTUAL = True     # m1's voice-leading memory follows what Duet sings
RULE = dict(
    LW=0.1, WH=0.4,
    LAM=1.0,
    MU=2.0, KAP=0.05,
    RHO=0.08, TAU=0.5, PMIN=0.0,
    BELOW=1, WS=0.0,
    SIL=0, KS=0.0,
    SUSP=0.7, LAMH=1.0, MUH=1.0,
    KSH=0.0, SILH=1, ROBH=1, MARGH=0.0,
    PREV_ACTUAL=PREV_ACTUAL)


def build(train):
    return LIB.build(train, RULE, BETA, ADAPT)
