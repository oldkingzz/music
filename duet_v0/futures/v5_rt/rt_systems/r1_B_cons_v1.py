"""r1_B_cons_v1 (round 1, idea B, run v5rt_1101): conservative look-ahead around reactive r2_mix_v2, built by
rt_systems.r1_B_lib_v1 with the constants below (see that file's docstring for the rules). Tuned on val only (no test
split), offline replay runs_mini/v5rt_1101/r1/B/off_d6_g2_b0a4*.json: the setting with the best worst-case rt_nd gain
over both mishearing seeds among those keeping fresh clashes at or below reactive's with a margin.

Plan points: move early only to a note whose attack risk (P(2nd / tritone / 7th) against her note at q) is <= 8 %, whose
P(hit) is >= 0.5, below her sounding note, and only when it beats keeping the sounding note after a risk penalty
(MU 2) and a move cost (KAP 0.05); never fade to silence. Hear points: r2_mix_v2's choice, except for a probably
misheard note (P(right) < 0.7), where the posterior-best of keep / silence / any alto note may replace it.
"""
from rt_systems import r1_B_lib_v1 as LIB

BETA = 0.0             # next pitch independent of the chord forecast (P(z | n) ~ b(z) Bc[z, n] only)
ADAPT = 4.0            # onset prior adapts to the song with 4 pseudo-counts
PREV_ACTUAL = False    # the filter's voice-leading memory follows r2_mix_v2's own choice
RULE = dict(
    LW=0.1, WH=0.4,                    # window weights: onset case (her first L) and hold case (until next decision)
    LAM=1.0,                           # value = P(hit) - LAM * P(harsh)
    MU=2.0, KAP=0.05,                  # move penalty: MU * attack risk + KAP
    RHO=0.08, TAU=0.5, PMIN=0.0,       # move only if attack risk <= RHO and P(hit) >= TAU (and p_on >= PMIN)
    BELOW=1, WS=0.0,                   # moves stay below her sounding note; no extra distance cost
    SIL=0, KS=0.0,                     # no fading to silence at plan points
    SUSP=0.7, LAMH=1.0, MUH=1.0,       # hear: robust choice when P(heard right) < SUSP
    KSH=0.0, SILH=1, ROBH=1, MARGH=0.0,
    PREV_ACTUAL=PREV_ACTUAL)


def build(train):
    return LIB.build(train, RULE, BETA, ADAPT)
