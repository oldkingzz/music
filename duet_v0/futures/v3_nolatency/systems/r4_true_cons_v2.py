"""r4_true_cons_v2 (v4 round 4, idea E5, belief + choice side): r4_true_cons_v1's belief side, plus r3_true_cons_v4's choice side
(expected consonance with her TRUE note on every note except a recurring out-of-scale pitch class, which is trusted)."""
from systems.r4_true_cons_v1 import TrueConsBelief, make

BELIEF = 'cur'             # a suspect heard note leaves its own chord belief alone (HMM without it, GRU = last trusted guess)
GRU_SUSPECT = 'prev'
CHOICE = 'v4'              # r3_true_cons_v4's scoring


class TrueConsBoth(TrueConsBelief):
    """TrueConsBelief with E5 on both sides."""
    belief_side = BELIEF
    gru_suspect = GRU_SUSPECT
    choice_side = CHOICE


def build(train):
    return make(TrueConsBoth, train)
