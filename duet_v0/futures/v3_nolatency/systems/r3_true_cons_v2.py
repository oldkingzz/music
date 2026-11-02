"""r3_true_cons_v2 (v4 round 3, idea E5): r3_true_cons_v1's expected consonance with her TRUE note, used only when the heard
note is outside the major scale (where a mishearing is most likely); on in-scale heard notes the champion's hard rule, unchanged."""
from systems.r3_true_cons_v1 import TrueCons, make
from theory import MAJOR

MODE = 'out_of_scale'      # E5 only when the heard pitch class (relative to the tonic) is not one of the 7 scale notes


class TrueConsOutOfScale(TrueCons):
    """TrueCons with E5 switched on only for heard notes outside the major scale."""
    mode = MODE

    def use_e5(self, hp):
        return hp not in MAJOR


def build(train):
    return make(TrueConsOutOfScale, train)
