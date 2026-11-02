"""r3_true_cons_v4 (v4 round 3, idea E5): r3_true_cons_v1 on every note, except that a heard pitch class outside the major scale which
she has already been heard singing on >= 2 and >= 3 % of her earlier notes in this song is trusted (the champion's hard rule)."""
from systems.r3_true_cons_v1 import TrueCons, make
from theory import MAJOR

MODE = 'all_but_recurring'   # E5 on every note except a recurring out-of-scale heard pitch class
MIN_COUNT = 2                # earlier heard notes with this pitch class needed before Duet trusts it ...
MIN_SHARE = 0.03             # ... and their share of her earlier notes (a mishearing hits one given pc on ~0.9 % of notes)


class TrueConsRecurring(TrueCons):
    """TrueCons that trusts its ear for an out-of-scale pitch class she keeps singing (a real chromatic note recurs)."""
    mode = MODE

    def use_e5(self, hp):
        if hp in MAJOR:
            return True
        earlier = [n[2] for n in self.notes[:-1]]          # relative pcs of her earlier heard notes in this song
        c = earlier.count(hp)
        return not (c >= MIN_COUNT and c >= MIN_SHARE * len(earlier))


def build(train):
    return make(TrueConsRecurring, train)
