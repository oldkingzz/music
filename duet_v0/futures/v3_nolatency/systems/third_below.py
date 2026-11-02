"""Baseline, no history: the diatonic third below her current note (the live harmonizer's rule)."""
from theory import ALTO, MAJOR


class ThirdBelow:
    def start(self, meta):
        self.t = meta['tonic']

    def step(self, i, onset, midi, prev_dur):
        pc = (midi - self.t) % 12
        if pc in MAJOR:                                  # two scale steps down
            j = MAJOR.index(pc)
            h = midi - ((pc - MAJOR[(j - 2) % 7]) % 12)
        else:
            h = midi - 3
        while h > ALTO[1]:
            h -= 12
        while h < ALTO[0]:
            h += 12
        return h


def build(train):
    return ThirdBelow()
