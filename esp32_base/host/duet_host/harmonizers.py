"""Python reference of every plug-in the board has (firmware/libraries/DuetCore/src/duet_harmonizer.cpp).

Same interface as the systems scored by bench3 (duet_v0/futures/v3_nolatency/bench3.py):

    obj.start(meta)                           meta = {'tonic', 'M', 'K', 'pos'}
    obj.step(i, onset, midi, prev_dur) -> Duet's MIDI note, or -1 = silent

A plug-in id on the board and the entry with the same id here must give the same notes. checks/check_notes.py
compares them note by note.
"""
from __future__ import annotations

MAJOR = (0, 2, 4, 5, 7, 9, 11)   # scale degrees, semitones above the tonic (duet_beat/theory.py)
ALTO = (53, 77)                  # F3..F5, the range Duet may sing in (duet_beat/theory.py)


class Echo:
    """Answers her own note. It tests the link and the note path, not music."""
    name = "echo"

    def start(self, meta):
        pass

    def step(self, i, onset, midi, prev_dur):
        return midi


class ThirdBelow:
    """Two scale steps below her note, kept inside the alto range. A copy of
    duet_v0/futures/v3_nolatency/systems/third_below.py (bench3 score 0.505)."""
    name = "third_below"

    def start(self, meta):
        self.t = meta["tonic"]

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


# id -> class. The ids are the wire ids (duet_harmonizer.h: H_ECHO = 0, H_THIRD_BELOW = 1).
PLUGINS = {0: Echo, 1: ThirdBelow}
ID_BY_NAME = {cls.name: k for k, cls in PLUGINS.items()}


def make(plugin):
    """plugin: id or name."""
    pid = ID_BY_NAME[plugin] if isinstance(plugin, str) else int(plugin)
    return PLUGINS[pid]()
