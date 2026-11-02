"""Choir rehearsal engine (run choir_1102): she sings the melody of a KNOWN song after Duet's 4-click count-in, Duet
sings the other three parts (alto, tenor, bass) of a fixed four-part arrangement (choir_songs_v3.json, written by
runs_mini/choir_1102/arrange_v3.py).

Because every part is known in advance, nothing is guessed and nothing waits for her: each part's note is written
Lo + PRE early so it sounds exactly on the beat of the click grid, whatever the latency (only the click grid matters,
the same way RTDuet's planned notes stay aligned with her). Her voice is still tracked (RTDuet's tracker and segmenter,
with a system that never sings) only to show what was heard on the page; it does not move the parts.
The arrangement is transposed to the key chosen on the page (nearest direction).

  ChoirDuet(arr, key_pc, bpm, click_all=False)   arr = choir_songs_v3.json['songs'][id]
"""
import json
import os

import numpy as np

import live_rt as LR
from live_bench3 import ReedVoice
from dsp import HOP, SR

HERE = os.path.dirname(os.path.abspath(__file__))
SONGS_FILE = os.path.join(HERE, 'choir_songs_v3.json')


def load_songs():
    with open(SONGS_FILE) as fh:
        return json.load(fh)['songs']


class _Listen:
    """A bench_rt-style system that never sings (plan / hear return None): the parts come from the score."""

    def start(self, meta):
        pass

    def plan(self, q, heard):
        return None

    def hear(self, i, onset, midi, prev_dur):
        return None


class ChoirDuet(LR.RTDuet):
    LEVEL = 0.2            # each part's peak; three parts stay under the page's clip
    GAP = 0.03             # s of silence before a part re-sings (a re-attack, not a slur)
    AHEAD = 0.3            # s: parts are written this far ahead of the output

    def __init__(self, arr, key_pc, bpm, click_all=False, log=lambda s: None):
        shift = (int(key_pc) - int(arr['tonic'])) % 12
        if shift > 6:
            shift -= 12
        super().__init__(_Listen(), 'choir:' + arr['id'], int(key_pc), arr['M'], arr['K'], bpm, count_in=4,
                         click_all=click_all, lead=0.5, log=log, listen_only=True)
        self.arr, self.shift = arr, shift
        self.choir = ReedVoice(level=self.LEVEL)
        notes = []
        for part, seq in arr['parts'].items():
            for on, off, m in seq:
                notes.append((float(on), float(off), int(m) + shift, part))
        self.parts = sorted(notes)
        self.next_part = 0
        self.melody = [(float(a), float(b), int(m) + shift) for a, b, m in arr['melody']]

    def _schedule(self):
        while self.next_part < len(self.parts):
            on, off, m, part = self.parts[self.next_part]
            start = int(round((self.t0 + on * self.T_b - self.Lo - self.PRE) * SR))
            if start > self.n + HOP + int(self.AHEAD * SR):
                break
            dur = max(0.05, (off - on) * self.T_b - self.GAP)
            self.choir.schedule(max(start, self.choir.pos), dur, m, tag=('choir', self.next_part))
            self.next_part += 1

    def part_at(self, beat, part='A'):
        for on, off, m, p in self.parts:
            if p == part and on <= beat < off:
                return m
        return -1

    def process(self, x):
        self._schedule()
        yc = self.choir.render(HOP)
        y = super().process(x)
        return y + yc

    def _react(self, t):
        n0 = len(self.events)
        super()._react(t)
        for e in self.events[n0:]:                       # show her heard note against the alto part
            e['duet'] = self.part_at(e['onset_beats'])
            e['action'] = 'keep'

    def _stop(self, t):
        pass

    def done(self):
        return self.n / SR > self.t0 + (self.K + 1) * self.T_b

    def summary(self):
        s = super().summary()
        s.update({'mode': 'choir', 'choir_song': self.arr['id'], 'choir_shift': self.shift,
                  'parts_written': self.next_part, 'parts_total': len(self.parts)})
        return s
