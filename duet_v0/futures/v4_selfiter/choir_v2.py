"""Choir rehearsal engine v2 (run choir_1102). v1 started the parts right after the 4-click count-in even when she had
not started singing (Vincent, 2026-11-02). v2 waits for her:

- The count-in clicks only give the tempo. Duet is silent until it hears her first note.
- Her first heard note sets the song's beat 0: its onset (Li-corrected) snapped to the nearest beat of the click grid
  when within 0.3 beat, else taken as it is. The first chord therefore comes in late by the hearing delay; every later
  part note is written Lo + PRE early and sounds on the beat.
- If she is silent for 3 beats after starting, the parts stop (release, no click) and the run ends. If she never
  starts, the run ends 16 beats after the count-in.

  ChoirDuet(arr, key_pc, bpm, click_all=False)   arr = choir_songs_v3.json['songs'][id]
"""
import numpy as np

import choir_v1 as C1
from dsp import HOP, SR

load_songs = C1.load_songs


class ChoirDuet(C1.ChoirDuet):
    SNAP = 0.3             # beats: her first onset this close to a click-grid beat is put on it
    QUIT = 3.0             # beats of silence after she started: the parts stop
    WAIT = 16.0            # beats after the count-in with no note from her: the run ends

    def __init__(self, arr, key_pc, bpm, click_all=False, log=lambda s: None):
        super().__init__(arr, key_pc, bpm, click_all=click_all, log=log)
        self.K = arr['K'] + int(self.WAIT) + 8               # RTDuet's own end check must not stop a late start
        self.click_t0 = self.t0
        self.song_t0 = None                                  # beat 0 of the song, set by her first note
        self.stopped = None

    def _schedule(self):
        if self.song_t0 is None or self.stopped:
            return
        while self.next_part < len(self.parts):
            on, off, m, part = self.parts[self.next_part]
            start = int(round((self.song_t0 + on * self.T_b - self.Lo - self.PRE) * SR))
            if start > self.n + HOP + int(self.AHEAD * SR):
                break
            dur = max(0.05, (off - on) * self.T_b - self.GAP)
            if start < self.choir.pos:                       # her first note: the chord comes in now
                dur -= (self.choir.pos - start) / SR
                start = self.choir.pos
            if dur > 0.05:
                self.choir.schedule(start, dur, m, tag=('choir', self.next_part))
            self.next_part += 1

    def _react(self, t):
        n0 = len(self.events)
        C1.LR.RTDuet._react(self, t)
        for e in self.events[n0:]:
            c0 = self.click_t0 + e['onset_raw'] * self.T_b
            if self.song_t0 is None:
                k = round((c0 - self.click_t0) / self.T_b)
                g = self.click_t0 + k * self.T_b
                self.song_t0 = g if abs(c0 - g) <= self.SNAP * self.T_b else c0
                self.log(f'she started: song beat 0 at {self.song_t0:.3f} s')
            e['song_beat'] = round((c0 - self.song_t0) / self.T_b, 3)
            e['duet'] = self.part_at(e['song_beat'])
            e['action'] = 'keep'

    def _silence(self):
        pos = self.choir.pos
        for nt in list(self.choir.notes):
            if nt.i0 >= pos:
                self.choir.notes.remove(nt)
                if nt.tag is not None:
                    self.choir.by_tag.pop(nt.tag, None)
            elif nt.i0 + int(nt.dur * SR) > pos:
                nt.dur = (pos - nt.i0) / SR

    def process(self, x):
        y = super().process(x)
        t = self.heard()
        if self.song_t0 is not None and not self.stopped:
            last = self.last_voiced if self.last_voiced is not None else self.song_t0
            if t - last > self.QUIT * self.T_b:
                self._silence()
                self.stopped = 'she_stopped'
        return y

    def done(self):
        t = self.n / SR
        if self.song_t0 is None:
            return t > self.click_t0 + self.WAIT * self.T_b
        if self.stopped:
            return t > self.heard() + 0.3 if False else True
        return t > self.song_t0 + (self.arr['K'] + 1) * self.T_b

    def summary(self):
        s = super().summary()
        s.update({'choir_engine': 'choir_v2', 'song_t0': self.song_t0, 'click_t0': self.click_t0,
                  'stopped': self.stopped})
        return s
