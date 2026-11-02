"""Choir rehearsal engine v3 (run choir_1102). Vincent (2026-11-02): v2 sounded good but Duet was a little late on
every note. The 09:49 run showed why: her onsets came a median 104 ms before Duet's grid, because v2 snapped her first
onset onto the page's click grid while the melody (a phone file) ran on its own phase; a singer who sits slightly ahead
of the clicks does the same. v3:

- No snapping: her first note sets beat 0 where she actually sang it. A note only counts as her first note if its pitch
  class is the melody's first note's (the phone file's own count-in clicks were taken for her start).
- Phase following: every later onset that lies within 0.35 beat of a melody onset gives an error (her onset minus the
  grid); with 3 or more, the median of the last 6 moves beat 0 by 60 % of it (at most 60 ms per note), so Duet keeps
  sitting on her beat even with a wrong latency estimate or a slow drift. Tempo is not changed.
- Parts are written LEAD = 30 ms earlier than v2, for the reed's 60 ms attack (its onset is heard a little after it starts).

  ChoirDuet(arr, key_pc, bpm, click_all=False)   arr = choir_songs_v3.json['songs'][id]
"""
import numpy as np

import choir_v1 as C1
import choir_v2 as C2
from dsp import HOP, SR

load_songs = C1.load_songs


class ChoirDuet(C2.ChoirDuet):
    LEAD = 0.030           # s: parts start this much earlier than the grid (beyond PRE)
    MATCH = 0.35           # beats: an onset this close to a melody onset is used for phase following
    GAIN, MAX_STEP, MIN_ERR = 0.6, 0.060, 0.012

    def __init__(self, arr, key_pc, bpm, click_all=False, log=lambda s: None):
        super().__init__(arr, key_pc, bpm, click_all=click_all, log=log)
        self.mel_on = np.array([a for a, b, m in self.melody])
        self.first_pc = self.melody[0][2] % 12
        self.errs, self.shifts = [], []

    def _schedule(self):
        if self.song_t0 is None or self.stopped:
            return
        while self.next_part < len(self.parts):
            on, off, m, part = self.parts[self.next_part]
            start = int(round((self.song_t0 + on * self.T_b - self.Lo - self.PRE - self.LEAD) * SR))
            if start > self.n + HOP + int(self.AHEAD * SR):
                break
            dur = max(0.05, (off - on) * self.T_b - self.GAP)
            if start < self.choir.pos:
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
            e['song_beat'], e['duet'], e['action'] = None, -1, 'keep'
            if self.song_t0 is None:
                if e['her_heard'] % 12 == self.first_pc:
                    self.song_t0 = c0
                    self.log(f'she started: song beat 0 at {c0:.3f} s')
                    e['song_beat'] = 0.0
                    e['duet'] = self.part_at(0.0)
                continue
            sb = (c0 - self.song_t0) / self.T_b
            e['song_beat'] = round(sb, 3)
            e['duet'] = self.part_at(sb)
            k = int(np.argmin(np.abs(self.mel_on - sb)))
            if abs(self.mel_on[k] - sb) <= self.MATCH and sb > 0.5:
                self.errs.append((sb - self.mel_on[k]) * self.T_b)
                if len(self.errs) >= 3:
                    off = float(np.median(self.errs[-6:]))
                    if abs(off) > self.MIN_ERR:
                        step = float(np.clip(self.GAIN * off, -self.MAX_STEP, self.MAX_STEP))
                        self.song_t0 += step
                        self.errs = [x - step for x in self.errs]
                        self.shifts.append(round(step * 1000, 1))

    def summary(self):
        s = super().summary()
        s.update({'choir_engine': 'choir_v3', 'phase_shifts_ms': self.shifts, 'lead_ms': self.LEAD * 1000})
        return s
