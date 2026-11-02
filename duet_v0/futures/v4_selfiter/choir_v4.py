"""Choir rehearsal engine v4 (run choir_1102), Vincent 2026-11-02:
(1) a cappella sounded worse than the phone file. His 09:54 run: the tracker split his low voice into many short notes
    (46/47 alternating), and v3's phase following moved Duet on each of them (47 moves of up to 60 ms).
(2) "don't make me pick the song: whatever I sing, if it is in the library".

v4 = ChoirAuto(songs, bpm):
- Song and key from her singing: her distinct pitches (repeats merged, notes < 0.2 beat ignored) give an interval
  sequence (octave-free, -6..+5); a song matches when its melody's first intervals agree (+-1 semitone). Duet is silent
  until exactly one song matches her first 2 intervals (3 distinct notes; more if still ambiguous); the key is her first
  note's pitch class minus the song's. Beat 0 = her first onset; the parts come in from where she is.
- Following: only onsets that hit the melody note expected there (pitch class +-1, at least 0.4 beat after the last used
  onset) move Duet. Phase: median of the last 5 errors, 40 % of it, at most 40 ms per note. Tempo: a straight-line fit
  of her last 6 used onsets against their melody beats gives her beat length, kept within +-15 % of the page tempo and
  moved 50 % of the way per note. Parts are written from this beat map, LEAD 30 ms early.
- She stops for 3 beats -> the parts stop; never identified within 24 beats -> the run ends.
"""
import numpy as np

import choir_v1 as C1
from live_bench3 import ReedVoice
from dsp import HOP, SR

load_songs = C1.load_songs


def _iv(a, b):
    return ((b - a + 6) % 12) - 6


def distinct(mel, min_beats=0.0):
    out = []
    for a, b, m in mel:
        if b - a < min_beats:
            continue
        if not out or out[-1][2] != m:
            out.append([a, b, m])
    return out


class ChoirAuto(C1.LR.RTDuet):
    LEVEL, GAP, AHEAD, LEAD, PRE = 0.2, 0.03, 0.3, 0.030, 0.025
    MIN_NOTE, QUIT, WAIT = 0.2, 3.0, 24.0
    PH_GAIN, PH_MAX, TEMPO_GAIN, TEMPO_SPAN = 0.4, 0.040, 0.5, 0.15

    def __init__(self, songs, bpm, click_all=False, log=lambda s: None):
        super().__init__(C1._Listen(), 'choir:auto', 0, 4, 400, bpm, count_in=4, click_all=click_all, lead=0.5,
                         log=log, listen_only=True)
        self.songs = songs
        self.sig = {sid: [_iv(x[2], y[2]) for x, y in zip(distinct(s['melody']), distinct(s['melody'])[1:])]
                    for sid, s in songs.items()}
        self.choir = ReedVoice(level=self.LEVEL)
        self.click_t0, self.T0_b = self.t0, self.T_b        # page tempo
        self.her = []                                        # her distinct notes [time s, midi]
        self.arr = self.parts = self.melody = None
        self.shift, self.song_id = 0, None
        self.anchor_t = self.anchor_b = None                 # beat map: time = anchor_t + (beat - anchor_b) * Tb
        self.Tb = self.T0_b
        self.next_part, self.used, self.moves = 0, [], []
        self.stopped = None

    # ---------------------------------------------------------------- beat map
    def t_of(self, beat):
        return self.anchor_t + (beat - self.anchor_b) * self.Tb

    def b_of(self, t):
        return self.anchor_b + (t - self.anchor_t) / self.Tb

    def part_at(self, beat, part='A'):
        for on, off, m, p in self.parts or []:
            if p == part and on <= beat < off:
                return m
        return -1

    # ---------------------------------------------------------------- identify the song
    def _identify(self):
        her = self.her
        if len(her) < 3:
            return
        hs = [_iv(a[1], b[1]) for a, b in zip(her, her[1:])]
        n = len(hs)
        hits = [sid for sid, sg in self.sig.items() if len(sg) >= n and all(abs(x - y) <= 1 for x, y in zip(hs, sg[:n]))]
        if len(hits) != 1:
            if not hits and len(her) > 6:                    # nothing fits: start over from her latest notes
                self.her = her[-2:]
            return
        sid = hits[0]
        arr = self.songs[sid]
        d = distinct(arr['melody'])
        shift = (her[0][1] - d[0][2]) % 12
        shift = shift - 12 if shift > 6 else shift
        self.song_id, self.arr, self.shift = sid, arr, shift
        self.melody = [(float(a), float(b), int(m) + shift) for a, b, m in arr['melody']]
        self.parts = sorted((float(on), float(off), int(m) + shift, p) for p, seq in arr['parts'].items()
                            for on, off, m in seq)
        self.mel_on = np.array([a for a, b, m in self.melody])
        self.mel_pc = np.array([m % 12 for a, b, m in self.melody])
        # her distinct notes so far are the song's first distinct notes: anchor on the first, tempo from the page
        self.anchor_t, self.anchor_b = her[0][0], float(d[0][0])
        for (t, m), (a, b, mm) in zip(her, d):
            self.used.append((t, float(a)))
        self.next_part = 0
        self.log(f'song {sid}, shift {shift}, beat 0 at {self.anchor_t:.3f} s')

    # ---------------------------------------------------------------- follow her
    def _follow(self, t, midi):
        b = self.b_of(t)
        k = int(np.argmin(np.abs(self.mel_on - b)))
        if abs(self.mel_on[k] - b) > 0.4 or min((midi - self.mel_pc[k]) % 12, (self.mel_pc[k] - midi) % 12) > 1:
            return
        if self.used and self.mel_on[k] - self.used[-1][1] < 0.4:
            return
        self.used.append((t, float(self.mel_on[k])))
        pts = self.used[-6:]
        if len(pts) >= 4:                                     # tempo: slope of her onsets against melody beats
            tt, bb = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            if np.ptp(bb) >= 2:
                slope = float(np.polyfit(bb, tt, 1)[0])
                slope = float(np.clip(slope, self.T0_b * (1 - self.TEMPO_SPAN), self.T0_b * (1 + self.TEMPO_SPAN)))
                now_b = self.b_of(t)                          # re-anchor here so the map stays continuous
                self.Tb += self.TEMPO_GAIN * (slope - self.Tb)
                self.anchor_t, self.anchor_b = t, now_b
        errs = [p[0] - self.t_of(p[1]) for p in self.used[-5:]]
        if len(errs) >= 3:
            off = float(np.median(errs))
            step = float(np.clip(self.PH_GAIN * off, -self.PH_MAX, self.PH_MAX))
            if abs(off) > 0.012:
                self.anchor_t += step
                self.moves.append(round(step * 1000, 1))

    def _react(self, t):
        n0 = len(self.events)
        C1.LR.RTDuet._react(self, t)
        for e in self.events[n0:]:
            c0 = self.click_t0 + e['onset_raw'] * self.T_b
            midi = int(e['her_heard'])
            e['duet'], e['action'], e['song_beat'] = -1, 'keep', None
            if self.arr is None:
                if not self.her or self.her[-1][1] != midi:
                    self.her.append([c0, midi])
                self._identify()
                continue
            self._follow(c0, midi)
            e['song_beat'] = round(self.b_of(c0), 3)
            e['duet'] = self.part_at(e['song_beat'])

    def _close(self):
        """A note shorter than MIN_NOTE beats does not count as one of her distinct notes for identification."""
        c = self.cur
        if c is not None and self.arr is None and self.her and abs(self.her[-1][0] - c[0]) < 1e-6 and \
                (c[1] - c[0]) / self.T_b < self.MIN_NOTE:
            self.her.pop()
        super()._close()

    # ---------------------------------------------------------------- parts
    def _schedule(self):
        if self.arr is None or self.stopped:
            return
        while self.next_part < len(self.parts):
            on, off, m, part = self.parts[self.next_part]
            start = int(round((self.t_of(on) - self.Lo - self.PRE - self.LEAD) * SR))
            if start > self.n + HOP + int(self.AHEAD * SR):
                break
            end = int(round((self.t_of(off) - self.Lo - self.PRE - self.LEAD) * SR)) - int(self.GAP * SR)
            start = max(start, self.choir.pos)
            if end - start > int(0.05 * SR):
                self.choir.schedule(start, (end - start) / SR, m, tag=('choir', self.next_part))
            self.next_part += 1

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
        self._schedule()
        yc = self.choir.render(HOP)
        y = super().process(x) + yc
        if self.arr is not None and not self.stopped:
            last = self.last_voiced if self.last_voiced is not None else self.anchor_t
            if self.heard() - last > self.QUIT * self.Tb:
                self._silence()
                self.stopped = 'she_stopped'
        return y

    def _stop(self, t):
        pass

    def done(self):
        t = self.n / SR
        if self.arr is None:
            return t > self.click_t0 + self.WAIT * self.T0_b
        if self.stopped:
            return True
        return t > self.t_of(self.arr['K'] + 1)

    def summary(self):
        s = super().summary()
        s.update({'mode': 'choir', 'choir_engine': 'choir_v4', 'choir_song': self.song_id, 'choir_shift': self.shift,
                  'her_distinct': self.her[:12], 'tempo_beat_s': self.Tb, 'page_beat_s': self.T0_b,
                  'phase_moves_ms': self.moves, 'used_onsets': len(self.used), 'stopped': self.stopped,
                  'parts_written': self.next_part})
        return s
