"""Any bench_rt system live (run v5rt_1101): the mic hears her, Duet's reed sings in the headphones, and the decisions
are bench_rt's two kinds (futures/v5_rt/bench_rt.py, read its SPEC), so a system scored there runs here unchanged.

RTDuet = live_bench3.Bench3Duet (same pitch tracker, same online note segmentation, same 0.25-beat onset snapping, same
reed voice) with bench_rt's decision points in place of one step() per note:

- PLAN POINTS every 0.5 beat at t0 + q * T_b (q = 0, 0.5, ... < K). system.plan(q, heard) is called as late as possible
  (when its note would have to be written within the next two 10 ms blocks) and the note is written early by the
  output latency Lo plus PRE = 0.025 s, as live_predict.PredictDuet writes its beat notes, so it sounds on the half beat
  (the reed needs 60 ms to reach full level). heard = her notes the tracker has answered so far, as fresh tuples
  (onset, offset or None, heard midi) in beats from beat 0; onsets are snapped like Bench3Duet's; the offset is known
  once the segmenter has closed the note and is onset + her snapped duration (the number the next hear() gets as
  prev_dur).
- HEAR POINTS: each of her notes, as soon as the tracker answers it (Bench3Duet._react: 3 frames on its pitch, 12 for
  the first note when beat 0 is taken from it), calls system.hear(i, onset, midi, prev_dur) with bench3 step's
  arguments (prev_dur = her previous answered note's snapped duration, None for the first); its note sounds at once.
- Every call returns the note Duet sounds from then on: None = keep, -1 = silence, h = sing h (no new attack when Duet
  already sounds h). A decision replaces a planned note that has not started yet: that note is dropped, never cut to a
  zero-length blip.
- She stops (no open note and no voiced frame for hold_gap = 0.25 s): Duet stops, as in Bench3Duet. bench_rt has no
  such rule (Duet's note rings on through her rests there), so the system still believes its note sounds; the wrapper
  remembers it (sys_note) and, when she comes back, a hear() that answers None re-enters that note at once (in bench_rt
  it would still be sounding). During the rest a plan() None keeps Duet silent; a plan() that names a note enters it on
  the half beat (an on-time entry), and such a note stops again if she has not sung within hold_gap of its start.
- Beat 0 = her first note (the first plan point used is then the first one still ahead), unless count_in > 0: Duet then
  plays a click count-in (PredictDuet._click, copied) and the grid t0 = lead + count_in * T_b is known in advance.
Every decision is logged in self.decisions (when it sounds, q or her note index, the system's result, the action,
Duet's note after it, the system call's microseconds) and through log(); summary() has PredictDuet's fields plus the
decision counts. This file opens no sound device: a caller (the page) feeds process(); the offline comparison on the
phone files is futures/v4_selfiter/live_rt_eval.py.
"""
import importlib
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_bench3 as LB  # noqa: E402  (puts duet_beat, live, panel and v3 on sys.path)
from live_bench3 import HOP, SR, Bench3Duet  # noqa: E402

V5 = os.path.join(LB.ABS, 'futures', 'v5_rt')
if V5 not in sys.path:
    sys.path.insert(0, V5)                               # the rt_systems package (not v3's `systems`)

STEP = 0.5                                               # beats between plan points (bench_rt)


def rt_build(name, train):
    """futures/v5_rt/rt_systems/<name>.py built on train (= bench3.train_sets())."""
    return importlib.import_module('rt_systems.' + name).build(train)


def _name(h):
    if h is None:
        return 'keep'
    return LB.lm.note_name(int(h)) if h >= 0 else 'silent'


class RTDuet(Bench3Duet):
    """One bench_rt system (plan / hear) live, on Bench3Duet's tracker, segmenter and voice."""
    PRE = 0.025            # s: a planned note starts this much before its plan point (as PredictDuet)

    def __init__(self, system, name, tonic, M, K, bpm, first_beat=0.0, listen_only=False, monitor=0.0,
                 hold_gap=0.25, grid=0.25, voice='reed', log=print, count_in=0, click_all=False, lead=0.5):
        super().__init__(system, name, tonic, M, K, bpm, first_beat=first_beat, listen_only=listen_only,
                         monitor=monitor, hold_gap=hold_gap, grid=grid, voice=voice, log=log)
        self.count_in, self.click_all, self.lead = count_in, click_all, lead
        self.heard_notes = []        # her answered notes [onset, offset or None, heard midi] in beats, in order
        self.heard_dur = []          # each one's snapped duration once the segmenter has closed it (None while open)
        self.by_start = {}           # segmenter start time (s) of an answered note -> its index in heard_notes
        self.next_j = None           # the next plan point is q = next_j * STEP (None until the grid is known)
        self.live = -1               # the note Duet sounds (or has scheduled) after the last decision; -1 = silent
        self.sys_note = -1           # the note the system believes Duet sounds (bench_rt's step function)
        self.plan_t = None           # acoustic time (s) the last note started by a plan point begins
        self.tags, self.n_tag = [], 0
        self.plans, self.decisions = [], []
        self.counts = Counter()
        self.us_plan, self.us_hear = [], []
        if count_in:                                     # she sings on Duet's click: the grid is known ahead
            self.t0 = lead + count_in * self.T_b
            self.next_j = 0

    # ---------------------------------------------------------------- one block
    def process(self, x):
        c0 = time.perf_counter()
        x = np.asarray(x, float)
        i = self.ft.step(x)
        t = self.ft.t(i) - self.Li
        self.ft_t.append(t)
        self.ft_m.append(np.nan_to_num(self.ft.m[-1]))
        self.ft_v.append(self.ft.v[-1])
        nov = self.ft.nov
        if self.ft.v[-1] and nov[-1] >= 0.5 and (len(nov) < 2 or nov[-2] < 0.5):
            self.on_t.append(t)
        if self.ft.v[-1]:
            self.last_voiced = t
        self._segment()
        self._react(t)                                   # a hear sounds now, before any plan note written ahead
        self._plan()
        self._stop(t)
        y = self.voice.render(HOP) + self._click(HOP) + self.monitor * x
        self.n += HOP
        self.proc_ms.append((time.perf_counter() - c0) * 1000)
        return y

    def _click(self, n):
        """PredictDuet._click: LiveDuet's click (count-in, or every beat with click_all), sent early by Lo."""
        if not self.count_in:
            return np.zeros(n)
        y = np.zeros(n)
        last = self.K if self.click_all else 0
        for k in range(-self.count_in, last):
            i0 = int(round((self.t0 + k * self.T_b - self.Lo) * SR)) - self.n
            L = int(0.025 * SR)
            if i0 + L <= 0 or i0 >= n:
                continue
            tt = np.arange(L) / SR
            f = 1760.0 if k % self.M == 0 else 1320.0
            s = (0.35 if k % self.M == 0 else 0.22) * np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.006)
            a, b = max(i0, 0), min(i0 + L, n)
            y[a:b] += s[a - i0:b - i0]
        return y

    # ---------------------------------------------------------------- her notes: heard list and offsets
    def _close(self):
        """PolicyDuet._close, and an answered note gets its offset (snapped like Bench3Duet's prev_dur)."""
        c = self.cur
        if c is not None:
            j = self.by_start.get(c[0])
            if j is not None and self.heard_dur[j] is None:
                d = self._snap((c[1] + 0.01 - c[0]) / self.T_b, True)
                self.heard_dur[j] = d
                self.heard_notes[j][1] = self.heard_notes[j][0] + d
        super()._close()

    def _prev_dur(self, i):
        d = self.heard_dur[i - 1]
        if d is not None:
            return d
        prev = self.notes[-1] if self.notes else None     # not closed yet (the segmenter always closes it first)
        return None if prev is None else self._snap((prev[1] - prev[0]) / self.T_b, True)

    # ---------------------------------------------------------------- hear point: her note answered
    def _react(self, t):
        c = self.cur
        need = 12 if self.t0 is None else 3              # the first note syncs beat 0: 120 ms, not a click
        if c is None or self.ended or c[0] == self.answered or sum(c[2].values()) < need:
            return
        self.answered = c[0]
        midi = int(max(c[2], key=c[2].get))
        if self.t0 is None:
            self.t0 = c[0] - self.first_beat * self.T_b
        if self.next_j is None:                          # grid from her first note: the plan points still ahead
            j = 0
            while j * STEP < self.K and self._start(j * STEP) < self.voice.pos:
                j += 1
            self.next_j = j
        onset_raw = (c[0] - self.t0) / self.T_b
        onset = float(self._snap(onset_raw))
        if onset >= self.K:
            self.ended = True
            return
        i = len(self.heard_notes)
        prev_dur = None if i == 0 else self._prev_dur(i)
        self.heard_notes.append([onset, None, midi])
        self.heard_dur.append(None)
        self.by_start[c[0]] = i
        before = self.live
        c1 = time.perf_counter()
        r = self.system.hear(i, onset, midi, prev_dur)
        us = (time.perf_counter() - c1) * 1e6
        self.us_hear.append(us)
        at = self.voice.pos
        action = self._apply(r, at, 'hear')
        lag = (t - c[0]) * 1000
        self.events.append({'i': i, 'onset_beats': round(onset, 3), 'onset_raw': round(onset_raw, 3),
                            'her_heard': midi, 'duet': self.live, 'before': before,
                            'result': None if r is None else int(r), 'action': action,
                            'prev_dur': None if prev_dur is None else round(prev_dur, 3),
                            'detect_ms': round(lag, 1), 'us': round(us, 1)})
        self._record('hear', at, None, i, r, action, us)
        bar, beat = divmod(int(np.floor(onset + 1e-6)), self.M)
        self.log(f'{bar + 1:3d}.{beat + 1}    she {LB.lm.note_name(midi):4s} -> Duet {_name(self.live):6s} '
                 f'{action:8s} heard {lag:3.0f} ms after her onset')

    # ---------------------------------------------------------------- plan points: every half beat, written early
    def _start(self, q):
        """Output sample at which the note for plan point q is written (heard PRE before the half beat)."""
        return int(round((self.t0 + q * self.T_b - self.Lo - self.PRE) * SR))

    def _plan(self):
        if self.next_j is None or self.ended:
            return
        while self.next_j * STEP < self.K:
            q = self.next_j * STEP
            start = self._start(q)
            if start >= self.n + 2 * HOP:
                return                                   # not yet: decide as late as possible
            self.next_j += 1
            view = [(n[0], n[1], n[2]) for n in self.heard_notes]
            c1 = time.perf_counter()
            r = self.system.plan(q, view)
            us = (time.perf_counter() - c1) * 1e6
            self.us_plan.append(us)
            late = max(0, self.voice.pos - start)
            if late:
                self.late.append(late)
            at = max(start, self.voice.pos)
            action = self._apply(r, at, 'plan')
            self.plans.append({'q': q, 'result': None if r is None else int(r), 'action': action, 'duet': self.live,
                               'heard': len(view), 'late_ms': round(late / SR * 1000, 1), 'us': round(us, 1)})
            self._record('plan', at, q, None, r, action, us)
            bar, beat = divmod(int(q), self.M)
            self.log(f'{bar + 1:3d}.{beat + 1}{"+" if q != int(q) else " "}  plan -> {_name(r):6s} {action:8s}'
                     f' Duet {_name(self.live)}' + (f'  LATE {late / SR * 1000:.0f} ms' if late else ''))

    # ---------------------------------------------------------------- applying an answer
    def _apply(self, r, at, kind):
        """Duet sounds the system's answer r from output sample `at` on; returns what happened."""
        if r is None:
            if self.live >= 0:
                return 'keep'
            if kind == 'hear' and self.sys_note >= 0:    # Duet stopped in her rest; the system's note still holds
                self._sing(self.sys_note, at)
                return 'reenter'
            return 'idle'
        r = int(r)
        self.sys_note = r
        if r < 0:
            if self.live >= 0:
                self._sing(-1, at)
                return 'silence'
            return 'rest'
        if r == self.live:
            return 'same'
        action = 'switch' if self.live >= 0 else 'enter'
        self._sing(r, at)
        if kind == 'plan':
            self.plan_t = at / SR + self.Lo
        return action

    def _sing(self, h, start):
        """From output sample `start` on Duet sounds h (-1: nothing). A note that has not started by then is dropped
        (never cut to a zero-length blip); a sounding note ends at `start`, unless it already is h: it then rings on."""
        v = self.voice
        start = max(int(start), v.pos)
        alive = []
        for tag in self.tags:
            nt = v.by_tag.get(tag)
            if nt is None:
                continue
            if nt.i0 >= start and nt.done == 0:
                v.notes.remove(nt)
                del v.by_tag[tag]
                nt.level = nt.dur = 0.0                  # marks it dropped: it never made a sound
                self.counts['dropped_unstarted'] += 1
            else:
                alive.append((tag, nt))
        ring = None
        if h >= 0:
            for tag, nt in alive:
                end = nt.i0 + int(nt.dur * SR)
                if int(round(nt.midi)) == h and end >= start and nt.done <= int(nt.dur * SR):
                    ring = tag
        keep = []
        for tag, nt in alive:
            if tag == ring:
                v.set_dur(tag, 60.0 + (start - nt.i0) / SR)
            elif nt.i0 + int(nt.dur * SR) > start:
                v.set_dur(tag, (start - nt.i0) / SR)
            if nt.i0 + int(nt.dur * SR) > v.pos:
                keep.append(tag)
        if h >= 0 and ring is None and not self.listen_only:
            self.n_tag += 1
            tag = ('rt', self.n_tag)
            v.schedule(start, 60.0, h, tag=tag)
            keep.append(tag)
        self.tags = keep
        self.live = h

    # ---------------------------------------------------------------- she stopped: Duet stops
    def _stop(self, t):
        """Bench3Duet's rule (no open note, no voiced frame for hold_gap), except that a note a plan point started
        gets hold_gap from its own start: a planned entry waits that long for her."""
        if self.live < 0:
            return
        why = None
        if self.ended:
            why = 'end'
        elif self.cur is None:
            refs = [x for x in (self.last_voiced, self.plan_t) if x is not None]
            if refs and t - max(refs) > self.hold_gap:
                why = 'she_stopped'
        if why is None:
            return
        at = self.voice.pos
        self._sing(-1, at)
        self._record('stop', at, None, None, -1, why, 0.0)
        self.log(f'          she stopped -> Duet silent ({why})')

    def _record(self, kind, at, q, i, r, action, us):
        self.counts[f'{kind}_{action}'] += 1
        self.decisions.append({'kind': kind, 't': round(at / SR + self.Lo, 4), 'q': q, 'i': i,
                               'result': None if r is None else int(r), 'action': action, 'duet': self.live,
                               'us': round(us, 1)})

    # ---------------------------------------------------------------- report
    def summary(self):
        s = super().summary()
        acts = Counter(e['action'] for e in self.events)
        us = self.us_plan + self.us_hear
        mean = lambda a: float(np.mean(a)) if a else None  # noqa: E731
        s.update({'mode': 'rt', 'count_in': self.count_in, 'click_all': self.click_all,
                  'kept': acts['keep'] + acts['same'], 'corrected': acts['switch'] + acts['silence'],
                  'late_start': acts['enter'] + acts['reenter'], 'plans': self.plans,
                  'decision_counts': dict(sorted(self.counts.items())), 'decisions': self.decisions,
                  'late_decisions': len(self.late),
                  'us_plan_mean': mean(self.us_plan), 'us_hear_mean': mean(self.us_hear), 'us_mean': mean(us),
                  'us_max': float(np.max(us)) if us else None})
        return s
