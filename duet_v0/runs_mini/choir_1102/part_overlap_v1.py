"""Does a part ever sing two notes at once? (run choir_1102) A part note's start and end are both fixed when it is
written (0.3 s ahead of its start); if the beat map moves earlier before the part's NEXT note is written, that next note
starts before the previous one ends, and the reed (polyphonic) sounds both pitches in one part. Replays each of
Vincent's phone runs (current engine defaults, offline, tripwire) and reports, per run, over consecutive notes of the
same part that are contiguous in the score: how often the next note starts more than 20 ms before the previous one
ends (beyond the designed 30 ms gap + 50 ms release), by how much, and how often a gap of more than 80 ms opens.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py overlap1 600 .venv/bin/python runs_mini/choir_1102/part_overlap_v1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v10 as E  # noqa: E402  (tripwire, page v13, logging)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

NOTES = []
_init = E.CV6.ChoirStart.__init__


def _init_full(self, *a, **k):
    _init(self, *a, **k)
    sch = self.choir.schedule

    def schedule(start, dur, midi, tag=None):
        NOTES.append((start / E.SR, dur, midi, tag[1] if tag else None, self.song_id, self.n / E.SR))
        return sch(start, dur, midi, tag=tag)
    self.choir.schedule = schedule


E.CV6.ChoirStart.__init__ = _init_full
E.P3.MODELS.load()
for run, sid in E.RUNS.items():
    if sid not in ('v11', 'twinkle') or run.startswith(('0954', '1102')):
        continue
    x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
    NOTES.clear()
    eng, _ = E.replay(x, {})
    by_part = {}
    for s, d, m, idx, song, now in NOTES:
        if song != eng.song_id or idx is None:
            continue
        on, off, mm, p = eng.parts[idx]
        by_part.setdefault(p, []).append((on, off, s, s + d, m))
    over, gaps, n, clash = [], [], 0, 0.0
    for p, seq in by_part.items():
        seq.sort()
        for (on0, off0, s0, e0, m0), (on1, off1, s1, e1, m1) in zip(seq, seq[1:]):
            if abs(off0 - on1) > 1e-6:
                continue                                   # a rest in the part between them
            n += 1
            o = e0 - s1                                    # > 0: the next note starts before this one ends
            if o > -0.03 + 0.02:                           # designed: e0 = s1 - 0.03 (the GAP)
                over.append(o + 0.03)
                if round(m0) != round(m1):
                    clash += o
            if -o > 0.03 + 0.08:
                gaps.append(-o - 0.03)
    f = lambda a: round(float(np.median(a)) * 1000) if a else None  # noqa: E731
    g = lambda a: round(float(np.max(a)) * 1000) if a else None  # noqa: E731
    print(f'{run} ({sid}): {n} contiguous part steps; overlaps > 20 ms: {len(over)} (median {f(over)}, max {g(over)} '
          f'ms; two pitches at once for {clash:.2f} s in all); gaps > 80 ms: {len(gaps)} (median {f(gaps)}, max '
          f'{g(gaps)} ms)', flush=True)
