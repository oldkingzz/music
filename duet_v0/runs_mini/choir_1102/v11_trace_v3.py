"""v3: the three v11 phone runs side by side, current engine defaults (run choir_1102). All three play the same
recording, so the true onsets (offline pyin, relative to her first note) should agree; where the chords miss in all
three runs, the cause is the recording's own timing (rubato), not the live detection. Per melody note: true onset
(relative), live - true, prediction - true, bpm after; per chord on a melody onset: sounds - true (target -35 ms).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py v11trace3 600 .venv/bin/python runs_mini/choir_1102/v11_trace_v3.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v10 as E  # noqa: E402  (tripwire, page v13, logging)

import soundfile as sf  # noqa: E402

RUNS = [r for r, s in E.RUNS.items() if s == 'v11']
E.P3.MODELS.load()
lib = E.P13.P12.P5.CHOIR
score = [(float(a), int(m)) for a, b, m in lib['v11']['melody']]
_acc = E.CV6.ChoirStart._accept
TRACE = []


def acc(self, note, c):
    b = self.mel[c][0]
    pred = self.t_of(b)
    _acc(self, note, c)
    TRACE.append((b, note['t'], pred, 60 / self.Tb))


E.CV6.ChoirStart._accept = acc
rows, chords = {}, {}
for run in RUNS:
    x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
    truth = E.match(score, E.true_notes(x), lib['v11']['bpm'])
    TRACE.clear()
    eng, log = E.replay(x, {})
    t0 = truth.get(0.0)
    for b, t, pred, bpm in TRACE:
        tr = truth.get(b)
        rows.setdefault(b, {})[run] = (None if tr is None or t0 is None else tr - t0,
                                       None if tr is None else t - tr, None if tr is None else pred - tr, bpm)
    seen = set()
    for start, idx, sid in log:
        if sid != eng.song_id or idx is None or idx >= len(eng.parts):
            continue
        on = eng.parts[idx][0]
        if on in seen or on not in truth:
            continue
        seen.add(on)
        chords.setdefault(on, {})[run] = (start / E.SR + eng.Lo) - truth[on]
    print(run, 'start', eng.diag.get('start_t'), 'id', eng.diag.get('id_time'), 'song', eng.song_id, flush=True)


def f(v, k=1000):
    return '     -' if v is None else f'{v * k:6.0f}'


print('beat  | ' + ' | '.join(f'{r[:6]}: true-rel  live-tr  pred-tr  bpm' for r in RUNS))
for b in sorted(rows):
    print(f'{b:5.2f} | ' + ' | '.join(
        (f'{f(rows[b][r][0])}   {f(rows[b][r][1])}  {f(rows[b][r][2])}  {rows[b][r][3]:5.1f}' if r in rows[b]
         else ' ' * 34) for r in RUNS))
print('chord beat | ' + '  '.join(f'{r[:6]}' for r in RUNS) + '   (sounds - true onset, ms; target -35)')
for on in sorted(chords):
    print(f'{on:10.2f} | ' + '  '.join(f(chords[on].get(r)) for r in RUNS))
