"""v5: one v11 run beat by beat with the onset tracing off and on (run choir_1102), to see which traced notes move the
beat map: per matched melody note, live onset - true onset, prediction - true onset and bpm after, for both; then every
note the tracing moved: when the tracker first heard it, where the trace put it, its pitch, and the nearest true onset.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py v11trace5 600 .venv/bin/python runs_mini/choir_1102/v11_trace_v5.py RUN
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402

import soundfile as sf  # noqa: E402

E = V.E
run = sys.argv[1] if len(sys.argv) > 1 else '113335_r2_mix_v2'
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
x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
sung = E.true_notes(x)
truth = E.match(score, sung, lib['v11']['bpm'])
rows, moved = {}, None
for tag, st in (('off', dict(ONSET_BACK=0)), ('on', dict(ONSET_BACK=0.3))):
    TRACE.clear()
    eng, _ = E.replay(x, st)
    for b, t, pred, bpm in TRACE:
        rows.setdefault(b, {})[tag] = (t, pred, bpm)
    if tag == 'on':
        moved = [(round(n['t_heard'], 3), round(n['t'], 3), round(n['pitch'], 1)) for n in eng.hnotes if 't_heard' in n]
print(f"{'beat':>6} {'true':>7} | {'off: live-tr':>12} {'pred-tr':>8} {'bpm':>6} | {'on: live-tr':>11} {'pred-tr':>8} {'bpm':>6}")
for b in sorted(rows):
    tr = truth.get(b)
    cells = []
    for tag in ('off', 'on'):
        if tag in rows[b]:
            t, pred, bpm = rows[b][tag]
            f = lambda v: '-' if tr is None else str(round((v - tr) * 1000))  # noqa: E731
            cells.append(f'{f(t):>12} {f(pred):>8} {bpm:6.1f}')
        else:
            cells.append(' ' * 28)
    print(f"{b:6.2f} {'-' if tr is None else round(tr, 3):>7} | " + ' | '.join(cells))
print('notes the trace moved (heard at -> traced to, pitch; nearest true onset):')
for th, t, p in moved:
    near = min(sung, key=lambda s: abs(s[0] - th))
    print(f'   {th:7.3f} -> {t:7.3f} ({(t - th) * 1000:+5.0f} ms)  pitch {p:5.1f}   true {near[0]:7.3f} '
          f'(pitch {near[2]:.1f}): traced - true {(t - near[0]) * 1000:+5.0f} ms')
