"""Trace one v11 phone run beat by beat (run choir_1102): for each melody note, its true onset (offline pyin on the run's
mic), the onset the live follower used, the beat map's prediction before it, and the tempo after; for each part chord
on a melody onset, when it sounded vs the true onset. Replays through page v13's choir (offline, tripwire).

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/v11_trace_v1.py [run]
"""
import sys

sys.argv = sys.argv[:1] + (sys.argv[1:] or ['114908_r2_mix_v2'])
import choir_timing_eval_v4 as E  # noqa: E402  (installs the tripwire, page v13, logging)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import os  # noqa: E402

run = sys.argv[1]
E.P3.MODELS.load()
lib = E.P13.P12.P5.CHOIR
x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
sung = E.true_notes(x)
score = [(float(a), int(m)) for a, b, m in lib['v11']['melody']]
truth = E.match(score, sung, lib['v11']['bpm'])
_acc = E.CV6.ChoirStart._accept
TRACE = []


def acc(self, note, c):
    b = self.mel[c][0]
    pred = self.t_of(b)
    _acc(self, note, c)
    TRACE.append((b, note['t'], pred, 60 / self.Tb, note['pitch']))


E.CV6.ChoirStart._accept = acc
eng, log = E.replay(x, {})
print('sung notes (true, offline):', [(round(a, 2), round(p, 1)) for a, e, p in sung[:40]])
print(f"{'beat':>6} {'true':>7} {'live':>7} {'live-true':>9} {'pred':>7} {'pred-true':>9} {'bpm after':>9}")
for b, t, pred, bpm, p in TRACE:
    tr = truth.get(b)
    print(f"{b:6.2f} {tr if tr is None else round(tr, 3):>7} {t:7.3f} "
          f"{'' if tr is None else round((t - tr) * 1000):>9} {pred:7.3f} {'' if tr is None else round((pred - tr) * 1000):>9} "
          f"{bpm:9.1f}")
print('start', eng.diag.get('start_t'), 'prov', eng.diag.get('prov_time'), 'id', eng.diag.get('id_time'))
seen = set()
for start, idx, sid in log:
    if sid != eng.song_id or idx is None or idx >= len(eng.parts):
        continue
    on = eng.parts[idx][0]
    if on in seen:
        continue
    seen.add(on)
    tr = truth.get(on)
    print(f"chord at beat {on:6.2f}: sounds {start / E.SR + eng.Lo:7.3f}  true onset {tr if tr is None else round(tr, 3)}"
          f"  diff {'' if tr is None else round((start / E.SR + eng.Lo - tr) * 1000)} ms")
