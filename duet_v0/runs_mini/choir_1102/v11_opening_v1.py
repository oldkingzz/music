"""How did the live tracker hear the first two notes of v11 (A3, then E3) in each phone run? (run choir_1102) The
opening interval A -> E (2 beats) sets the starting tempo, so an onset heard late on either note makes the first chords
early or late. Prints, per run: the true onsets (offline pyin), the live notes around them, and the tracker frames
(10 ms: voiced flag and pitch) from 0.15 s before each true onset to 0.3 s after.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py v11open1 600 .venv/bin/python runs_mini/choir_1102/v11_opening_v1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402

import soundfile as sf  # noqa: E402

E = V.E
RUNS = sys.argv[1:] or ['113335_r2_mix_v2', '114908_r2_mix_v2', '124132_r2_mix_v2', '132458_r2_mix_v2']
E.P3.MODELS.load()
lib = E.P13.P12.P5.CHOIR
score = [(float(a), int(m)) for a, b, m in lib['v11']['melody']]
for run in RUNS:
    x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
    truth = E.match(score, E.true_notes(x), lib['v11']['bpm'])
    eng, _ = E.replay(x, {})
    a, e = truth.get(0.0), truth.get(2.0)
    print(f'== {run}: true A {a and round(a, 3)}, true E {e and round(e, 3)}; engine start {eng.diag.get("start_t")} '
          f'(A {eng.diag.get("start_t") - a if a else 0:+.3f}), id {eng.diag.get("id_time")} '
          f'(E {eng.diag.get("id_time") - e if e else 0:+.3f}), opening tempo '
          f'{120 / (eng.diag.get("id_time") - eng.diag.get("start_t")):.1f} bpm (true {120 / (e - a) if a and e else 0:.1f})')
    for name, t0 in (('A', a), ('E', e)):
        if t0 is None:
            continue
        hn = [(round(n['t'], 3), round(n['pitch'], 1)) for n in eng.hnotes if t0 - 0.4 <= n['t'] <= t0 + 0.4]
        fr = [(round(t, 2), int(v), round(m, 1)) for t, v, m in zip(eng.ft_t, eng.ft_v, eng.ft_m)
              if t0 - 0.15 <= t <= t0 + 0.3]
        print(f'  {name}: live notes {hn}')
        print('     frames (t, voiced, midi):', ' '.join(f'{t:.2f}:{"v" if v else "-"}{m:.1f}' for t, v, m in fr))
