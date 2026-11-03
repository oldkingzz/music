"""Tonight's first live Ode to Joy run (205400, phone through the AirPods mic, rated 差: the follower took 143 bpm for
the file's 96 and matched 7 notes) and the v11 run after it (205440, 一般), replayed through page v15's engine
(library choir_songs_v7) with the fixed 174 ms round trip: beat-by-beat trace of the follower against the true
onsets, chord timing, and the first true notes (offline pyin) so the opening can be checked against the score.
Offline (tripwire).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py ode_live_trace_v1 600 .venv/bin/python runs_mini/choir_1102/ode_live_trace_v1.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tonight_eval_v1 as T  # noqa: E402  (tripwire, replay, accept trace, evaluate)
import live_panel_v15  # noqa: E402,F401  (three-song library)

import soundfile as sf  # noqa: E402

E = T.E

if __name__ == '__main__':
    E.LI, E.LO = T.LI, T.LO
    E.P3.MODELS.load()
    for run, sid in (('205400_r2_mix_v2', 'ode'), ('205440_r2_mix_v2', 'v11')):
        print('########', run, flush=True)
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        print('first true notes (s, end, midi):', [[round(a, 2), round(b, 2), round(m, 1)] for a, b, m in
                                                  E.true_notes(x)[:16]], flush=True)
        r = T.evaluate(run, sid, trace=True)
        print('   ', json.dumps({k: r.get(k) for k in ('song', 'shift', 'chords', 'chord_med_abs_from_target',
                                                       'chord_p90_abs_from_target', 'pred_med', 'open_bpm',
                                                       'open_true_bpm', 'replay_vs_heard_lag_ms')}), flush=True)
