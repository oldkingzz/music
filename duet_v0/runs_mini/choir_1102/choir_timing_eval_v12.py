"""v12 of the choir timing eval (run choir_1102): tracing each note's onset back through a soft or octave-slipped
attack (choir_v6.ONSET_BACK; the live tracker heard v11's soft E3 up to 0.21 s late and its A3 attack an octave high).
Adds Vincent's 13:24 v11 run (rated 一般) and reports, per run, the opening tempo (from the first two onsets) against the
true one, and how late the live onsets are against the true ones (detect), besides chord timing and one-voice stats.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py choir_teval12 900 .venv/bin/python runs_mini/choir_1102/choir_timing_eval_v12.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402  (tripwire, page v13, truth, replay, NOTES, voice_stats)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

E = V.E
OUT = os.path.join(E.ABS, 'runs_mini/choir_1102/choir_timing_eval_v12.json')
RUNS = dict(E.RUNS)
RUNS['132458_r2_mix_v2'] = 'v11'
VARIANTS = {
    'L now (legato)': dict(ONSET_BACK=0),
    'O + onset traced back': dict(ONSET_BACK=0.3),
}

if __name__ == '__main__':
    E.P3.MODELS.load()
    lib = E.P13.P12.P5.CHOIR
    res = {}
    for run_name, sid in RUNS.items():
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run_name, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        sung = E.true_notes(x)
        print(run_name, sid, flush=True)
        for vname, st in VARIANTS.items():
            V.NOTES.clear()
            eng, _ = E.replay(x, st)
            score = [(float(a), int(m) + (eng.S or 0)) for a, b, m in lib[sid]['melody']]
            truth = E.match(score, sung, lib[sid]['bpm'] if sid == 'v11' else 60.0 / eng.Tb)
            log = [(nt.i0, idx, song) for idx, song, nt in V.NOTES if nt is not None]
            r = E.score_run(eng, log, truth)
            pe = np.abs(eng.diag['pred_err_ms'])
            r.update({'pred_med': round(float(np.median(pe))) if len(pe) else None,
                      'pred_p90': round(float(np.percentile(pe, 90))) if len(pe) else None,
                      'traced': eng.diag.get('onset_traced', 0)})
            (t0, b0), (t1, b1) = eng.pairs[:2] if len(eng.pairs) >= 2 else ((0, 0), (0, 1))
            if b0 in truth and b1 in truth:
                r['open_bpm'] = round(60 * (b1 - b0) / (t1 - t0), 1)
                r['open_true_bpm'] = round(60 * (b1 - b0) / (truth[b1] - truth[b0]), 1)
            r.update(V.voice_stats(eng))
            res[f'{run_name} | {vname}'] = r
            keep = ('chord_med_abs_from_target', 'chord_p90_abs_from_target', 'detect_median_ms', 'detect_med_abs_ms',
                    'pred_med', 'pred_p90', 'open_bpm', 'open_true_bpm', 'traced', 'two_at_once', 'holes', 'song')
            print(f'   {vname:24s}', json.dumps({k: r.get(k) for k in keep}), flush=True)
            print('      per chord (beat, ms):', sorted(set(r['per_chord'])), flush=True)
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1)
