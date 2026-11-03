"""v14 of the choir timing eval (run choir_1102): tracing the onset of every note but keeping it only when the trace
moves the note far (choir_v6.TRACE_MIN; start notes are always traced). On tonight's v11 run the tracker heard the soft
E4 at beat 7.25 220 ms late; the follower took that as her slowing down and the chords at 9.25 and 10.5 came 0.12-0.17 s
late (in the afternoon the same note was dropped and they came early). Each run is replayed with the calibration it
was sung with (afternoon 130.5 ms round trip, tonight 174 ms). Phone runs: chord timing against the true onsets
(target -35 ms). Vincent's own a cappella runs have no reliable truth: the follower's own prediction errors only.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py choir_teval14 900 .venv/bin/python runs_mini/choir_1102/choir_timing_eval_v14.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402  (tripwire, page v13, truth, replay, NOTES, voice_stats)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

E = V.E
OUT = os.path.join(E.ABS, 'runs_mini/choir_1102/choir_timing_eval_v14.json')
LO = 0.0421
RUNS = {  # run: (song for the truth match or None for no truth, calibrated round trip)
    '195940_r2_mix_v2': ('v11', 0.174), '113335_r2_mix_v2': ('v11', 0.1305), '114908_r2_mix_v2': ('v11', 0.1305),
    '124132_r2_mix_v2': ('v11', 0.1305), '132458_r2_mix_v2': ('v11', 0.1305),
    '195914_r2_mix_v2': ('twinkle', 0.174), '113217_r2_mix_v2': ('twinkle', 0.1305),
    '114844_r2_mix_v2': ('twinkle', 0.1305),
    '110255_r2_mix_v2': (None, 0.1305), '095417_r2_mix_v2': (None, 0.1305)}
VARIANTS = {'S now (start notes only)': dict(TRACE_MIN=0.0), 'T60': dict(TRACE_MIN=0.06),
            'T100': dict(TRACE_MIN=0.10)}

if __name__ == '__main__':
    E.P3.MODELS.load()
    lib = E.P13.P12.P5.CHOIR
    res = {}
    for run, (sid, rt) in RUNS.items():
        E.LI, E.LO = rt - LO, LO
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        sung = E.true_notes(x)
        print(run, sid, flush=True)
        for vname, st in VARIANTS.items():
            V.NOTES.clear()
            eng, _ = E.replay(x, st)
            pe = np.abs(eng.diag['pred_err_ms'])
            r = {'pred_med': round(float(np.median(pe))) if len(pe) else None,
                 'pred_p90': round(float(np.percentile(pe, 90))) if len(pe) else None,
                 'late_traced': eng.diag.get('onset_traced_late', 0), 'song': eng.song_id}
            if sid:
                score = [(float(a), int(m) + (eng.S or 0)) for a, b, m in lib[sid]['melody']]
                truth = E.match(score, sung, lib[sid]['bpm'] if sid == 'v11' else 60.0 / eng.Tb)
                log = [(nt.i0, idx, song) for idx, song, nt in V.NOTES if nt is not None]
                r.update(E.score_run(eng, log, truth))
            r.update(V.voice_stats(eng))
            res[f'{run} | {vname}'] = r
            keep = ('chord_med_abs_from_target', 'chord_p90_abs_from_target', 'chord_median_ms', 'pred_med', 'pred_p90',
                    'late_traced', 'two_at_once', 'holes')
            print(f'   {vname:26s}', json.dumps({k: r.get(k) for k in keep}), flush=True)
            if sid == 'v11':
                print('      per chord:', sorted(set(r['per_chord'])), flush=True)
    for vname in VARIANTS:
        rows = {k.split(' | ')[0]: v for k, v in res.items() if k.endswith(vname)}
        v11 = [rows[r]['chord_med_abs_from_target'] for r in rows if RUNS[r][0] == 'v11']
        tw = [rows[r]['chord_med_abs_from_target'] for r in rows if RUNS[r][0] == 'twinkle']
        own = [rows[r]['pred_med'] for r in rows if RUNS[r][0] is None]
        print(f'{vname:26s} v11 chord |from target| medians {v11} mean {np.mean(v11):.0f}; twinkle {tw} '
              f'mean {np.mean(tw):.0f}; his own (prediction error) {own}', flush=True)
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1)
