"""Where would tracing every note's onset move it, and is a big move right (run choir_1102)? Replays each run with
choir_v6.TRACE_ALL on and lists every note the trace moved: when the live tracker heard it, where the trace put it,
the move, and both against the nearest true onset (offline pyin, minus the input latency). A trace that is only used
when it moves the onset a lot (a soft attack the tracker heard late) is right if big moves land near the true onset.
Latency: the 19:59 calibration for tonight's runs, the 11:02 one for the afternoon's (the runs' own).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py trace_moves_v1 900 .venv/bin/python runs_mini/choir_1102/trace_moves_v1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402  (tripwire, page v13, truth, replay)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

E = V.E
RUNS = {'195940_r2_mix_v2': 0.1740, '113335_r2_mix_v2': 0.1305, '114908_r2_mix_v2': 0.1305,
        '124132_r2_mix_v2': 0.1305, '132458_r2_mix_v2': 0.1305, '195914_r2_mix_v2': 0.1740,
        '113217_r2_mix_v2': 0.1305, '114844_r2_mix_v2': 0.1305, '110255_r2_mix_v2': 0.1305,
        '095417_r2_mix_v2': 0.1305}

if __name__ == '__main__':
    E.P3.MODELS.load()
    big = []
    for run, rt in RUNS.items():
        E.LI, E.LO = rt - 0.0421, 0.0421
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        sung = E.true_notes(x)
        eng, _ = E.replay(x, {'TRACE_ALL': True})
        print(f'######## {run} ({eng.song_id})', flush=True)
        for n in eng.hnotes:
            if 't_heard' not in n:
                continue
            near = min(sung, key=lambda s: abs(s[0] - n['t'])) if sung else (n['t'], 0, 0)
            mv = (n['t_heard'] - n['t']) * 1000
            row = (run[:6], round(n['t_heard'], 2), round(mv), round((n['t'] - near[0]) * 1000),
                   round((n['t_heard'] - near[0]) * 1000), round(n['pitch'], 1))
            print('   heard %7.2f  moved %+5d ms  traced-true %+5d  heard-true %+5d  pitch %5.1f' % row[1:], flush=True)
            big.append(row)
    a = np.array([(r[2], r[3], r[4]) for r in big], float)
    for lo, hi in ((0, 60), (60, 100), (100, 150), (150, 400)):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.any():
            print(f'moved {lo}-{hi} ms: {int(m.sum())} notes, |traced-true| median {np.median(np.abs(a[m, 1])):.0f} ms, '
                  f'|heard-true| median {np.median(np.abs(a[m, 2])):.0f} ms, traced-true > 60 ms off: '
                  f'{int((np.abs(a[m, 1]) > 60).sum())}', flush=True)
