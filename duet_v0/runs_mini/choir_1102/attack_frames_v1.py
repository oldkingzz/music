"""What did the live tracker see during v11's late-heard attacks? (run choir_1102) Recomputes the live FeatureTracker's
per-frame YIN values on each run's mic channel (same window, hop and voicing rule: ap < 0.25 and level > running
peak - 30 dB) and prints, around each true onset that was heard late, every frame's pitch, aperiodicity and level
below the running peak, so a hysteresis rule (looser voicing to extend a note's onset back) can be chosen from data.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py attack1 300 .venv/bin/python runs_mini/choir_1102/attack_frames_v1.py
"""
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, ABS)
sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import follow as F  # noqa: E402
from dsp import HOP, SR, hz_to_midi  # noqa: E402

LI = 0.1305 - 0.0421
WINDOWS = {'113335_r2_mix_v2': [('A', 5.452), ('E', 6.292)], '114908_r2_mix_v2': [('A', 3.842), ('E', 4.822)],
           '124132_r2_mix_v2': [('A', 3.892), ('E', 4.782)], '132458_r2_mix_v2': [('A', 4.492), ('E', 5.452)]}
for run, wins in WINDOWS.items():
    x = sf.read(os.path.join(ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
    buf, peak, rows = np.zeros(F.WIN), -80.0, []
    for k in range(len(x) // HOP):
        buf = np.concatenate([buf[HOP:], x[k * HOP:(k + 1) * HOP]])
        f0, ap, rms = F.yin(buf)
        db = 20 * np.log10(rms + 1e-9)
        peak = max(db, peak - 0.02)
        v = ap < 0.25 and db > peak - 30 and db > -60 and F.FMIN <= f0 <= F.FMAX
        rows.append(((k + 1) * HOP / SR - F.CONTENT_DELAY - LI, float(hz_to_midi(f0)) if f0 > 0 else 0.0, ap,
                     db - peak, v))
    for name, t0 in wins:
        print(f'{run} {name} (true onset {t0}):')
        print('   ' + '  '.join(f'{t:.2f}:{"V" if v else "-"}{m:.1f}/ap{ap:.2f}/{d:+.0f}dB' for t, m, ap, d, v in rows
                                if t0 - 0.06 <= t <= t0 + 0.22))
