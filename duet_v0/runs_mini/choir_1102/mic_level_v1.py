"""Mic level of tonight's runs (run choir_1102): the loud part (90th percentile of 100 ms RMS, dB) of each run's mic
channel, phone files (real singer v11, synthetic Twinkle / Ode) against Vincent singing. Plain file reads, no engine.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py mic_level_v1 120 .venv/bin/python runs_mini/choir_1102/mic_level_v1.py
"""
import json
import os

import numpy as np
import soundfile as sf

LIVE = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/live'
for run in ('110255_r2_mix_v2', '195914_r2_mix_v2', '195940_r2_mix_v2', '203926_r2_mix_v2', '205400_r2_mix_v2',
            '205440_r2_mix_v2', '211009_r2_mix_v2'):
    x = sf.read(os.path.join(LIVE, run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
    n = len(x) // 2400
    r = 20 * np.log10(np.sqrt(np.mean(x[:n * 2400].reshape(n, 2400) ** 2, axis=1)) + 1e-9)
    song = json.load(open(os.path.join(LIVE, run, 'notes.json'))).get('choir_song')
    print(run, song, 'loud part %.1f dB, median %.1f dB' % (np.percentile(r, 90), np.median(r)), flush=True)
