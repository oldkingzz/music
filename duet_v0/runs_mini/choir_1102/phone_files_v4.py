"""v4 of phone_files_v3.py: Ode to Joy for the phone with a voice-like 'ah' instead of dsp.render_voice's near-sine
'oo'. Through the AirPods mic the 'oo' files came in about 10 dB under the v11 phone runs (real singer), with the
first 7-9 s near silent (runs 205400, 211009): the AirPods voice processing takes a steady near-sine for noise. Here:
24 harmonics falling 1/k through two vowel formants (~750 and 1200 Hz), 5.5 Hz vibrato, a little breath noise, notes
parted as in v3, loud (peak 0.95). Writes futures/live_test/phone/ode_C_96bpm_lead3s_v4.wav / .mp3, new files only.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py phone_files_v4 120 .venv/bin/python runs_mini/choir_1102/phone_files_v4.py
"""
import json
import os
import sys

import numpy as np
import scipy.signal as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phone_files_v1 as P1  # noqa: E402

SR = P1.SR


def ah(notes, n):
    out = np.zeros(n)
    rng = np.random.default_rng(0)
    for start, dur, m in notes:
        i0, length = int(start * SR), int((dur + 0.06) * SR)
        t = np.arange(length) / SR
        vib = 0.2 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.12) / 0.2, 0, 1)
        f = 440.0 * 2 ** ((m + vib - 69) / 12)
        ph = 2 * np.pi * np.cumsum(f) / SR
        y = sum(np.sin(k * ph) / k for k in range(1, 25) if k * f.max() < SR / 2 - 500)
        y = y + 0.02 * rng.normal(0, 1, length)
        env = np.clip(t / 0.03, 0, 1) * np.clip((dur + 0.05 - t) / 0.05, 0, 1)
        i1 = min(n, i0 + length)
        out[i0:i1] += (y * env)[:i1 - i0]
    v = np.zeros(n)
    for fc, bw, g in ((750, 150, 1.0), (1200, 200, 0.6), (2600, 300, 0.25)):
        b, a = ss.iirpeak(fc, fc / bw, fs=SR)
        v += g * ss.lfilter(b, a, out)
    return v + 0.15 * out


if __name__ == '__main__':
    with open(os.path.join(P1.ABS, 'futures/v4_selfiter/choir_songs_v7.json')) as fh:
        ode = json.load(fh)['songs']['ode']
    sb = 60.0 / ode['bpm']
    notes = [(a * sb, (b - a) * sb - 0.15, m) for a, b, m in ode['melody']]
    P1.PEAK = 0.95
    P1.write('ode_C_96bpm_lead3s_v4', ah(notes, int((ode['K'] * sb + 0.5) * SR)))
