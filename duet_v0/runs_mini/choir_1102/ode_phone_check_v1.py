"""Quick offline check of the Ode phone files (run choir_1102): v1 (30 ms breaths) and v3 (notes parted by ~90 ms),
each clean and through a rough stand-in for phone speaker -> room -> AirPods mic (300-3400 Hz band, 250 ms room
decay, noise at -40 dB). Per file: song recognised, follower's tempo, chords sung and their timing (target -35 ms).
Offline (tripwire), fixed 174 ms round trip.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py ode_phone_check_v1 900 .venv/bin/python runs_mini/choir_1102/ode_phone_check_v1.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selftest_panel_v15 as S  # noqa: E402  (tripwire, three-song page, run())

import numpy as np  # noqa: E402
import scipy.signal as ss  # noqa: E402
import soundfile as sf  # noqa: E402

E = S.E


def room(x):
    b, a = ss.butter(2, [300, 3400], btype='band', fs=E.SR)
    y = ss.lfilter(b, a, x)
    t = np.arange(int(0.25 * E.SR)) / E.SR
    ir = np.exp(-t / 0.06) * np.random.default_rng(0).normal(0, 1, len(t)) * 0.3
    ir[0] = 1.0
    y = ss.fftconvolve(y, ir)[:len(x)]
    y = y / np.abs(y).max() * 0.5 + np.random.default_rng(1).normal(0, 0.005, len(y))
    return y.astype(np.float32)


if __name__ == '__main__':
    E.P3.MODELS.load()
    for name in ('ode_C_96bpm_lead3s', 'ode_C_96bpm_lead3s_v3'):
        x = sf.read(os.path.join(E.ABS, 'futures/live_test/phone', name + '.wav'), dtype='float32',
                    always_2d=True)[0][:, 0]
        for tag, y in (('clean', x), ('room', room(x))):
            r = S.run(y, 'ode', 0.174)
            print(name, tag, json.dumps({k: r.get(k) for k in ('song', 'chords', 'chord_med_abs_from_target',
                                                               'chord_p90_abs_from_target', 'semitone_share')}),
                  flush=True)
