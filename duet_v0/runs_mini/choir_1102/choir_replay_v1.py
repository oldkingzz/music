"""Replay Vincent's 09:49 choir run (runs_mini/live/094919_r2_mix_v2/run.wav, left = the microphone) through page v7's
choir (choir_v3), offline, no sound device, with the same calibration latency (Li 194 ms, Lo 42 ms). Reports where her
onsets fall against Duet's grid (v2 had a median of -104 ms: Duet late), the phase shifts v3 made, and the start.
Also a synthetic singer 80 ms ahead of the clicks.

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/choir_replay_v1.py
"""
import sys
import types

ABS = '/Users/san/claude_sandbox/music/duet_v0'


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('offline test: no sound device (sounddevice.' + name + ')')


for _m in ('sounddevice', '_sounddevice'):
    sys.modules[_m] = _NoSound(_m)

import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import live_panel_v7 as P7  # noqa: E402
P3, P5 = P7.P3, P7.P5
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import singer  # noqa: E402
from dsp import SR  # noqa: E402

ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_choir', time.strftime('%H%M%S') + '_v7')
OK = []


def check(what, ok, detail=''):
    OK.append(bool(ok))
    print(('OK   ' if ok else 'FAIL ') + what, detail, flush=True)


def run(x, lat, label):
    s = P3.Session(ROOT)
    s.cals[('m', 'p')] = {'in': 'm', 'out': 'p', 'rt_s': lat[0] + lat[1] - 0.020, 'rt_ms': (lat[0] + lat[1] - 0.02) * 1000,
                          'n_used': 8, 'spread_ms': 5, 'when': 'test', 'stream_Li_ms': 64.8, 'stream_Lo_ms': 42.1}
    s.start({'song': 'twinkle', 'key': 'C', 'bpm': 80.0, 'mode': 'choir', 'click': False, 'input': 'mic'},
            source=np.asarray(x, np.float32), sim={'devs': ('m', 'p'), 'latency': (0.0648, lat[1])})
    while s.state != 'await':
        time.sleep(0.05)
    eng = s.eng
    s.verdict({'run': s.status()['run'], 'verdict': '作废', 'note': 'choir v7 replay ' + label})
    time.sleep(1.1)
    errs = []
    for e in eng.events:
        sb = e.get('song_beat')
        if sb is None or sb < 0.5:
            continue
        k = int(np.argmin(np.abs(eng.mel_on - sb)))
        if abs(eng.mel_on[k] - sb) <= 0.35:
            errs.append((sb - eng.mel_on[k]) * eng.T_b * 1000)
    late = errs[6:]
    print(f'{label}: Li {eng.Li:.3f} Lo {eng.Lo:.3f} song_t0 {eng.song_t0} shifts {eng.shifts}')
    print(f'   her onset - grid (ms), notes 7+: median {np.median(late):.0f}, mean {np.mean(late):.0f}, n {len(late)}')
    return eng, late


P3.MODELS.load()
wav, sr = sf.read(os.path.join(ABS, 'runs_mini/live/094919_r2_mix_v2/run.wav'), dtype='float32', always_2d=True)
assert sr == SR
eng, late = run(wav[:, 0], (0.194, 0.0421), 'replay 09:49')
first = [e for e in eng.events if e.get('song_beat') == 0.0]
check('replay: the start is her first C, not the phone count-in clicks', first and first[0]['her_heard'] % 12 == 0,
      first[0]['her_heard'] if first else None)
check('replay: after 6 notes her onsets sit within 25 ms of Duet\'s grid (median)', abs(np.median(late)) <= 25,
      round(float(np.median(late))))
T_b = 0.75
T0 = 0.5 + 4 * T_b
mel = P5.CHOIR['twinkle']['melody'][:24]
notes = [(T0 + a * T_b - 0.080, T0 + b * T_b - 0.080, m) for a, b, m in mel]
x = np.concatenate([np.zeros(int(0.15 * SR), np.float32), singer.render(notes)[0], np.zeros(2 * SR, np.float32)])
eng, late = run(x, (0.15, 0.010), 'singer 80 ms ahead')
check('singer ahead: median within 25 ms after 6 notes', abs(np.median(late)) <= 25, round(float(np.median(late))))
print('ALL PASS' if all(OK) else 'SOME FAILED', len(OK))
