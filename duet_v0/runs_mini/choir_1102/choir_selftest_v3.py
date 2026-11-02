"""Offline test of page v6's choir (choir_v2: waits for her first note), no sound device (tripwire first).
Cases (Twinkle, C, 80 bpm, stream latency 63 / 10 ms, true mic delay 150 ms):
 a) she never sings: Duet's output has the 4 count-in clicks and nothing else; the run ends by itself.
 b) she starts 4 beats late: no part sounds before her first note; the parts' second chord is written on her grid
    (song beat 0 = her first onset), and parts sound while she sings.
 c) she sings 8 beats and stops: the parts are silent from about 3 beats after her last note; the run ends.

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/choir_selftest_v3.py
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

sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import live_panel_v6 as P6  # noqa: E402
P3, P5 = P6.P3, P6.P5
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import singer  # noqa: E402
from dsp import SR  # noqa: E402

ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_choir', time.strftime('%H%M%S') + '_v6')
STREAM, LI = (0.063, 0.010), 0.150
BPM = 80.0
T_b = 60.0 / BPM
T0 = 0.5 + 4 * T_b                     # the click grid's beat 0
OK = []


def check(what, ok, detail=''):
    OK.append(bool(ok))
    print(('OK   ' if ok else 'FAIL ') + what, detail, flush=True)


def her(start_beat, n_beats):
    mel = [(a, b, m) for a, b, m in P5.CHOIR['twinkle']['melody'] if b <= n_beats]
    notes = [(T0 + (start_beat + a) * T_b, T0 + (start_beat + b) * T_b, m) for a, b, m in mel]
    return singer.render(notes)[0]


def run(x, label):
    s = P3.Session(ROOT)
    x = np.concatenate([np.zeros(int(LI * SR), np.float32), np.asarray(x, np.float32)])
    s.start({'song': 'twinkle', 'key': 'C', 'bpm': BPM, 'mode': 'choir', 'click': False, 'input': 'mic'},
            source=x, sim={'devs': ('m', 'p'), 'latency': STREAM})
    t_wait = time.time()
    while s.state != 'await' and time.time() - t_wait < 120:
        time.sleep(0.05)
    eng = s.eng
    out = np.concatenate(s.rec_out)
    s.verdict({'run': s.status()['run'], 'verdict': '作废', 'note': 'choir v6 selftest ' + label})
    time.sleep(1.1)
    return eng, out, s


def env(out, t0, t1):
    a, b = int(max(t0, 0) * SR), int(t1 * SR)
    return float(np.max(np.abs(out[a:b]))) if b > a else 0.0


P3.MODELS.load()
# a) silence
eng, out, s = run(np.zeros(int((T0 + 20 * T_b) * SR), np.float32), 'silence')
check('a) never sings: nothing after the count-in', env(out, T0 - 0.05, len(out) / SR) < 1e-3, env(out, T0 - 0.05, len(out) / SR))
check('a) the run ended by itself, song never started', eng.song_t0 is None and s.state == 'idle')
# b) late start
x = np.concatenate([her(4, 16), np.zeros(int(2 * SR), np.float32)])
eng, out, s = run(x, 'late start')
first = T0 + 4 * T_b
check('b) no part before her first note', env(out, T0 + 0.05, first - 0.05) < 1e-3, env(out, T0 + 0.05, first - 0.05))
check('b) song beat 0 = her first onset on the click grid', eng.song_t0 is not None and abs(eng.song_t0 - first) < 0.01,
      eng.song_t0)
check('b) parts sound while she sings', env(out, first + 2 * T_b, first + 10 * T_b) > 0.1)
w2 = int(round((eng.song_t0 + 4 * T_b - eng.Lo - eng.PRE) * SR))       # the second chord (IV at song beat 4)
pre = env(out, (w2 - int(0.004 * SR)) / SR, w2 / SR)
print('   second chord write sample', w2, 'first part sounded', round(first, 3))
# c) stop after 8 beats
x = np.concatenate([her(0, 8), np.zeros(int(8 * SR), np.float32)])
eng, out, s = run(x, 'stops')
stop_by = T0 + (8 + 3) * T_b + 0.6
check('c) parts silent ~3 beats after she stops', env(out, stop_by, len(out) / SR) < 1e-3 and eng.stopped == 'she_stopped',
      (env(out, stop_by, len(out) / SR), eng.stopped))
check('c) parts sounded while she sang', env(out, T0 + 1 * T_b, T0 + 7 * T_b) > 0.1)
print('ALL PASS' if all(OK) else 'SOME FAILED', len(OK))
