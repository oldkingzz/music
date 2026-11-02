"""Offline check of page v11's watchdog (no sound device, tripwire first): a Session in state 'running' whose audio
callback stopped 2 s ago is ended with the device message; one whose callback keeps running is left alone; and a full
offline choir run through v11 still works (Twinkle phone recording replay).

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/watchdog_test_v1.py
"""
import sys
import threading
import time
import types

ABS = '/Users/san/claude_sandbox/music/duet_v0'


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('offline (sounddevice.' + name + ')')


for _m in ('sounddevice', '_sounddevice'):
    sys.modules[_m] = _NoSound(_m)

import os  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import live_panel_v11 as P11  # noqa: E402
P3 = P11.P3
OK = []


def check(what, ok, detail=''):
    OK.append(bool(ok))
    print(('OK   ' if ok else 'FAIL ') + what, detail, flush=True)


ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_choir', time.strftime('%H%M%S') + '_v11')
s = P3.Session(ROOT)
s.state, s.done, s.last_cb = 'running', threading.Event(), time.time() - 2.0
fired = P11.watchdog(s)
check('stalled callback: the run is ended with the device message', fired and s.done.is_set() and '断开' in s.msg)
s2 = P3.Session(ROOT)
s2.state, s2.done, s2.last_cb = 'running', threading.Event(), time.time()


def ticker():
    for _ in range(15):
        s2.last_cb = time.time()
        time.sleep(0.1)
    s2.state = 'await'


threading.Thread(target=ticker, daemon=True).start()
check('a live callback keeps the run going', not P11.watchdog(s2) and not s2.done.is_set() and s2.msg == '')
P3.MODELS.load()
w, sr = sf.read(os.path.join(ABS, 'runs_mini/live/110335_r2_mix_v2/run.wav'), dtype='float32', always_2d=True)
s3 = P3.Session(ROOT)
s3.cals[('m', 'p')] = {'in': 'm', 'out': 'p', 'rt_s': 0.1305, 'rt_ms': 130.5, 'n_used': 8, 'spread_ms': 5, 'when': 't',
                       'stream_Li_ms': 62.7, 'stream_Lo_ms': 42.1}
s3.start({'song': 'auto', 'bpm': 80.0, 'mode': 'choir', 'click': False, 'input': 'mic'}, source=w[:, 0],
         sim={'devs': ('m', 'p'), 'latency': (0.0627, 0.0421)})
t0 = time.time()
while s3.state != 'await' and time.time() - t0 < 120:
    time.sleep(0.05)
eng = s3.eng
out = np.concatenate(s3.rec_out)
check('offline choir run through v11: Twinkle in C, parts sounded', eng.song_id == 'twinkle' and eng.S == 0
      and float(np.max(np.abs(out))) > 0.1, (eng.song_id, eng.S, round(float(np.max(np.abs(out))), 3)))
s3.verdict({'run': s3.status()['run'], 'verdict': '作废', 'note': 'v11 watchdog test'})
print('ALL PASS' if all(OK) else 'SOME FAILED', len(OK))
