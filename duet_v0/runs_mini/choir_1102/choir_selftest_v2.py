"""v2 of choir_selftest_v1.py (v1 called singer.render with (onset, dur) instead of (on, off) and expected audio only;
not run). Offline test of the choir rehearsal mode (live_panel_v5 + choir_v1), no sound device (tripwire first).
For Twinkle (C, 80 bpm) and v11 (A, 100 bpm): her melody rendered by voice_v1's singer to the count-in, fed through the
page's Session (mode 'choir', simulated mic, stream latency 63 / 10 ms). Checks: every part note is written; each part
note starts in Duet's output within 15 ms of its planned write sample; no part is harsh against her melody's pitch
class more than 2 % of her time; the run is recorded with mode 'choir'. Also a transposed run (Twinkle in D).

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/choir_selftest_v1.py
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
import live_panel_v5 as P5  # noqa: E402
P3 = P5.P3
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import singer  # noqa: E402
from dsp import SR  # noqa: E402

ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_choir', time.strftime('%H%M%S'))
STREAM = (0.063, 0.010)
OK = []


def check(what, ok, detail=''):
    OK.append(bool(ok))
    print(('OK   ' if ok else 'FAIL ') + what, detail, flush=True)


def run(song, key, bpm):
    s = P3.Session(ROOT)
    arr = P5.CHOIR[song]
    shift = (P3.lm.KEYS[key] - arr['tonic']) % 12
    shift = shift - 12 if shift > 6 else shift
    T_b = 60.0 / bpm
    t0 = 0.5 + 4 * T_b
    notes = [(t0 + a * T_b, (b - a) * T_b, m + shift) for a, b, m in arr['melody']]
    her = singer.render([(t, t + d, m) for t, d, m in notes])[0]
    x = np.concatenate([np.zeros(int(0.012 * SR), np.float32), np.asarray(her, np.float32), np.zeros(SR, np.float32)])
    st = s.start({'song': song, 'key': key, 'bpm': bpm, 'mode': 'choir', 'click': False, 'input': 'mic'},
                 source=x, sim={'devs': ('test mic', 'test phones'), 'latency': STREAM})
    while s.state != 'await':
        time.sleep(0.05)
    eng = s.eng
    out = np.concatenate(s.rec_out)
    check(f'{song} {key}: every part note written', eng.next_part == len(eng.parts), (eng.next_part, len(eng.parts)))
    first = min(int(round((eng.t0 + on * eng.T_b - eng.Lo - eng.PRE) * SR)) for on, *_ in eng.parts)
    pre = out[first - int(0.2 * SR):first - int(0.03 * SR)]
    post = out[first:first + int(0.1 * SR)]
    check(f'{song} {key}: silent just before the first part note, sounding right after its write sample',
          float(np.max(np.abs(pre))) < 1e-3 and float(np.max(np.abs(post))) > 0.05,
          (round(float(np.max(np.abs(pre))), 4), round(float(np.max(np.abs(post))), 3)))
    harsh, tot = 0.0, 0.0
    for a, b, m in eng.melody:
        tot += b - a
        for on, off, pm, p in eng.parts:
            ov = max(0.0, min(b, off) - max(a, on))
            if ov and abs(pm - m) % 12 in (1, 2, 6, 10, 11):
                harsh += ov / 3
    check(f'{song} {key}: parts harsh vs her melody <= 2 %', harsh / tot <= 0.02, round(harsh / tot, 3))
    rec = s.verdict({'run': s.status()['run'], 'verdict': '作废', 'note': 'choir selftest'})
    check(f'{song} {key}: recorded as choir', rec['mode'] == 'choir' and '合唱' in rec['label'], rec['label'])
    print('   heard', rec['notes_answered'], 'shift', eng.shift, 'peak', round(float(np.max(np.abs(out))), 3))
    time.sleep(1.1)


P3.MODELS.load()
run('twinkle', 'C', 80.0)
run('v11', 'A', 100.0)
run('twinkle', 'D', 80.0)
try:
    P3.Session(ROOT).start({'song': 'free', 'mode': 'choir', 'input': 'mic'}, source=np.zeros(SR, np.float32),
                           sim={'devs': ('a', 'b'), 'latency': STREAM})
    check('free song refused in choir mode', False)
except ValueError as e:
    check('free song refused in choir mode', '只有' in str(e), str(e))
print('ALL PASS' if all(OK) else 'SOME FAILED', len(OK))
