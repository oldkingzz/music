"""Offline check of page v12 (no sound device, tripwire first): choir runs with the metronome box off play no count-in
clicks (Duet's output is silent until the parts come in) and still find song and key; with the box on the clicks are
there. Replays Vincent's 11:32 Twinkle phone run and 11:33 v11 phone run with the latest calibration.

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/v12_check_v1.py
"""
import sys
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
import live_panel_v12 as P12  # noqa: E402
P3 = P12.P3
from dsp import SR  # noqa: E402

ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_choir', time.strftime('%H%M%S') + '_v12')
OK = []


def check(what, ok, detail=''):
    OK.append(bool(ok))
    print(('OK   ' if ok else 'FAIL ') + what, detail, flush=True)


def run(run_name, click, bpm):
    w, sr = sf.read(os.path.join(ABS, 'runs_mini/live', run_name, 'run.wav'), dtype='float32', always_2d=True)
    s = P3.Session(ROOT)
    s.cals[('m', 'p')] = {'in': 'm', 'out': 'p', 'rt_s': 0.1305, 'rt_ms': 130.5, 'n_used': 8, 'spread_ms': 5,
                          'when': 't', 'stream_Li_ms': 62.7, 'stream_Lo_ms': 42.1}
    s.start({'song': 'auto', 'bpm': bpm, 'mode': 'choir', 'click': click, 'input': 'mic'}, source=w[:, 0],
            sim={'devs': ('m', 'p'), 'latency': (0.0627, 0.0421)})
    t0 = time.time()
    while s.state != 'await' and time.time() - t0 < 120:
        time.sleep(0.05)
    eng, out = s.eng, np.concatenate(s.rec_out)
    s.verdict({'run': s.status()['run'], 'verdict': '作废', 'note': 'v12 check'})
    time.sleep(1.1)
    return eng, out


P3.MODELS.load()
for run_name, want in (('113217_r2_mix_v2', ('twinkle', 0)), ('113335_r2_mix_v2', ('v11', 0))):
    eng, out = run(run_name, False, 80.0)
    first = int(eng.diag.get('prov_time', eng.diag.get('id_time', 0)) * SR) - int(0.05 * SR)
    check(f'{run_name}, no metronome: {want[0]} in shift {want[1]}', (eng.song_id, eng.S) == want, (eng.song_id, eng.S))
    check(f'{run_name}, no metronome: Duet silent before the parts (no count-in)',
          float(np.max(np.abs(out[:max(first, 1)]))) < 1e-4, round(float(np.max(np.abs(out[:max(first, 1)]))), 4))
    check(f'{run_name}, no metronome: parts sound', float(np.max(np.abs(out))) > 0.1)
eng, out = run('113217_r2_mix_v2', True, 80.0)
click_zone = out[int((eng.click_t0 - 4 * eng.T0_b - eng.Lo) * SR): int((eng.click_t0 - eng.Lo) * SR)]
check('with the metronome box: the count-in clicks are there', float(np.max(np.abs(click_zone))) > 0.1,
      round(float(np.max(np.abs(click_zone))), 3))
print('ALL PASS' if all(OK) else 'SOME FAILED', len(OK))
