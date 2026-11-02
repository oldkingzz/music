"""Debug choir_v5 on Vincent's a cappella run 095417 (AirPods Max mic): his notes, every match (her onset -> melody
beat, prediction error), unmatched notes and the beat map over time. Engine fed directly, tripwire first.

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/debug_replay_v1.py [run]
"""
import sys
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

import soundfile as sf  # noqa: E402

sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import live_bench3  # noqa: E402,F401
import choir_v5 as CV5  # noqa: E402
from dsp import HOP  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else '095417_r2_mix_v2'
x = sf.read(os.path.join(ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
eng = CV5.ChoirFollow(CV5.load_songs(), 80.0, log=print)
eng.Li, eng.Lo = 0.17, 0.042
_fol = CV5.ChoirFollow._follow


def fol(self, note):
    m0, n0 = self.mpos, len(self.pairs)
    _fol(self, note)
    if len(self.pairs) > n0:
        t, b = self.pairs[-1]
        print(f'  {note["t"]:6.2f} {note["pitch"]:6.2f} -> beat {b:5.1f} (mel {self.mel[self.mpos][1]}) '
              f'err {self.diag["pred_err_ms"][-1]:7.1f} ms, tempo {60 / self.Tb:5.1f}')
    else:
        print(f'  {note["t"]:6.2f} {note["pitch"]:6.2f}    no match (at beat {self.mel[m0][0]}, unmatched {self.unmatched})')


CV5.ChoirFollow._follow = fol
for k in range(len(x) // HOP):
    eng.process(x[k * HOP:(k + 1) * HOP])
print('song', eng.song_id, 'shift', eng.S, 'tuning', round(eng.u, 2), 'low', eng.low, 'stopped', eng.stopped)
print('her notes', [(round(n['t'], 2), round(n['pitch'], 2)) for n in eng.hnotes])
