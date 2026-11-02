"""v3: the whole song (48 beats), logging unmatched notes. v2 of debug_follow_v1.py: the Vincent-like singer (A2, intervals x0.85, 30 cents flat, 88 bpm). Debug choir_v5 recognition on a clean synthetic Twinkle (C, 80 bpm): her segmented notes and, at each recognition
attempt, the best hypotheses. Feeds the engine directly (no page), tripwire first.

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/debug_follow_v1.py
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

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import live_bench3  # noqa: E402,F401
import choir_v5 as CV5  # noqa: E402
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import singer  # noqa: E402
from dsp import HOP, SR  # noqa: E402

songs = CV5.load_songs()
orig = CV5.ChoirFollow._recognise


def rec(self):
    hd = self._her_distinct()
    if len(hd) >= 3 and self.arr is None:
        h = [n['pitch'] for n in hd]
        hyps = []
        for sid, dist in self.dist.items():
            d = [m for b, m in dist]
            for T in range(12):
                for u in self.TUNINGS:
                    cost, pairs = self._align(h, d, T + u)
                    hyps.append((round(cost, 2), sid, T, u, len(pairs), pairs[-1] if pairs else None))
        hyps.sort()
        print('  her distinct', [round(x, 2) for x in h])
        for x in hyps[:5]:
            print('    ', x)
    orig(self)


CV5.ChoirFollow._recognise = rec
_fol = CV5.ChoirFollow._follow


def fol(self, note):
    n0, m0 = self.unmatched, self.mpos
    _fol(self, note)
    if self.unmatched > n0 or self.diag['relocations']:
        print('  note', round(note['t'] - 3.5, 2), round(note['pitch'], 2), 'mpos', m0, '->', self.mpos, 'unmatched', self.unmatched,
              'reloc', self.diag['relocations'], 'expect', [self.mel[k][1] + self.T + self.u for k in range(m0 + 1, min(m0 + 4, len(self.mel)))])


CV5.ChoirFollow._follow = fol
eng = CV5.ChoirFollow(songs, 80.0, log=print)
eng.Li, eng.Lo = 0.15, 0.01
T_b = 0.75
T0 = 0.5 + 4 * T_b
T_b = 60 / 88.0
notes = [(T0 + 0.75 + a * T_b, T0 + 0.75 + b * T_b, 60 - 15 - 0.3 + 0.85 * (m - 60)) for a, b, m in songs['twinkle']['melody']]
x = np.concatenate([np.zeros(int(0.15 * SR), np.float32), singer.render(notes)[0], np.zeros(SR, np.float32)])
for k in range(len(x) // HOP):
    eng.process(x[k * HOP:(k + 1) * HOP])
    if False:
        break
print('her notes', [(round(n['t'] - T0, 2), round(n['pitch'], 2)) for n in eng.hnotes[:20]])
print('song', eng.song_id, eng.S, eng.u, eng.low, 60 / eng.Tb)
print('pairs', [(round(t - T0, 2), b) for t, b in eng.pairs])
print('pred err', eng.diag['pred_err_ms'], 'reloc', eng.diag['relocations'], 'u track', eng.diag['tune_track'])
