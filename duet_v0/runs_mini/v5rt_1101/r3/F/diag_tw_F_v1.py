"""Round 3, idea F DIAGNOSTIC (not a scored bench): every hear() call of an idea-F system on a phone file, with the path
it took (stats counters), the copy model's prediction for that note and, when the posterior hear fired, its top true
pitches. Offline only: the _NoSound tripwire is installed before any other import; nothing is played.

  .venv/bin/python runs_mini/v5rt_1101/r3/F/diag_tw_F_v1.py --cfg '<json>' [--file phone_twinkle|phone_v11]
      [--lib r3_F_lib_v5 --cls PlanF5] --out <new json under runs_mini/v5rt_1101/r3/F/>
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('offline diagnostic: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from collections import Counter  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r3', 'F')
sys.path.insert(0, HERE)
import tune_F_v1 as T1  # noqa: E402

import numpy as np  # noqa: E402

LBR = T1.LBR


def logged(Cls):
    class L(Cls):
        def start(self, meta):
            super().start(meta)
            self.log = []
            self._post = None

        def _hear_post(self, i, onset, midi, prev_dur, post):
            j = np.argsort(-post)[:3]
            self._post = [[int(midi + T1.FL.E_ALL[k]), round(float(post[k]), 3)] for k in j]
            return super()._hear_post(i, onset, midi, prev_dur, post)

        def hear(self, i, onset, midi, prev_dur):
            b = Counter(self.stats)
            hs = self.h
            self._post = None
            pred = self.copy.pred if self.copy is not None else None
            r = super().hear(i, onset, midi, prev_dur)
            self.log.append({'i': i, 'onset': onset, 'heard': int(midi), 'before': hs, 'result': r, 'after': self.h,
                             'path': sorted(Counter(self.stats) - b), 'copy_top': None if pred is None else pred['top'],
                             'copy_best': None if pred is None else pred['best'], 'post': self._post})
            return r
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--file', default='phone_twinkle', choices=('phone_twinkle', 'phone_v11'))
    ap.add_argument('--lib', default='r3_F_lib_v5')
    ap.add_argument('--cls', default='PlanF5')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(HERE + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit(f'--out must be a new .json under {HERE}/')
    Cls = logged(getattr(importlib.import_module('rt_systems.' + a.lib), a.cls))
    train = T1.B.bench3.train_sets()
    mix = T1.r2_mix_v2.build(train)
    obj = Cls(mix, T1.Prior(), dict(T1.BASE, **json.loads(a.cfg)), T1.load_copy_tab(), T1.LIB3.load_hm_tab())
    out, st = LBR.phone_run(obj, 'diag', a.file)
    res = {'cfg': json.loads(a.cfg), 'file': a.file, 'hear_log': obj.log, 'answered': out['answered'],
           'metrics': out.get('metrics'), 'proxies': out['proxies'], 'fresh_clashes': out.get('fresh_clashes')}
    for e in obj.log:
        print(json.dumps(e), flush=True)
    print('metrics', json.dumps(out.get('metrics')), 'fresh', json.dumps(out.get('fresh_clashes')), flush=True)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1, default=LBR._js)


if __name__ == '__main__':
    main()
