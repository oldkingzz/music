"""Idea D dev runs (run v5rt_1101, round 3): dev_v3.py's live / bench modes (imported, unchanged) with any adapter
library rt_systems.r3_D_lib_* (dev_v3 only accepted v1 / v2). Offline: the _NoSound tripwire is installed before any other
import (dev_v3 installs its own too); nothing is played. Val only, tuned on val (no test split).

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/D/dev_v4.py --lib r3_D_lib_v3 --mode live|bench --runs '<json>' \
      [--voices ...] [--songs N] [--seed2] --out runs_mini/v5rt_1101/r3/D/<new>.json
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('dev_v4 is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/r3/D')
import dev_v3 as D3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', required=True)
    ap.add_argument('--mode', required=True, choices=('live', 'bench'))
    ap.add_argument('--runs', required=True)
    ap.add_argument('--voices', default='sung_low')
    ap.add_argument('--songs', type=int, default=None)
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if not a.lib.startswith('r3_D_lib_'):
        sys.exit('--lib must be an r3_D_lib_* module')
    path = os.path.abspath(a.out)
    if os.path.exists(path) or not path.endswith('.json'):
        sys.exit('refusing: ' + path)
    D3.DL = importlib.import_module('rt_systems.' + a.lib)
    runs = json.loads(a.runs)
    if a.mode == 'bench':
        D3.BR.bench3.B2.heard = D3.BR.heard_seed2 if a.seed2 else D3.BR.HEARD1
    train = D3.BR.bench3.train_sets()
    rows = D3.live(runs, a.voices.split(','), a.songs, train) if a.mode == 'live' else D3.bench(runs, a.seed2, train)
    with open(path, 'x') as fh:
        json.dump({'note': 'idea D dev runs; val only, tuned on val, no test split', 'lib': a.lib, 'mode': a.mode,
                   'rows': rows}, fh, indent=1, default=D3.LBR._js)
    print('wrote', path)


if __name__ == '__main__':
    main()
