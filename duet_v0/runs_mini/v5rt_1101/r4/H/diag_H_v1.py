"""Idea H: phase 1's latency diagnostic (runs_mini/v5rt_1101/r4/B/diag_lat_v1.py, imported unchanged, read its
docstring) with its output directory pointed at runs_mini/v5rt_1101/r4/H/ (its main() only accepts new files under its
OUT_DIR). Offline, the _NoSound tripwire is installed before any other import; nothing is played. Val only, tuned on val
(no test split). Same CLI as diag_lat_v1.py.
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

import os  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r4', 'B'))
import diag_lat_v1 as D  # noqa: E402

D.OUT_DIR = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r4', 'H')

if __name__ == '__main__':
    D.main()
