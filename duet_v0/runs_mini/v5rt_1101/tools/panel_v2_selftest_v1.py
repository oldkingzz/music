"""Offline self-test of futures/v4_selfiter/live_panel_v2.py with NO sound device (Vincent asleep, 2026-11-02 night).

Installs live_rt_eval_v2's sounddevice tripwire before anything is imported (any public sounddevice call raises), then
runs live_panel_v2.selftest(): the phone files are fed through the page's own Session code 10 ms at a time; runs go
to runs_mini/live/selftest_v2/ and are rated 作废.

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python runs_mini/v5rt_1101/tools/panel_v2_selftest_v1.py
"""
import os
import sys
import types

ABS = '/Users/san/claude_sandbox/music/duet_v0'


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('panel selftest is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import live_panel_v2 as P  # noqa: E402

if __name__ == '__main__':
    P.selftest()
