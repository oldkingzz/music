"""Builds a static mock of the v3 page for a render check in a browser, with NO server and NO sound device.

Real Session.status() dicts from futures/v4_selfiter/live_panel_v3.py, produced offline (tripwire installed first;
calibrations fed latcal_v1.make_test recordings, a song fed a rendered Twinkle, everything rated 作废 into
runs_mini/live/selftest_v3/mock_<HHMMSS>/), are embedded in a copy of live_panel_v3.html whose window.fetch is replaced by
a stub that answers /api/state, /api/devices, /api/runs, /api/calibrate, /api/start, /api/stop, /api/verdict from them.
window.__mock.cur picks the state shown. Output (new file): runs_mini/v5rt_1101/r4/P/page_mock_v1.html.

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python runs_mini/v5rt_1101/tools/page_mock_v1.py
"""
import sys
import types

ABS = '/Users/san/claude_sandbox/music/duet_v0'


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('page mock is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

V4 = os.path.join(ABS, 'futures', 'v4_selfiter')
sys.path.insert(0, V4)
import live_panel_v3 as P3  # noqa: E402
import latcal_v1 as LC  # noqa: E402
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import voice_v1 as VV  # noqa: E402
from songs import twinkle  # noqa: E402
from dsp import SR, HOP  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r4', 'P', 'page_mock_v1.html')
ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_v3', 'mock_' + time.strftime('%H%M%S'))
STREAM = (0.063, 0.010)
PAIR_A = ('AirPods Pro', 'External Headphones')
PAIR_B = ('MacBook Mic', 'External Headphones')


def wait(s, states):
    t0 = time.time()
    while s.state not in states and time.time() - t0 < 120:
        time.sleep(0.02)
    assert s.state in states, s.state


def snap(s):
    return json.loads(json.dumps(s.status(), default=float, ensure_ascii=False))


def main():
    if os.path.exists(OUT):
        sys.exit(f'refusing to overwrite {OUT}')
    P3.MODELS.load()
    assert P3.MODELS.err is None, P3.MODELS.err
    st = {}
    s = P3.Session(os.path.join(ROOT, 'main'))
    st['idle_nocal'] = snap(s)
    # calibrating, half way: a CalRun fed by hand up to the third response click
    t = LC.make_test(0.160, 31)
    cal = P3.CalRun()
    s.eng, s.devs, s.state, s.started = cal, PAIR_A, 'calibrating', time.time()
    stop = cal.sched['clicks'][2] + HOP
    for b in range(stop // HOP):
        cal.process(t['rec'][b * HOP:(b + 1) * HOP])
    st['cal_mid'] = snap(s)
    for b in range(stop // HOP, (cal.sched['count_in'][2]) // HOP + 1):
        pass
    s.eng, s.state, s.started = None, 'idle', None
    cal2 = P3.CalRun()
    s.eng, s.state, s.started = cal2, 'calibrating', time.time()
    for b in range((cal2.sched['count_in'][1] + HOP) // HOP):
        cal2.process(t['rec'][b * HOP:(b + 1) * HOP])
    st['cal_count'] = snap(s)
    s.eng, s.state, s.started = None, 'idle', None
    # a real calibration through the page code, then a failed retry
    s.calibrate({'input': 'mic'}, source=t['rec'], sim={'devs': PAIR_A, 'latency': STREAM})
    wait(s, ('idle',))
    st['cal_ok'] = snap(s)
    time.sleep(1.1)
    s.calibrate({'input': 'mic'}, source=LC.make_test(0.160, 32, respond=False)['rec'],
                sim={'devs': PAIR_A, 'latency': STREAM})
    wait(s, ('idle',))
    st['cal_fail'] = snap(s)
    time.sleep(1.1)
    # a song with the calibration (Twinkle, voice_v1 'sung', 90 bpm, micbt latency), stopped at 'await'
    song = twinkle()
    T_b = 60.0 / 90.0
    her = VV.render(song, 'sung', 0.5 + 4 * T_b, T_b)[0]
    x = np.concatenate([np.zeros(int(0.150 * SR), np.float32), her, np.zeros(SR, np.float32)])
    s.start({'song': 'twinkle', 'mode': 'rt', 'click': True, 'bpm': 90, 'input': 'mic'}, source=x,
            sim={'devs': PAIR_A, 'latency': STREAM})
    wait(s, ('await',))
    st['await'] = snap(s)
    run = copy.deepcopy(st['await'])
    run.update({'state': 'running', 'last': None, 'synced': True, 'beat': 21.5})
    st['running'] = run
    s.verdict({'run': st['await']['run'], 'verdict': '作废', 'note': 'page mock v1'})
    time.sleep(1.1)
    # a list with by_latency (in memory only) and pair A calibrated at about 160 ms
    test_rt = {k: v for k, v in P3.RT.items() if k != 'file'}
    test_rt['by_latency'] = {'threshold_ms': 120, 'system': 'r3_X_regtrkstsafe_v1'}
    test_rt['file'] = 'mock list in memory'
    s2 = P3.Session(os.path.join(ROOT, 'bl'), rt=test_rt)
    s2.calibrate({'input': 'mic'}, source=LC.make_test(0.160, 33)['rec'], sim={'devs': PAIR_A, 'latency': STREAM})
    wait(s2, ('idle',))
    st['bl_cal_A'] = snap(s2)
    mock = {'states': st, 'cur': 'idle_nocal', 'calls': [],
            'devices': {'in': [PAIR_B[0], PAIR_A[0], 'iPhone Microphone'], 'out': [PAIR_A[1], 'MacBook Speakers']},
            'runs': []}
    stub = ('<script>\n/* page_mock_v1: fetch stub with real status() dicts (offline, no server) */\n'
            'window.__mock = ' + json.dumps(mock, ensure_ascii=False, default=float) + ';\n'
            'window.fetch = async (path, opts) => {\n'
            '  const m = window.__mock; m.calls.push([path, opts && opts.body ? JSON.parse(opts.body) : null]);\n'
            '  let body = null;\n'
            '  if (path === "/api/state") body = m.states[m.cur];\n'
            '  else if (path === "/api/devices") body = m.devices;\n'
            '  else if (path === "/api/runs") body = m.runs;\n'
            '  else if (path === "/api/calibrate") { m.cur = "cal_mid"; body = m.states[m.cur]; }\n'
            '  else if (path === "/api/start") { m.cur = "running"; body = m.states[m.cur]; }\n'
            '  else if (path === "/api/stop") body = m.states[m.cur];\n'
            '  else if (path === "/api/verdict") { m.cur = "cal_ok"; body = {blind: false}; }\n'
            '  else return {ok: false, statusText: "not found", json: async () => ({error: "not found"})};\n'
            '  return {ok: true, json: async () => JSON.parse(JSON.stringify(body))};\n'
            '};\n</script>\n')
    with open(os.path.join(V4, 'live_panel_v3.html'), encoding='utf-8') as fh:
        html = fh.read()
    assert html.count('</head>') == 1
    html = html.replace('</head>', stub + '</head>')
    with open(OUT, 'x', encoding='utf-8') as fh:
        fh.write(html)
    print('wrote', OUT, 'states', sorted(st), flush=True)
    print('cal_ok cal_last', st['cal_ok']['cal_last']['rt_ms'], st['cal_ok']['cal_last']['ok'],
          '| cal_fail', st['cal_fail']['cal_last']['why'], '| await lat', st['await']['lat'],
          '| bl pick', st['bl_cal_A']['by_latency'], flush=True)


if __name__ == '__main__':
    main()
