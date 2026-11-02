"""Offline self-test of futures/v4_selfiter/live_panel_v4.py (page v3 + 20 ms calibration margin) with the real list
live_rt_systems_v3.json, NO sound device (tripwire first, as panel_v3_selftest_v1.py). Every song run is rated 作废.
Runs go to runs_mini/live/selftest_v4/<HHMMSS>/; summary runs_mini/v5rt_1101/verify/panel_v4_selftest_v1.json.

Checks: the newest list is v3 (default r3_X_regtrkst_v1, by_latency 250 ms -> r4_X_hgeh_v1) and every system of it
builds; latcal_v1.make_test recordings calibrated through the page for three device pairs (A: AirPods mic + wired
phones, true round trip 160 ms; C: AirPods both ways, 330 ms; B: not calibrated); 'auto' Twinkle runs (phone file,
14 s) on A run the default with Li = RT + 20 ms - Lo, on C run r4_X_hgeh_v1 with Li = RT + 20 ms - Lo, on B run the
default with PortAudio's Li and no margin; the rt blind A/B on C only plays reactive r2_mix_v2 or r4_X_hgeh_v1; the
page, latcal, v2 and list files are unchanged by the run.

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python runs_mini/v5rt_1101/tools/panel_v4_selftest_v1.py
"""
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

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import soundfile as sf  # noqa: E402

V4DIR = os.path.join(ABS, 'futures', 'v4_selfiter')
sys.path.insert(0, V4DIR)
import live_panel_v4 as P4  # noqa: E402
P3 = P4.P3
import latcal_v1 as LC  # noqa: E402
from dsp import SR  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'verify', 'panel_v4_selftest_v1.json')
ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_v4', time.strftime('%H%M%S'))
W, BIG = 'r3_X_regtrkst_v1', 'r4_X_hgeh_v1'
PAIR_A = ('selftest AirPods mic', 'selftest wired phones')
PAIR_B = ('selftest other mic', 'selftest wired phones')
PAIR_C = ('selftest AirPods mic', 'selftest AirPods')
STREAM_A, STREAM_C = (0.063, 0.010), (0.063, 0.042)        # what PortAudio reports (live bench v1 / v2 settings)
FILES = ['live_panel_v2.py', 'live_panel_v2.html', 'live_panel_v3.py', 'live_panel_v3.html', 'live_panel_v4.py',
         'latcal_v1.py']
CHECKS = []


def check(what, ok, detail=None):
    CHECKS.append({'what': what, 'ok': bool(ok), 'detail': detail})
    print(f"{'OK  ' if ok else 'FAIL'} {what}" + ('' if detail is None else f'  {detail}'), flush=True)


def sha_file(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def wait(s, states, timeout=300.0):
    t0 = time.time()
    while s.state not in states and time.time() - t0 < timeout:
        time.sleep(0.02)
    if s.state not in states:
        raise RuntimeError(f'state {s.state} after {timeout} s, wanted {states}')


def cal_run(s, rec, pair, stream):
    st = s.calibrate({'input': 'mic'}, source=rec, sim={'devs': pair, 'latency': stream})
    assert st['state'] == 'calibrating', st['state']
    wait(s, ('idle',))
    time.sleep(1.1)                                        # folders are named by the second
    return s.cal_last


def song_run(s, p, x, pair, stream, note):
    st = s.start(p, source=x, sim={'devs': pair, 'latency': stream})
    assert st['state'] == 'running', st['state']
    wait(s, ('await',))
    st = s.status()
    eng = s.eng
    rec = s.verdict({'run': st['run'], 'verdict': '作废', 'note': note})
    out = {'run': rec['run'], 'name': s.name, 'mode': rec['mode'], 'Li': float(eng.Li), 'Lo': float(eng.Lo),
           'lat': dict(s.lat or {}), 'pick': dict(s.pick or {}), 'answered': rec['notes_answered'], 'sung': rec['sung']}
    time.sleep(1.1)
    return out


def main():
    if os.path.exists(OUT):
        sys.exit(f'refusing to overwrite {OUT}')
    t_start = time.time()
    shas0 = {f: sha_file(os.path.join(V4DIR, f)) for f in FILES}
    lists0 = {os.path.basename(p): sha_file(p) for p in glob.glob(os.path.join(V4DIR, 'live_rt_systems_v*.json'))}
    rt = P3.RT
    check('newest list is live_rt_systems_v3.json', rt['file'] == 'live_rt_systems_v3.json', rt['file'])
    check('default r3_X_regtrkst_v1, by_latency 250 ms -> r4_X_hgeh_v1',
          rt['default'] == W and rt['by_latency'] == {'threshold_ms': 250, 'system': BIG}, rt.get('by_latency'))
    check('v4 patched Session._set_lat', P3.Session._set_lat is P4._set_lat)
    P3.MODELS.load()
    check('every system of the list builds', P3.MODELS.err is None and all(n in P3.MODELS.rt for n in P3.rt_names(rt)),
          P3.MODELS.err or sorted(P3.MODELS.rt))
    s = P3.Session(ROOT)
    ca = cal_run(s, LC.make_test(0.160, 41)['rec'], PAIR_A, STREAM_A)
    cc = cal_run(s, LC.make_test(0.330, 42)['rec'], PAIR_C, STREAM_C)
    check('pair A calibrated within 15 ms of 160 ms', ca['ok'] and abs(ca['rt_ms'] - 160) <= 15, ca['rt_ms'])
    check('pair C calibrated within 15 ms of 330 ms', cc['ok'] and abs(cc['rt_ms'] - 330) <= 15, cc['rt_ms'])
    x, sr = sf.read(os.path.join(ABS, 'futures', 'live_test', 'phone', 'twinkle_C_80bpm_countin.wav'), dtype='float32',
                    always_2d=True)
    assert sr == SR
    x = x.mean(1)[:int(14.0 * SR)]
    p = {'song': 'twinkle', 'mode': 'rt'}
    a = song_run(s, p, x, PAIR_A, STREAM_A, 'selftest v4 auto A')
    want_a = max(0.0, s.cals[PAIR_A]['rt_s'] + 0.020 - STREAM_A[1])
    check("A (160 ms): 'auto' runs the default", a['name'] == W and '<' in a['pick']['why'], (a['name'], a['pick']['why']))
    check('A: Li = RT + 20 ms - Lo, Lo = stream, margin recorded, page shows the measured RT',
          abs(a['Li'] - want_a) < 1e-12 and a['Lo'] == STREAM_A[1] and a['lat'].get('margin_ms') == 20.0
          and a['lat'].get('rt_ms') == ca['rt_ms'] and a['lat'].get('source') == 'calibrated', (a['Li'], want_a, a['lat']))
    c = song_run(s, p, x, PAIR_C, STREAM_C, 'selftest v4 auto C')
    want_c = max(0.0, s.cals[PAIR_C]['rt_s'] + 0.020 - STREAM_C[1])
    check("C (330 ms): 'auto' runs r4_X_hgeh_v1 and says why", c['name'] == BIG and '≥' in c['pick']['why'],
          (c['name'], c['pick']['why']))
    check('C: Li = RT + 20 ms - Lo', abs(c['Li'] - want_c) < 1e-12 and c['Lo'] == STREAM_C[1], (c['Li'], want_c))
    b = song_run(s, p, x, PAIR_B, STREAM_A, 'selftest v4 auto B')
    check("B (not calibrated): default, PortAudio's Li, no margin",
          b['name'] == W and b['Li'] == STREAM_A[0] and 'margin_ms' not in b['lat'] and b['lat'].get('source') == 'portaudio',
          (b['name'], b['Li'], b['lat']))
    arms = []
    for k in range(4):
        random.seed(k)
        o = song_run(s, {'song': 'twinkle', 'mode': 'rt', 'system': 'blind'}, x, PAIR_C, STREAM_C, f'selftest v4 blind {k}')
        arms.append((o['mode'], o['name']))
    check('C: the rt blind A/B plays only reactive r2_mix_v2 or r4_X_hgeh_v1',
          all(t in {('reactive', 'r2_mix_v2'), ('rt', BIG)} for t in arms), arms)
    shas1 = {f: sha_file(os.path.join(V4DIR, f)) for f in FILES}
    lists1 = {os.path.basename(q): sha_file(q) for q in glob.glob(os.path.join(V4DIR, 'live_rt_systems_v*.json'))}
    check('page, latcal and list files unchanged by the run', shas0 == shas1 and lists0 == lists1)
    res = {'what': 'offline selftest of live_panel_v4 (page v3 + 20 ms calibration margin), list v3, no sound device',
           'root': ROOT.replace(ABS + '/', ''), 'checks': CHECKS, 'all_pass': all(k['ok'] for k in CHECKS),
           'cal': {'A': ca, 'C': cc}, 'runs': {'A': a, 'C': c, 'B': b, 'blind_C': arms}, 'code': shas1, 'lists': lists1,
           'seconds': round(time.time() - t_start, 1), 'written': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1, default=float, ensure_ascii=False)
    print(json.dumps({'all_pass': res['all_pass'], 'n': len(CHECKS), 'seconds': res['seconds']}), flush=True)


if __name__ == '__main__':
    main()
