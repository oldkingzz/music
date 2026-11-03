"""Offline selftest of page v14 (run choir_1102): the fixed AirPods Max round trip (174 ms) reaches the engine with no
calibration this session; a calibration on that pair is measured and logged but does not replace it; another pair
still takes its new calibration. Tripwire (no sound): tonight's Twinkle run (195914) is replayed under the AirPods
pair's device names, then tonight's 19:59 calibration recording is replayed 40 ms late on the AirPods pair and on a
test pair. Outputs go to a new runs_mini/live/selftest_choir/<time>_panel_v14 folder.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py selftest_v14 400 .venv/bin/python runs_mini/choir_1102/selftest_panel_v14.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v10 as E  # noqa: E402  (tripwire, page v13 on the path)
import live_panel_v14 as P14  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

P3, A = P14.P3, P14.AIRPODS
SIM = {'devs': (A, A), 'latency': (0.0627, 0.0421)}


def wait(s, state, secs):
    t0 = time.time()
    while s.state != state and time.time() - t0 < secs:
        time.sleep(0.05)
    assert s.state == state, (s.state, state)


if __name__ == '__main__':
    assert P3 is E.P3 and P3.PAGE == 'live_panel_v14.html'
    with open(os.path.join(P14.HERE, P3.PAGE)) as fh:
        html = fh.read()
    assert '实时测试 v14' in html and 'id="calh"' in html and 'l.fixed' in html
    page = P3.SESSION.cals[(A, A)]
    print('page session, AirPods pair:', page, flush=True)
    assert page['rt_ms'] == 174.0 and page['fixed']

    P3.MODELS.load()
    s = P3.Session(os.path.join(P3.LB.OUT_ROOT, 'selftest_choir', time.strftime('%H%M%S') + '_panel_v14'))
    s.cals.update({k: dict(v) for k, v in P3.SESSION.cals.items()})    # what the page's own session starts with
    x = sf.read(os.path.join(E.ABS, 'runs_mini/live/195914_r2_mix_v2/run.wav'), dtype='float32', always_2d=True)[0]
    s.start({'song': 'auto', 'bpm': 80.0, 'mode': 'choir', 'click': False, 'input': 'mic'}, source=x[:, 0], sim=SIM)
    wait(s, 'await', 120)
    eng = s.eng
    print('run on the AirPods pair:', s.lat, 'engine Li %.4f Lo %.4f' % (eng.Li, eng.Lo), eng.song_id, flush=True)
    assert s.lat['source'] == 'calibrated' and s.lat['fixed'] and abs(eng.Li - (0.174 - 0.0421)) < 1e-6
    s.verdict({'run': s.status()['run'], 'verdict': '作废', 'note': 'selftest v14'})
    time.sleep(1.1)

    y = sf.read(os.path.join(E.ABS, 'runs_mini/live/195901_latcal/cal.wav'), dtype='float32', always_2d=True)[0]
    y = np.concatenate([np.zeros(int(0.04 * E.SR), np.float32), y[:, 0]])  # every "da" 40 ms later than at 19:59
    for devs in ((A, A), ('m', 'p')):
        wait(s, 'idle', 10)
        s.calibrate({'input': 'mic'}, source=y, sim=dict(SIM, devs=devs))
        time.sleep(0.5)
        wait(s, 'idle', 60)
        r, kept = s.cal_last, s.cals.get(devs)
        print(devs, 'measured ok=%s rt=%s ms -> Duet uses %s ms, fixed=%s' % (r.get('ok'), r.get('rt_ms'),
              kept and kept['rt_ms'], kept and kept.get('fixed')), flush=True)
        assert r.get('ok') and abs(r['rt_ms'] - 214) < 15
        if devs == (A, A):
            assert kept['rt_ms'] == 174.0 and kept['fixed']
        else:
            assert kept['rt_ms'] == r['rt_ms'] and not kept.get('fixed')
    print('selftest v14 ok', flush=True)
