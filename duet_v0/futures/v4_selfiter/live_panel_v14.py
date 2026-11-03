"""Duet live test page, version 14 = page v13 with the round trip of Vincent's AirPods Max FIXED at 174 ms (his 19:59
calibration of 2026-11-02, 8 of 8 beeps used), so there is nothing to calibrate before singing (Vincent, 2026-11-02:
"我们要不固定死延迟？因为不能每次都测啊"). Each calibration is timed by his own "da" against the beeps: one calibration's
beeps spread 119-216 ms, and the day's good ones gave 113-235 ms; all 43 good beeps of the day have their median at
181 ms. For the fixed pair a calibration is still measured and shown, but Duet keeps the fixed value (change FIXED to
move it); other device pairs calibrate as before. Serves live_panel_v14.html (v13's page saying so).

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v14.py        # then open http://127.0.0.1:8766
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v13 as P13  # noqa: E402

P3 = P13.P3
P3.PAGE = 'live_panel_v14.html'
AIRPODS = '泽森的AirPods Max'
FIXED = {(AIRPODS, AIRPODS): {'in': AIRPODS, 'out': AIRPODS, 'rt_s': 0.174, 'rt_ms': 174.0, 'n_used': 8,
                              'spread_ms': 24.2, 'when': '11-02 19:59', 'fixed': True,
                              'stream_Li_ms': 62.7, 'stream_Lo_ms': 42.1}}
_finish_cal_v3 = P3.Session._finish_cal
_set_lat_v9 = P3.Session._set_lat


def _finish_cal(self):
    """v3's calibration end (logged, shown as cal_last); on a fixed pair Duet keeps FIXED."""
    _finish_cal_v3(self)
    if self.devs in FIXED:
        self.cals[self.devs] = dict(FIXED[self.devs])


def _set_lat(self, eng, cal, stream_lat):
    _set_lat_v9(self, eng, cal, stream_lat)
    self.lat['fixed'] = bool(cal is not None and cal.get('fixed'))


P3.Session._finish_cal, P3.Session._set_lat = _finish_cal, _set_lat
P3.SESSION.cals.update({k: dict(v) for k, v in FIXED.items()})

if __name__ == '__main__':
    print('page v14 = page v13 + fixed round trip:', {k: v['rt_ms'] for k, v in P3.SESSION.cals.items()}, flush=True)
    P3.main()
