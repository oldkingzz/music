"""Duet live test page, version 4 = live_panel_v3.py with a +20 ms safety margin on the calibrated round trip.

Why (run v5rt_1101, round 4, val only, tuned on val): telling the engine a round trip that is too SHORT costs a lot and
one that is too LONG costs almost nothing. Live bench v2, mean of sung and sung_low, calibration off by 25 ms:
  r3_X_regtrkst_v1 at micbt_cal 0.5486 exact, 0.5264 short, 0.5484 long; at bt_cal 0.4570, 0.4435 short, 0.4604 long;
  r4_X_hgeh_v1 at bt_cal 0.4848 exact, 0.4612 short, 0.4827 long.
latcal_v1's estimate is within +-9 ms on the C1 test signals, but about 1-2 % of calibrations on new seeds come out
15-22 ms off, and a singer who anticipates the clicks (or whose 'd' burst comes before the vowel she aligns) reads short.
So a calibrated microphone run tells the engine Li = max(0, RT + 0.020 - Lo). Everything else is live_panel_v3
unchanged: the same HTML (live_panel_v3.html), port 8766, records and history; the page shows the MEASURED round trip;
by_latency compares the MEASURED round trip; notes.json's 'lat' gets margin_ms = 20 and page 'v4'.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v4.py        # then open http://127.0.0.1:8766
  offline check, no sound device: .venv/bin/python runs_mini/v5rt_1101/tools/panel_v4_selftest_v1.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v3 as P3  # noqa: E402

MARGIN_S = 0.020
_set_lat_v3 = P3.Session._set_lat


def _set_lat(self, eng, cal, stream_lat):
    """v3's _set_lat with the calibrated round trip read MARGIN_S long (no change without a calibration)."""
    if cal is not None:
        cal = dict(cal, rt_s=float(cal['rt_s']) + MARGIN_S)
    _set_lat_v3(self, eng, cal, stream_lat)
    if cal is not None:
        self.lat.update({'margin_ms': round(MARGIN_S * 1000, 1), 'page': 'v4'})


P3.Session._set_lat = _set_lat

if __name__ == '__main__':
    print('page v4 = page v3 + 20 ms calibration margin', flush=True)
    P3.main()
