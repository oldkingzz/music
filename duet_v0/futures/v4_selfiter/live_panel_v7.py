"""Duet live test page, version 7 = page v6 with the choir engine v3 (choir_v3.ChoirDuet): no snapping to the click
grid, phase following on her onsets, parts 30 ms earlier for the reed's attack. Serves live_panel_v7.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v7.py        # then open http://127.0.0.1:8766
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v6 as P6  # noqa: E402
import choir_v3 as CV3  # noqa: E402

P5, P3 = P6.P5, P6.P3
P5.CV = CV3
P3.PAGE = 'live_panel_v7.html'

if __name__ == '__main__':
    print('page v7 = page v6 + choir phase following', flush=True)
    P3.main()
