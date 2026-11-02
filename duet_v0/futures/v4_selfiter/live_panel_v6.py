"""Duet live test page, version 6 = page v5 with the choir engine v2 (choir_v2.ChoirDuet): after the count-in Duet
waits for her first note before the parts come in, and stops when she stops for 3 beats. Serves live_panel_v6.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v6.py        # then open http://127.0.0.1:8766
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v5 as P5  # noqa: E402
import choir_v2 as CV2  # noqa: E402

P5.CV = CV2                                              # P5._engine builds CV.ChoirDuet
P3 = P5.P3
P3.PAGE = 'live_panel_v6.html'

if __name__ == '__main__':
    print('page v6 = page v5 + choir waits for her first note', flush=True)
    P3.main()
