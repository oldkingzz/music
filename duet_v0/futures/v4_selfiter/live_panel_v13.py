"""Duet live test page, version 13 = page v12 with a cleaner choir form: in mode 'choir' the key field, the tempo field
(unless the metronome box is ticked), the one-button system choice and the 'free' song are hidden, since the choir takes
key and tempo from her singing (Vincent, 2026-11-02: "那这个还有意义吗？"). Serves live_panel_v13.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v13.py        # then open http://127.0.0.1:8766
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v12 as P12  # noqa: E402

P3 = P12.P3
P3.PAGE = 'live_panel_v13.html'

if __name__ == '__main__':
    print('page v13 = page v12 + cleaner choir form', flush=True)
    P3.main()
