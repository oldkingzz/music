"""Duet live test page, version 12 = page v11 with the choir library choir_songs_v5.json (v11 re-arranged with slower
chord changes, runs_mini/choir_1102/arrange_v5.py) and the choir_v6 follower fixes for v11 (a sixth leap is no longer
taken for a tracker error; tempo bounded by the song's own tempo, not the page's; 6 notes look-ahead; 80 ms notes).
Serves live_panel_v10.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v12.py        # then open http://127.0.0.1:8766
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v11 as P11  # noqa: E402

P10 = P11.P10
P5, P3 = P10.P5, P10.P3
with open(os.path.join(HERE, 'choir_songs_v5.json')) as fh:
    P5.CHOIR = json.load(fh)['songs']
P3.PAGE = 'live_panel_v12.html'
_engine_v10 = P3.Session._engine


def _engine(self, mode, name, song, click, on_rt):
    """Choir runs need no key and no tempo: the follower takes both from her singing. Without the page's metronome
    box there is no count-in either (the 4 clicks only ever gave a tempo she does not have to keep): Duet just waits
    for her. The click grid stays internal (time reference only)."""
    eng = _engine_v10(self, mode, name, song, click, on_rt)
    if mode == 'choir' and not click:
        eng._click = lambda n: np.zeros(n)
    return eng


P3.Session._engine = _engine

if __name__ == '__main__':
    print('page v12 = page v11 + choir_songs_v5', flush=True)
    P3.main()
