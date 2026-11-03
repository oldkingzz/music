"""Duet live test page, version 18 = page v17 whose choir waits until it is sure of the song (choir_v9, sure margin 1.0). v17 was page v16 with the 10-song demo library choir_songs_v9.json (Twinkle, v11, Ode
to Joy and seven public-domain tunes, runs_mini/choir_1102/arrange_v9.py) for Vincent's demo of 2026-11-03. Choir
runs use choir_v8.ChoirMany (several songs share an opening interval; v7's re-identification never ran). Serves live_panel_v18.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v18.py        # then open http://127.0.0.1:8766
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v16 as P16  # noqa: E402
import choir_v9 as CV9  # noqa: E402

P3, P5 = P16.P3, P16.P5
CV9.ChoirMany.SURE_MARGIN = 1.0                           # selftest_demo10_v4: 1.0 -> no wrong-song note, waits 0.7-6.5 s
P16.P15.P14.P13.P12.P10.CV6.ChoirStart = CV9.ChoirMany     # page v10's _engine builds CV6.ChoirStart(...) at run time
with open(os.path.join(HERE, 'choir_songs_v9.json')) as fh:
    P5.CHOIR = json.load(fh)['songs']
P3.PAGE = 'live_panel_v18.html'
P3.SONGS['auto'] = dict(P3.SONGS['auto'], label='自动识别：曲库 10 首里任意一首，任意调')

if __name__ == '__main__':
    print('page v18 = 10-song demo library, choir waits until sure:', sorted(P5.CHOIR), flush=True)
    P3.main()
