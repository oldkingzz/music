"""Duet live test page, version 17 = page v16 with the 10-song demo library choir_songs_v9.json (Twinkle, v11, Ode
to Joy and seven public-domain tunes, runs_mini/choir_1102/arrange_v9.py) for Vincent's demo of 2026-11-03. Choir
runs use choir_v8.ChoirMany (several songs share an opening interval; v7's re-identification never ran). Serves live_panel_v17.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v17.py        # then open http://127.0.0.1:8766
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v16 as P16  # noqa: E402
import choir_v8 as CV8  # noqa: E402

P3, P5 = P16.P3, P16.P5
P16.P15.P14.P13.P12.P10.CV6.ChoirStart = CV8.ChoirMany     # page v10's _engine builds CV6.ChoirStart(...) at run time
with open(os.path.join(HERE, 'choir_songs_v9.json')) as fh:
    P5.CHOIR = json.load(fh)['songs']
P3.PAGE = 'live_panel_v17.html'
P3.SONGS['auto'] = dict(P3.SONGS['auto'], label='自动识别：曲库 10 首里任意一首，任意调')

if __name__ == '__main__':
    print('page v17 = page v16 + 10-song demo library:', sorted(P5.CHOIR), flush=True)
    P3.main()
