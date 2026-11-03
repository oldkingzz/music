"""Duet live test page, version 16 = page v15 with the 100-song choir library choir_songs_v8.json (Twinkle, v11,
Ode to Joy and 97 POP909 songs, runs_mini/choir_1102/arrange_v8.py). Melody MIDI files for Logic Pro are in
futures/live_test/midi_v1/ (runs_mini/choir_1102/midi_export_v1.py). Choir runs use choir_v7.ChoirMany, which keeps
every song with her opening interval as a candidate and settles the song over her next notes (choir_v6 decided on the
opening interval alone: 16 of 17 POP909 songs taken for another). Serves live_panel_v16.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v16.py        # then open http://127.0.0.1:8766
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v15 as P15  # noqa: E402
import choir_v7 as CV7  # noqa: E402

P3, P5 = P15.P3, P15.P5
P15.P14.P13.P12.P10.CV6.ChoirStart = CV7.ChoirMany     # page v10's _engine builds CV6.ChoirStart(...) at run time
with open(os.path.join(HERE, 'choir_songs_v8.json')) as fh:
    P5.CHOIR = json.load(fh)['songs']
P3.PAGE = 'live_panel_v16.html'
P3.SONGS['auto'] = dict(P3.SONGS['auto'], label='自动识别：曲库 100 首里任意一首（Twinkle / v11 / 欢乐颂 / POP909），任意调')

if __name__ == '__main__':
    print('page v16 = page v15 + 100-song library:', len(P5.CHOIR), flush=True)
    P3.main()
