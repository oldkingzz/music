"""Duet live test page, version 15 = page v14 with a third choir song, Ode to Joy (choir_songs_v7.json 'ode', C major,
96 bpm, 64 beats; arranged by runs_mini/choir_1102/arrange_v6.py; Twinkle and v11 unchanged). Its phone file is
futures/live_test/phone/ode_C_96bpm_lead3s.mp3 (synthetic singer, 3 s of silence first). Serves live_panel_v15.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v15.py        # then open http://127.0.0.1:8766
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v14 as P14  # noqa: E402

P3, P5 = P14.P3, P14.P13.P12.P5
with open(os.path.join(HERE, 'choir_songs_v7.json')) as fh:
    P5.CHOIR = json.load(fh)['songs']
P3.PAGE = 'live_panel_v15.html'
ODE = {'song': 'ode', 'label': '欢乐颂（C 大调，96 拍）', 'key': 'C', 'meter': 4, 'beats': 64, 'bpm': 96.0,
       'first_beat': 0.0, 'file': 'futures/live_test/phone/ode_C_96bpm_lead3s.mp3',
       'wav': 'futures/live_test/phone/ode_C_96bpm_lead3s.wav'}
P3.SONGS = {k: v for k, v in P3.SONGS.items() if k != 'free'} | {'ode': ODE, 'free': P3.SONGS['free']}
P3.SONGS['auto'] = dict(P3.SONGS['auto'], label='自动识别：唱曲库里任意一首（Twinkle / v11 / 欢乐颂），任意调')

if __name__ == '__main__':
    print('page v15 = page v14 + Ode to Joy:', sorted(P5.CHOIR), flush=True)
    P3.main()
