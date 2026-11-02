"""Duet live test page, version 8 = page v7 plus song 'auto' for the choir (choir_v4.ChoirAuto): she sings any song of
the choir library (Twinkle, v11) in any key, Duet recognises it from her first notes and sings the other three parts,
following her phase and tempo on the melody notes it can match. Serves live_panel_v8.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v8.py        # then open http://127.0.0.1:8766
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v7 as P7  # noqa: E402
import choir_v4 as CV4  # noqa: E402

P5, P3 = P7.P5, P7.P3
P3.PAGE = 'live_panel_v8.html'
AUTO = {'song': 'auto', 'label': '自动识别：唱曲库里任意一首（Twinkle / v11），任意调', 'key': 'C', 'meter': 4,
        'beats': 64, 'bpm': 80.0, 'first_beat': 0.0, 'file': '', 'wav': ''}
P3.SONGS = {'auto': AUTO, **P3.SONGS}
_engine_v7 = P3.Session._engine


def _engine(self, mode, name, song, click, on_rt):
    if mode == 'choir' and song['song'] in ('auto', 'free'):
        return CV4.ChoirAuto(P5.CHOIR, song['bpm'], click_all=bool(click))
    return _engine_v7(self, mode, name, song, click, on_rt)


P3.Session._engine = _engine

if __name__ == '__main__':
    print('page v8 = page v7 + choir song auto', flush=True)
    P3.main()
