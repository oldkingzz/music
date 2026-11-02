"""Duet live test page, version 10 = page v9 with the choir engine v6 (choir_v6.ChoirStart: starts on her first two
notes, no retuning, ignores tracker errors) and the LATEST calibration per device pair again (v9's median of the last
five mixed 204-235 ms measurements from 09:19-09:49 with 113 / 130 ms from 09:53 / 11:02 and told 204 ms; Duet came
about 100 ms early). Serves live_panel_v10.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v10.py        # then open http://127.0.0.1:8766
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v9 as P9  # noqa: E402
import choir_v6 as CV6  # noqa: E402

P5, P3 = P9.P5, P9.P3
P3.PAGE = 'live_panel_v10.html'
_engine_v8 = P9._engine_v8


def _engine(self, mode, name, song, click, on_rt):
    if mode != 'choir':
        return _engine_v8(self, mode, name, song, click, on_rt)
    sid = song['song']
    lib = P5.CHOIR if sid in ('auto', 'free') else ({sid: P5.CHOIR[sid]} if sid in P5.CHOIR else None)
    if lib is None:
        raise ValueError('合唱排练只能唱曲库里的歌：选“自动识别”、Twinkle 或 v11')
    return CV6.ChoirStart(lib, song['bpm'], click_all=bool(click))


P3.Session._engine = _engine
P3.Session._finish_cal = P9._finish_cal_v3                # a new calibration replaces the old one (v3)
P3.SESSION.cals.clear()
for k, v in P9.load_cals(P3.SESSION.root, last=1).items():   # at start: each pair's latest good calibration
    P3.SESSION.cals[k] = dict(v, when=v['when'].replace('最近 1 次的中位数（最后一次 ', '上次测的（'))

if __name__ == '__main__':
    print('page v10 = page v9 + choir_v6 + latest calibration:', {k: v['rt_ms'] for k, v in P3.SESSION.cals.items()},
          flush=True)
    P3.main()
