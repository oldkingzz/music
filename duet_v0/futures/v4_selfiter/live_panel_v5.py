"""Duet live test page, version 5 = page v4 (v3 + 20 ms calibration margin) plus mode 'choir' (合唱排练):
she sings the melody of Twinkle or v11 after the 4-click count-in, Duet sings alto, tenor and bass from a fixed
four-part arrangement (choir_v1.ChoirDuet, choir_songs_v3.json), on the click grid, latency compensated.
Same server, records and history as v3; serves live_panel_v5.html (v3's page plus the choir mode).

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v5.py        # then open http://127.0.0.1:8766
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v4 as P4  # noqa: E402  (patches v3's latency margin)
import choir_v1 as CV  # noqa: E402

P3 = P4.P3
CHOIR = CV.load_songs()
P3.PAGE = 'live_panel_v5.html'
P3.MODES = P3.MODES + ('choir',)
_engine_v3, _start_v3, _label_v3 = P3.Session._engine, P3.Session.start, P3.label


def _engine(self, mode, name, song, click, on_rt):
    if mode != 'choir':
        return _engine_v3(self, mode, name, song, click, on_rt)
    if song['song'] not in CHOIR:
        raise ValueError('合唱排练只有 Twinkle 和 v11 两首有谱的歌')
    return CV.ChoirDuet(CHOIR[song['song']], P3.lm.KEYS[song['key']], song['bpm'], click_all=bool(click))


def _start(self, p, source=None, sim=None):
    if p.get('mode') == 'choir':
        if p.get('input') == 'file' and source is None:
            raise ValueError('合唱排练要你自己唱：把“输入”换成麦克风')
        p = dict(p, system='r2_mix_v2')                  # the system field is not used by the choir
    return _start_v3(self, p, source=source, sim=sim)


def label(name, mode, rt=None):
    if mode == 'choir':
        return '合唱排练：Duet 按谱唱中音、次中音、低音'
    return _label_v3(name, mode, rt)


P3.Session._engine, P3.Session.start, P3.label = _engine, _start, label

if __name__ == '__main__':
    print('page v5 = page v4 + choir rehearsal mode', flush=True)
    P3.main()
