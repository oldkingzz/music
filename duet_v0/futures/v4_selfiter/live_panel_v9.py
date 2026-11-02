"""Duet live test page, version 9 = page v8 with the choir score follower (choir_v5.ChoirFollow) for every choir run:
song (with 'auto' / 'free': any song of the library; with Twinkle / v11: that song), key and tuning come from her
singing; tempo and phase follow her. Also:
- the latest good latency calibration of each device pair is loaded from runs_mini/live/latcal_log.jsonl at start
  (v3-v8 forgot it whenever the page restarted), shown as measured earlier; a new calibration replaces it;
- choir runs are told the calibrated round trip itself (no rt-system 20 ms margin: the choir's own LEAD sets its timing).
Serves live_panel_v9.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v9.py        # then open http://127.0.0.1:8766
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v8 as P8  # noqa: E402
import live_panel_v4 as P4  # noqa: E402
import choir_v5 as CV5  # noqa: E402

P5, P3 = P8.P5, P8.P3
P3.PAGE = 'live_panel_v9.html'
_engine_v8 = P3.Session._engine


def _engine(self, mode, name, song, click, on_rt):
    if mode != 'choir':
        return _engine_v8(self, mode, name, song, click, on_rt)
    sid = song['song']
    lib = P5.CHOIR if sid in ('auto', 'free') else ({sid: P5.CHOIR[sid]} if sid in P5.CHOIR else None)
    if lib is None:
        raise ValueError('合唱排练只能唱曲库里的歌：选“自动识别”、Twinkle 或 v11')
    return CV5.ChoirFollow(lib, song['bpm'], click_all=bool(click))


def _set_lat(self, eng, cal, stream_lat):
    if isinstance(eng, CV5.ChoirFollow):
        return P4._set_lat_v3(self, eng, cal, stream_lat)
    return P4._set_lat(self, eng, cal, stream_lat)


P3.Session._engine, P3.Session._set_lat = _engine, _set_lat


def load_cals(root, last=5):
    """Per (input, output) pair: the MEDIAN round trip of its last `last` ok calibrations in latcal_log.jsonl (the
    append-only log of every attempt). One calibration can be far off (09:53 gave 113 ms where 204 / 235 / 216 ms were
    measured within the hour on the same AirPods), and a round trip told too short makes Duet late by the difference."""
    path = os.path.join(root, 'latcal_log.jsonl')
    runs = {}
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('ok') and r.get('rt_ms'):
                runs.setdefault((r['in'], r['out']), []).append(r)
    cals = {}
    for (i, o), rs in runs.items():
        rs = rs[-last:]
        rt = float(sorted(r['rt_ms'] for r in rs)[len(rs) // 2])
        sl = rs[-1].get('stream_latency_ms') or [None, None]
        cals[(i, o)] = {'in': i, 'out': o, 'rt_s': rt / 1000.0, 'rt_ms': rt, 'n_used': rs[-1].get('n_used'),
                        'spread_ms': rs[-1].get('spread_ms'),
                        'when': f'最近 {len(rs)} 次的中位数（最后一次 {str(rs[-1].get("time", ""))[5:16]}）',
                        'stream_Li_ms': sl[0], 'stream_Lo_ms': sl[1]}
    return cals


P3.SESSION.cals.update(load_cals(P3.SESSION.root))
_finish_cal_v3 = P3.Session._finish_cal


def _finish_cal(self):
    """v3's calibration end, then the pair's value becomes the median of its recent ok calibrations (this one included)."""
    _finish_cal_v3(self)
    if self.cal_last and self.cal_last.get('ok'):
        med = load_cals(self.root).get(self.devs)
        if med is not None:
            self.cals[self.devs] = med


P3.Session._finish_cal = _finish_cal

if __name__ == '__main__':
    print('page v9 = page v8 + choir score follower; calibrations loaded:',
          {k: v['rt_ms'] for k, v in P3.SESSION.cals.items()}, flush=True)
    P3.main()
