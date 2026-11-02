"""Report-only robustness sweep of futures/v4_selfiter/latcal_v1.estimate beyond C1 (run v5rt_1101, round 4). Synthetic
recordings from latcal_v1.make_test (never played; sounddevice tripwire first). Seeds 0 and 1, fixed in advance.

  jitter30     her timing sd 30 ms instead of 15 (an untrained singer)         round trips 60 / 160 / 330 ms
  noise-25     room noise 25 dB below her voice instead of 40                    60 / 160 / 330
  noise-15     room noise 15 dB below her voice (very noisy room)                60 / 160 / 330
  bleed        the clicks also leak into the microphone (loudspeakers)          20 / 160 / 330
  miss+early   one missing response (click 2) and one 120 ms early (click 7)     60 / 160 / 330
  long         very long round trips (AirPods both ways, bad link)              400 / 480 / 540
  too_long     round trips beyond one click interval (600 ms at 100 bpm)         620 / 700: should refuse, never be
               confidently wrong
Output: runs_mini/v5rt_1101/r4/P/latcal_robust_v1.json (new file).

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python runs_mini/v5rt_1101/tools/latcal_robust_v1.py
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('latcal sweep is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import latcal_v1 as LC  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r4', 'P', 'latcal_robust_v1.json')
GROUPS = [('jitter30', (0.060, 0.160, 0.330), {'jitter': 0.030}),
          ('noise-25', (0.060, 0.160, 0.330), {'noise_db': -25.0}),
          ('noise-15', (0.060, 0.160, 0.330), {'noise_db': -15.0}),
          ('bleed', (0.020, 0.160, 0.330), {'bleed': True}),
          ('miss+early', (0.060, 0.160, 0.330), {'missing': 1, 'early': 6}),
          ('long', (0.400, 0.480, 0.540), {}),
          ('too_long', (0.620, 0.700), {})]


def main():
    if os.path.exists(OUT):
        sys.exit(f'refusing to overwrite {OUT}')
    rows, summary = [], {}
    for name, rts, kw in GROUPS:
        g = []
        for rt in rts:
            for seed in (0, 1):
                t = LC.make_test(rt, seed, **kw)
                r = LC.estimate(t['rec'], t['sr'], t['clicks'])
                err = None if r['rt_s'] is None else round((r['rt_s'] - rt) * 1000, 1)
                row = {'group': name, 'rt_true_ms': round(rt * 1000), 'seed': seed, 'ok': r['ok'], 'err_ms': err,
                       'rt_ms': None if r['rt_s'] is None else round(r['rt_s'] * 1000, 1), 'n_used': r['n_used'],
                       'spread_ms': None if r['spread_s'] is None else round(r['spread_s'] * 1000, 1),
                       'why': r['why'], 'rejected': r.get('rejected')}
                rows.append(row)
                g.append(row)
                print(f"{name:10s} rt {row['rt_true_ms']:3d} s{seed} ok {str(r['ok']):5s} rt {row['rt_ms']} err {err} "
                      f"used {r['n_used']} spread {row['spread_ms']} {r['why']}", flush=True)
        okr = [x for x in g if x['ok']]
        summary[name] = {'cases': len(g), 'ok': len(okr),
                         'max_abs_err_ms_when_ok': max((abs(x['err_ms']) for x in okr), default=None),
                         'ok_but_off_by_more_than_25ms': sum(abs(x['err_ms']) > 25 for x in okr)}
    res = {'what': 'report-only robustness sweep of latcal_v1.estimate (synthetic, never played)',
           'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'summary': summary, 'cases': rows}
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False, default=float)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    main()
