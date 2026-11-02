"""Idea D dev runs on the frozen live bench (run v5rt_1101, round 3): wraps base rt systems in r3_D_lib_v1.RegAdapter with
configs given on the command line and runs futures/v5_rt/live_bench_rt.evaluate(..., system=obj) on them (the frozen
bench's own code, nothing re-implemented). Offline: the _NoSound tripwire is installed before any other import and
nothing is played. Val only (17 Bach chorales + Twinkle) and the phone files; tuning on val (no test split).

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/D/dev_live_v1.py --runs '<json list>' --voices sung_low,phone_v11 \
      [--songs N] --out runs_mini/v5rt_1101/r3/D/<new>.json
  a run = {"name": .., "base": "r2_copyhm_v1" | "reactive:r2_mix_v2", "cfg": {...} or null (the base itself)}
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('dev_live is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import live_bench_rt as LBR  # noqa: E402  (its own tripwire too; bench_rt, rt_systems paths)
from rt_systems import r3_D_lib_v1 as DL  # noqa: E402

KEYS = ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100', 'stumbles_own_per100', 'on_time_right',
        'wrong_att_per100', 'late_entry', 'double', 'first_100ms_hit', 'first_100ms_harsh', 'attacks_per100', 'blips')


def build_base(name, train, cache):
    if name not in cache:
        if name.startswith('reactive:'):
            cache[name] = LBR.BR.load(name, train)
        else:
            cache[name] = importlib.import_module('rt_systems.' + name).build(train)
    return cache[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--voices', default='sung_low')
    ap.add_argument('--songs', type=int, default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if os.path.exists(path) or not path.endswith('.json'):
        sys.exit('refusing: ' + path)
    runs = json.loads(a.runs)
    train = LBR.bench3.train_sets()
    cache = {}
    out = {'note': 'dev runs on the frozen live bench; val only, tuned on val, no test split', 'songs': a.songs,
           'runs': []}
    for r in runs:
        base = build_base(r['base'], train, cache)
        for voice in a.voices.split(','):
            system = base if r.get('cfg') is None else DL.RegAdapter(base, r['cfg'])
            c0 = time.perf_counter()
            res = LBR.evaluate('dev:' + r['name'], voice, 'wired', a.songs, system=system)
            row = {'name': r['name'], 'base': r['base'], 'cfg': r.get('cfg'), 'voice': voice,
                   'secs': round(time.perf_counter() - c0, 1), 'digest': res['digest']}
            if voice == 'phone_v11':
                row['proxies'] = res['phone']['proxies']
                row['answered'] = res['phone']['answered']
                row['duet_s'] = res['phone']['duet_s']
                row['decision_counts'] = res['phone']['decision_counts']
            elif voice == 'phone_twinkle':
                row['metrics'] = {k: res['phone']['metrics'][k] for k in KEYS}
                row['proxies'] = res['phone']['proxies']
            else:
                row['headline'] = {k: res['headline'][k] for k in KEYS}
                row['twinkle'] = {k: res['twinkle'][k] for k in KEYS}
                row['per_song'] = [{'id': s['id'], 'm': {k: s['metrics'][k] for k in ('rt_nd_live', 'harsh', 'fresh',
                                                                                       'stumbles_per100')},
                                    'clashes': s['fresh_clashes']} for s in res['songs']]
                row['decision_counts'] = res['decision_counts']
            if isinstance(system, DL.RegAdapter):
                row['adapter'] = system.songs[-(len(res.get('songs', [])) + (0 if voice.startswith('phone') else 1)
                                                if not voice.startswith('phone') else 1):]
                system.songs = []
            out['runs'].append(row)
            if voice == 'phone_v11':
                p = row['proxies']
                print(f"{r['name']:28s} {voice:9s} answered {p['answered']} sung {p['sung']} attacks {p['attacks']} "
                      f"cons_heard {p['cons_heard']} changes {p['changes']} on_time_cons {p['on_time_cons']} "
                      f"fresh_heard {p['fresh_heard']} harsh_heard {p['harsh_heard']:.4f}"
                      + (f" s {row['adapter']}" if 'adapter' in row else ''), flush=True)
            elif voice == 'phone_twinkle':
                m = row['metrics']
                print(f"{r['name']:28s} {voice:9s} rt_nd_live {m['rt_nd_live']:.4f} harsh {m['harsh']:.4f} "
                      f"fresh/100 {m['fresh_per100']:.2f} stumbles/100 {m['stumbles_per100']:.2f}", flush=True)
            else:
                m, t = row['headline'], row['twinkle']
                print(f"{r['name']:28s} {voice:9s} rt_nd_live {m['rt_nd_live']:.4f} harsh {m['harsh']:.4f} "
                      f"fresh/100 {m['fresh_per100']:.2f} ({m['fresh']}) stumbles/100 {m['stumbles_per100']:.2f} "
                      f"on_time_right {m['on_time_right']:.4f} wrong_att/100 {m['wrong_att_per100']:.1f} "
                      f"double {m['double']:.4f} | twinkle {t['rt_nd_live']:.4f} fresh/100 {t['fresh_per100']:.2f} "
                      f"| {row['secs']:.0f} s", flush=True)
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1, default=LBR._js)
    print('wrote', path)


if __name__ == '__main__':
    main()
