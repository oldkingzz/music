"""Idea D dev runs (run v5rt_1101, round 3), two modes, both on the frozen benches' own code:
  --mode live   futures/v5_rt/live_bench_rt.evaluate(..., system=obj) on the given voices (offline; the _NoSound tripwire
                is installed before any other import; nothing is played)
  --mode bench  bench_rt's evaluate loop (bench_rt.simulate / score_song / row, the same heard melody, L = 0.1) and
                runs_mini/v5rt_1101/tools/attack_metric.song_counts on the same step functions: rt_nd, harsh and fresh
                clashes per dataset, and the keep-rule checks against the reactive files on disk (G1 preview only; the
                scored G1 numbers come from the CLI runs of the variant files)
A run = {"name", "base": "<rt_systems name>" | "reactive:r2_mix_v2", "base_cfg": {overrides of the base module's CFG,
rebuilt with rt_systems.r2_lib_v3.PlanR2C} or absent, "cfg": RegAdapter cfg (r3_D_lib_v1) or null (the base itself)}.
Val only, tuned on val (no test split).

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/D/dev_v2.py --mode live --runs '<json>' --voices sung_low,phone_v11 \
      [--songs N] --out runs_mini/v5rt_1101/r3/D/<new>.json
  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/D/dev_v2.py --mode bench --runs '<json>' [--seed2] --out <new>.json
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('dev_v2 is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
sys.path.insert(0, os.path.join(RUN, 'tools'))
import live_bench_rt as LBR  # noqa: E402  (its own tripwire too; bench_rt, rt_systems paths)
import attack_metric as AM  # noqa: E402
from rt_systems import r3_D_lib_v1 as DL  # noqa: E402

BR = LBR.BR
KEYS = ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100', 'stumbles_own_per100', 'on_time_right',
        'wrong_att_per100', 'late_entry', 'double', 'first_100ms_hit', 'first_100ms_harsh', 'attacks_per100', 'blips')
DS = ('bach', 'folk', 'pop', 'pop909')


def build_base(r, train, cache):
    key = json.dumps([r['base'], r.get('base_cfg')], sort_keys=True)
    if key not in cache:
        if r['base'].startswith('reactive:'):
            cache[key] = BR.load(r['base'], train)
        elif r.get('base_cfg'):
            from systems import r2_mix_v2
            from rt_systems import r2_lib_v3 as LIB
            from rt_systems.r2_lib_v1 import Prior, load_copy_tab
            cfg = dict(importlib.import_module('rt_systems.' + r['base']).CFG)
            cfg.update(r['base_cfg'])
            cache[key] = LIB.PlanR2C(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab(), LIB.load_hm_tab())
        else:
            cache[key] = importlib.import_module('rt_systems.' + r['base']).build(train)
    return cache[key]


def live(runs, voices, n_songs, train):
    cache, rows = {}, []
    for r in runs:
        base = build_base(r, train, cache)
        for voice in voices:
            system = base if r.get('cfg') is None else DL.RegAdapter(base, r['cfg'])
            c0 = time.perf_counter()
            res = LBR.evaluate('dev:' + r['name'], voice, 'wired', n_songs, system=system)
            row = {'name': r['name'], 'base': r['base'], 'base_cfg': r.get('base_cfg'), 'cfg': r.get('cfg'),
                   'voice': voice, 'secs': round(time.perf_counter() - c0, 1), 'digest': res['digest']}
            if voice == 'phone_v11':
                row['proxies'] = res['phone']['proxies']
                row['answered'] = res['phone']['answered']
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
                row['adapter'] = system.songs
            rows.append(row)
            if voice == 'phone_v11':
                p = row['proxies']
                print(f"{r['name']:30s} {voice:9s} answered {p['answered']} sung {p['sung']} attacks {p['attacks']} "
                      f"cons_heard {p['cons_heard']} changes {p['changes']} on_time_cons {p['on_time_cons']} "
                      f"fresh_heard {p['fresh_heard']} harsh_heard {p['harsh_heard']:.4f}", flush=True)
            elif voice == 'phone_twinkle':
                m = row['metrics']
                print(f"{r['name']:30s} {voice:9s} rt_nd_live {m['rt_nd_live']:.4f} harsh {m['harsh']:.4f} "
                      f"fresh/100 {m['fresh_per100']:.2f} stumbles/100 {m['stumbles_per100']:.2f}", flush=True)
            else:
                m, t = row['headline'], row['twinkle']
                sh = ''
                if 'adapter' in row:
                    fin = [x['s'][-1][1] if x['s'] else 0 for x in row['adapter']]
                    sh = f" | shifted {sum(1 for v in fin[:-1] if v)}/{len(fin) - 1} twinkle s {fin[-1]}"
                print(f"{r['name']:30s} {voice:9s} rt_nd_live {m['rt_nd_live']:.4f} harsh {m['harsh']:.4f} "
                      f"fresh/100 {m['fresh_per100']:.2f} ({m['fresh']}) stumbles/100 {m['stumbles_per100']:.2f} "
                      f"on_time_right {m['on_time_right']:.4f} wrong_att/100 {m['wrong_att_per100']:.1f} "
                      f"| twinkle {t['rt_nd_live']:.4f} fresh/100 {t['fresh_per100']:.2f}{sh} | {row['secs']:.0f} s",
                      flush=True)
    return rows


def bench(runs, seed2, train, L=0.1):
    heard = BR.heard_seed2 if seed2 else BR.HEARD1
    ref = {}
    for tag, rel in (('s', 'scores/B_reactive_r2_mix_v2_L0.1.json'), ('s2', 'scores/B_reactive_r2_mix_v2_L0.1_seed2.json'),
                     ('a', 'attack/reactive_r2_mix_v2_L0.1.json')):
        with open(os.path.join(RUN, rel)) as fh:
            ref[tag] = json.load(fh)
    rs = ref['s2' if seed2 else 's']
    cache, rows = {}, []
    for r in runs:
        base = build_base(r, train, cache)
        obj = base if r.get('cfg') is None else DL.RegAdapter(base, r['cfg'])
        c0 = time.perf_counter()
        out, us_all = {}, []
        for ds, songs in BR.bench3.eval_sets('val').items():
            acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                                 'on_time_right', 'changes', 'late_entry'), 0.0)
            acc2 = dict.fromkeys(('notes', 'attacks', 'fresh_clash', 'attack_miss', 'd', 'harsh_fresh', 'harsh_bleed'),
                                 0.0)
            up, uh = [], []
            for s in songs:
                notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
                meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
                T, H, a, b = BR.simulate(obj, meta, notes, L)
                up += a
                uh += b
                BR.score_song(s, T, H, L, acc)
                AM.song_counts(s, T, H, L, acc2)
            row = BR.row(acc, up, uh, len(songs))
            row['fresh_per100'] = 100 * acc2['fresh_clash'] / (acc2['notes'] or 1)
            out[ds] = row
            us_all += up + uh
        head = {k: sum(out[d][k] for d in DS) / 4 for k in ('rt_nd', 'harsh', 'fresh_per100', 'on_time_right')}
        head['us_mean'] = sum(us_all) / len(us_all)
        x = {'name': r['name'], 'seed': 'seed2' if seed2 else 'seed1', 'headline': head,
             'rows': {d: {k: out[d][k] for k in ('rt_nd', 'harsh', 'fresh_per100', 'on_time_right')} for d in DS},
             'd_vs_reactive': {'rt_nd': head['rt_nd'] - rs['headline']['rt_nd'],
                               'harsh': head['harsh'] - rs['headline']['harsh']},
             'secs': round(time.perf_counter() - c0, 1)}
        if isinstance(obj, DL.RegAdapter):
            sh = {}
            k = 0
            for ds, songs in BR.bench3.eval_sets('val').items():
                fin = [y['s'][-1][1] if y['s'] else 0 for y in obj.songs[k:k + len(songs)]]
                sh[ds] = sum(1 for v in fin if v)
                k += len(songs)
            x['songs_shifted'] = sh
        if not seed2:
            ra = ref['a']
            x['checks'] = {'rt_nd +0.005': head['rt_nd'] >= rs['headline']['rt_nd'] + 0.005,
                           'harsh <= reactive': head['harsh'] <= rs['headline']['harsh'],
                           'us < 2000': head['us_mean'] < 2000,
                           'fresh <= reactive': head['fresh_per100'] <= ra['headline']['fresh_clash_per100'] + 1e-12}
            for d in DS:
                x['checks'][f'{d} harsh'] = out[d]['harsh'] <= rs['rows'][d]['harsh'] + 0.005
                x['checks'][f'{d} rt_nd'] = out[d]['rt_nd'] >= rs['rows'][d]['rt_nd'] - 0.01
                x['checks'][f'{d} fresh'] = out[d]['fresh_per100'] <= ra['rows'][d]['fresh_clash_per100'] + 1.0
        rows.append(x)
        print(f"{r['name']:30s} {x['seed']}: rt_nd {head['rt_nd']:.4f} ({x['d_vs_reactive']['rt_nd']:+.4f}) harsh "
              f"{head['harsh']:.4f} fresh {head['fresh_per100']:.2f} us {head['us_mean']:.0f} | "
              + ' '.join(f"{d} {out[d]['rt_nd']:.4f}/{out[d]['harsh']:.4f}/{out[d]['fresh_per100']:.2f}" for d in DS)
              + (f" | shifted {x['songs_shifted']}" if 'songs_shifted' in x else '')
              + (f" | failed {[k for k, v in x['checks'].items() if not v]}" if 'checks' in x else '')
              + f" | {x['secs']:.0f} s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('live', 'bench'))
    ap.add_argument('--runs', required=True)
    ap.add_argument('--voices', default='sung_low')
    ap.add_argument('--songs', type=int, default=None)
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if os.path.exists(path) or not path.endswith('.json'):
        sys.exit('refusing: ' + path)
    runs = json.loads(a.runs)
    if a.mode == 'bench':
        BR.bench3.B2.heard = BR.heard_seed2 if a.seed2 else BR.HEARD1       # as bench_rt.evaluate, before building
    train = BR.bench3.train_sets()
    if a.mode == 'live':
        rows = live(runs, a.voices.split(','), a.songs, train)
    else:
        rows = bench(runs, a.seed2, train)
    with open(path, 'x') as fh:
        json.dump({'note': 'idea D dev runs; val only, tuned on val, no test split', 'mode': a.mode, 'rows': rows}, fh,
                  indent=1, default=LBR._js)
    print('wrote', path)


if __name__ == '__main__':
    main()
