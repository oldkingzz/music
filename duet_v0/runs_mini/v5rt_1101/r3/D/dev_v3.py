"""Idea D dev runs (run v5rt_1101, round 3): dev_v2.py with the adapter library chosen by --lib (r3_D_lib_v1 or
r3_D_lib_v2) and, for phone_v11, the engine run kept so that every fresh_heard attack and the notes with harsh_heard
time are listed (the proxies are live_rt_eval_v2.heard_proxy's, as live_bench_rt.phone_run computes them). Both modes run
the frozen benches' own code; offline: the _NoSound tripwire is installed before any other import and nothing is played.
Val only, tuned on val (no test split).

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/D/dev_v3.py --lib r3_D_lib_v2 --mode live --runs '<json>' \
      --voices sung_low,sung,phone_v11 [--songs N] --out runs_mini/v5rt_1101/r3/D/<new>.json
  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/D/dev_v3.py --lib r3_D_lib_v2 --mode bench --runs '<json>' [--seed2] \
      --out runs_mini/v5rt_1101/r3/D/<new>.json
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('dev_v3 is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
sys.path.insert(0, os.path.join(RUN, 'tools'))
import live_bench_rt as LBR  # noqa: E402  (its own tripwire too; bench_rt, rt_systems paths)
import attack_metric as AM  # noqa: E402

BR = LBR.BR
EV = LBR.EV
KEYS = ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100', 'stumbles_own_per100', 'on_time_right',
        'wrong_att_per100', 'late_entry', 'double', 'first_100ms_hit', 'first_100ms_harsh', 'attacks_per100', 'blips')
DS = ('bach', 'folk', 'pop', 'pop909')
NAMES = ('C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B')
DL = None


def nn(m):
    return f'{NAMES[m % 12]}{m // 12 - 1}' if m is not None and m >= 0 else '-'


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


def wrap(base, r):
    return base if r.get('cfg') is None else DL.RegAdapter(base, r['cfg'])


def v11_run(system, name):
    """live_bench_rt.phone_run('phone_v11') with the engine kept: proxies + the fresh_heard attacks + harsh notes."""
    cfg = EV.V11
    x = EV.read(cfg['file']).astype(np.float64)
    facts = {'tonic': cfg['tonic'], 'M': cfg['M'], 'K': cfg['K'], 'bpm': cfg['bpm']}
    eng = LBR.make_engine(system, name, facts, 0)
    track, notes, blips, secs = LBR.run_engine(eng, x, LBR.PHONE_LAT)
    prox = EV.heard_proxy(eng, track, notes, blips, EV.LO)
    t_out = (np.arange(len(track)) + 2) * LBR.HOP / LBR.SR + EV.LO
    her = [(float(on), float(off), int(m), None) for on, off, m in eng.notes]
    dec = {round(d['t'], 4): d for d in eng.decisions}
    fresh = []
    for xa, h in EV.attacks(notes):
        nt = EV.her_at(xa, her)
        if nt is not None and abs(nt[2] - h) % 12 in LBR.HARSH:
            d = dec.get(round(xa, 4))
            fresh.append([round(xa, 3), nn(h), nn(nt[2]), None if d is None else f"{d['kind']}{d['q'] if d['q'] is not None else d['i']}"])
    harsh_notes = []
    for k, (on, off, m, _) in enumerate(her):
        idx = np.where((t_out >= on) & (t_out < off))[0]
        bad = sum(1 for i in idx if track[i] >= 0 and abs(m - int(track[i])) % 12 in LBR.HARSH)
        if bad:
            harsh_notes.append([k, round(on, 2), nn(m), int(bad), int(len(idx))])
    return {'proxies': prox, 'fresh_attacks': fresh, 'harsh_notes': harsh_notes,
            'answered': [[e['onset_beats'], e['her_heard'], e['duet'], e['action']] for e in eng.events],
            'decision_counts': eng.summary()['decision_counts']}


def live(runs, voices, n_songs, train):
    cache, rows = {}, []
    for r in runs:
        base = build_base(r, train, cache)
        for voice in voices:
            system = wrap(base, r)
            c0 = time.perf_counter()
            row = {'name': r['name'], 'base': r['base'], 'base_cfg': r.get('base_cfg'), 'cfg': r.get('cfg'),
                   'voice': voice}
            if voice == 'phone_v11':
                row.update(v11_run(system, 'dev:' + r['name']))
            else:
                res = LBR.evaluate('dev:' + r['name'], voice, 'wired', n_songs, system=system)
                row['digest'] = res['digest']
                if voice == 'phone_twinkle':
                    row['metrics'] = {k: res['phone']['metrics'][k] for k in KEYS}
                    row['proxies'] = res['phone']['proxies']
                else:
                    row['headline'] = {k: res['headline'][k] for k in KEYS}
                    row['twinkle'] = {k: res['twinkle'][k] for k in KEYS}
                    row['per_song'] = [{'id': s['id'], 'm': {k: s['metrics'][k] for k in ('rt_nd_live', 'harsh', 'fresh',
                                                                                           'stumbles_per100')},
                                        'clashes': s['fresh_clashes']} for s in res['songs']]
                    row['decision_counts'] = res['decision_counts']
            row['secs'] = round(time.perf_counter() - c0, 1)
            if system is not base:
                row['adapter'] = system.songs
            rows.append(row)
            if voice == 'phone_v11':
                p = row['proxies']
                print(f"{r['name']:32s} {voice:9s} answered {p['answered']} sung {p['sung']} attacks {p['attacks']} "
                      f"cons_heard {p['cons_heard']} changes {p['changes']} on_time_cons {p['on_time_cons']} "
                      f"fresh_heard {p['fresh_heard']} harsh_heard {p['harsh_heard']:.4f}"
                      + (f" first {row['adapter'][-1]['first']} s {row['adapter'][-1]['s']}" if 'adapter' in row else ''),
                      flush=True)
                print(f"{'':32s} fresh: {row['fresh_attacks']}", flush=True)
                print(f"{'':32s} harsh notes [k, on, her, bad, samples]: {row['harsh_notes']}", flush=True)
            elif voice == 'phone_twinkle':
                m = row['metrics']
                print(f"{r['name']:32s} {voice:9s} rt_nd_live {m['rt_nd_live']:.4f} harsh {m['harsh']:.4f} "
                      f"fresh/100 {m['fresh_per100']:.2f} stumbles/100 {m['stumbles_per100']:.2f}", flush=True)
            else:
                m, t = row['headline'], row['twinkle']
                sh = ''
                if 'adapter' in row:
                    fin = [x['s'][-1][1] if x['s'] else 0 for x in row['adapter']]
                    sh = f" | shifted {sum(1 for v in fin[:-1] if v)}/{len(fin) - 1} twinkle s {fin[-1]}"
                print(f"{r['name']:32s} {voice:9s} rt_nd_live {m['rt_nd_live']:.4f} harsh {m['harsh']:.4f} "
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
        obj = wrap(base, r)
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
        if obj is not base:
            sh, k = {}, 0
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
        print(f"{r['name']:32s} {x['seed']}: rt_nd {head['rt_nd']:.4f} ({x['d_vs_reactive']['rt_nd']:+.4f}) harsh "
              f"{head['harsh']:.4f} fresh {head['fresh_per100']:.2f} us {head['us_mean']:.0f} | "
              + ' '.join(f"{d} {out[d]['rt_nd']:.4f}/{out[d]['harsh']:.4f}/{out[d]['fresh_per100']:.2f}" for d in DS)
              + (f" | shifted {x['songs_shifted']}" if 'songs_shifted' in x else '')
              + (f" | failed {[k for k, v in x['checks'].items() if not v]}" if 'checks' in x else '')
              + f" | {x['secs']:.0f} s", flush=True)
    return rows


def main():
    global DL
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', required=True, choices=('r3_D_lib_v1', 'r3_D_lib_v2'))
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
    DL = importlib.import_module('rt_systems.' + a.lib)
    runs = json.loads(a.runs)
    if a.mode == 'bench':
        BR.bench3.B2.heard = BR.heard_seed2 if a.seed2 else BR.HEARD1       # as bench_rt.evaluate, before building
    train = BR.bench3.train_sets()
    rows = live(runs, a.voices.split(','), a.songs, train) if a.mode == 'live' else bench(runs, a.seed2, train)
    with open(path, 'x') as fh:
        json.dump({'note': 'idea D dev runs; val only, tuned on val, no test split', 'lib': a.lib, 'mode': a.mode,
                   'rows': rows}, fh, indent=1, default=LBR._js)
    print('wrote', path)


if __name__ == '__main__':
    main()
