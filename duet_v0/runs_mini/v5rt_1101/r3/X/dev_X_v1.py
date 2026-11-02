"""dev_X_v1 (run v5rt_1101, round 3, combine step): dev / tuning harness for the combined variants built from
rt_systems.r3_X_lib_v1 (idea E's tracker-aware hear + idea F's stumble guards, optionally inside idea D's register
adapter). VAL ONLY, tuned on val (this bundle has no test split). Offline: the _NoSound tripwire is installed before any
other import; nothing is played. Not a scored run: the scored variants go through the frozen bench_rt.py,
attack_metric.py and live_bench_rt.py CLIs.

Each config is one JSON object:
  {"tag": "...", "e": "trk" | "trkhold" | "trkholdk" | "trksafe"   (the idea-E system whose CFG is the starting point),
   "over": {cfg overrides, e.g. idea F's kappa_st / hop_win / theta_hop / guard / rec_win / g_mu / g_sil / g_above},
   "reg": null | {overrides on r3_D_reghm_v1.CFG}   (null = no register adapter)}
For each config, built afresh for every part (r2_mix_v2 / tables built once per process):
  bench  bench_rt.simulate / score_song + attack_metric.song_counts (unchanged) on all of val at L = 0.1, seed 1
         (and seed 2 with --seed2)                                                         -> G1 numbers
  live   live_bench_rt.make_engine / run_engine / score / metrics (unchanged) on the 17 val chorales, wired latency,
         voices sung and sung_low (renders cached: they do not depend on the system)          -> G2 numbers
  v11    live_bench_rt.phone_run(..., 'phone_v11') proxies                                    -> G3 numbers
  tw     live_bench_rt.phone_run(..., 'phone_twinkle') exact metrics (reported only)
Prints one line per config with the gate checks against reactive:r2_mix_v2's files on disk and appends one JSON line to
runs_mini/v5rt_1101/r3/X/dev_log_X_v1.jsonl.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/X/dev_X_v1.py [--seed2] [--no-bench] [--no-tw] '<json>' ...
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('dev_X_v1 is offline: no sound device may be used (sounddevice.' + name + ')')


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
import live_bench_rt as LBR  # noqa: E402  (its own tripwire too; bench_rt and the rt_systems path)
import attack_metric as AM  # noqa: E402

import numpy as np  # noqa: E402

from systems import r2_mix_v2  # noqa: E402
from rt_systems import r2_lib_v3 as R3  # noqa: E402
from rt_systems import r3_D_lib_v3 as DL  # noqa: E402
from rt_systems import r3_E_lib_v2 as EL  # noqa: E402
from rt_systems import r3_X_lib_v1 as XL  # noqa: E402
from rt_systems.r2_lib_v1 import Prior, load_copy_tab  # noqa: E402
from rt_systems.r3_D_reghm_v1 import CFG as REG0  # noqa: E402

B = LBR.BR
DS = ('bach', 'folk', 'pop', 'pop909')
LOG = os.path.join(RUN, 'r3', 'X', 'dev_log_X_v1.jsonl')
FRESH_MAX = 1.8100634866373206


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def reactive():
    s, s2, a = (load('scores/B_reactive_r2_mix_v2_L0.1.json'), load('scores/B_reactive_r2_mix_v2_L0.1_seed2.json'),
                load('attack/reactive_r2_mix_v2_L0.1.json'))
    g1 = {'rt_nd': s['headline']['rt_nd'], 'harsh': s['headline']['harsh'], 'seed2': s2['headline']['rt_nd'],
          'fresh': a['headline']['fresh_clash_per100'], 'ds_rt_nd': {d: s['rows'][d]['rt_nd'] for d in DS},
          'ds_harsh': {d: s['rows'][d]['harsh'] for d in DS},
          'ds_fresh': {d: a['rows'][d]['fresh_clash_per100'] for d in DS}}
    lv = {v: load(f'live_bench/L_reactive_r2_mix_v2_{v}_wired.json')['headline'] for v in ('sung', 'sung_low')}
    lv['v11'] = load('live_bench/L_reactive_r2_mix_v2_phone_v11.json')['phone']['proxies']
    return g1, lv


def bench(obj, heard, L=0.1):
    """bench_rt.evaluate's loop (simulate / score_song / row) plus attack_metric.song_counts, unchanged pieces."""
    B.bench3.B2.heard = heard
    rows, att, us = {}, {}, []
    for ds, songs in B.bench3.eval_sets('val').items():
        acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                             'on_time_right', 'changes', 'late_entry'), 0.0)
        aacc = dict.fromkeys(('notes', 'attacks', 'fresh_clash', 'attack_miss', 'd', 'harsh_fresh', 'harsh_bleed'), 0.0)
        up, uh = [], []
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, a, b = B.simulate(obj, meta, notes, L)
            up += a
            uh += b
            B.score_song(s, T, H, L, acc)
            AM.song_counts(s, T, H, L, aacc)
        rows[ds] = B.row(acc, up, uh, len(songs))
        us += up + uh
        n = aacc['notes'] or 1
        att[ds] = 100 * aacc['fresh_clash'] / n
    head = {'rt_nd': sum(r['rt_nd'] for r in rows.values()) / 4, 'harsh': sum(r['harsh'] for r in rows.values()) / 4,
            'fresh': sum(att.values()) / 4, 'us': sum(us) / len(us),
            'on_time_right': sum(r['on_time_right'] for r in rows.values()) / 4}
    return head, {d: {'rt_nd': rows[d]['rt_nd'], 'harsh': rows[d]['harsh'], 'fresh': att[d]} for d in DS}


RENDER = {}


def live(obj, voice):
    """live_bench_rt.rendered_song's engine run and score (unchanged pieces) on the 17 val chorales, wired latency."""
    lat = LBR.LAT['wired']
    T_b = 60.0 / LBR.BPM
    acc = LBR.new_acc()
    for song in LBR.bench3.eval_sets('val')['bach']:
        facts = {'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'bpm': LBR.BPM}
        eng = LBR.make_engine(obj, 'dev', facts, LBR.COUNT_IN)
        t0h = eng.t0 + (lat['Lo'] - lat['Lo_told'])
        key = (song['id'], voice, round(t0h, 6))
        if key not in RENDER:
            RENDER[key] = LBR.VV.render(song, voice, t0h, T_b)
        y, her, vf = RENDER[key]
        x = np.concatenate([np.zeros(int(round(lat['Li'] * LBR.SR))), y.astype(np.float64), np.zeros(LBR.SR)])
        track, notes, blips, secs = LBR.run_engine(eng, x, lat)
        LBR.score(her, sorted(notes + blips), [bt['z'] for bt in song['beats']], song['tonic'], t0h, T_b, acc)
    return LBR.metrics(acc)


def stats_of(obj):
    base = obj.base if isinstance(obj, DL.RegAdapter) else obj
    return dict(base.stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--no-bench', action='store_true')
    ap.add_argument('--no-tw', action='store_true')
    ap.add_argument('cfgs', nargs='+')
    a = ap.parse_args()
    train = B.bench3.train_sets()
    mix = r2_mix_v2.build(train)
    prior, tab, hm, trk = Prior(), load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab()
    rg, rl = reactive()

    def build(spec):
        cfg = dict(importlib.import_module('rt_systems.r3_E_' + spec['e'] + '_v1').CFG, **spec.get('over', {}))
        base = XL.make_base(mix, prior, cfg, tab, hm, trk)
        if spec.get('reg') is None:
            return base, cfg, None
        reg = dict(REG0, **spec['reg'])
        return DL.RegAdapter(base, reg), cfg, reg

    for js in a.cfgs:
        spec = json.loads(js)
        t0 = time.time()
        rec = {'time': time.strftime('%H:%M:%S'), 'spec': spec}
        parts, fails = [spec.get('tag', '')], []
        if not a.no_bench:
            obj, cfg, reg = build(spec)
            head, rows = bench(obj, B.HEARD1)
            rec['bench'] = {'head': head, 'rows': rows, 'stats': stats_of(obj)}
            c = {'rt_nd +0.005': head['rt_nd'] >= rg['rt_nd'] + 0.005, 'harsh': head['harsh'] <= rg['harsh'],
                 'fresh': head['fresh'] <= FRESH_MAX + 1e-12, 'us': head['us'] < 2000}
            for d in DS:
                c[d + ' harsh'] = rows[d]['harsh'] <= rg['ds_harsh'][d] + 0.005
                c[d + ' rt_nd'] = rows[d]['rt_nd'] >= rg['ds_rt_nd'][d] - 0.01
                c[d + ' fresh'] = rows[d]['fresh'] <= rg['ds_fresh'][d] + 1.0
            txt = f"bench {head['rt_nd']:.4f}/{head['harsh']:.4f}/{head['fresh']:.3f}"
            if a.seed2:
                obj, cfg, reg = build(spec)
                h2, _ = bench(obj, B.heard_seed2)
                B.bench3.B2.heard = B.HEARD1
                rec['bench']['seed2'] = h2['rt_nd']
                c['seed2 +0.003'] = h2['rt_nd'] - rg['seed2'] >= 0.003
                txt += f" s2d {h2['rt_nd'] - rg['seed2']:+.4f}"
            fails += ['G1 ' + k for k, v in c.items() if not v]
            parts.append(txt)
        rec['live'] = {}
        for v in ('sung', 'sung_low'):
            obj, cfg, reg = build(spec)
            m = live(obj, v)
            rec['live'][v] = {k: m[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100',
                                                 'stumbles_own_per100', 'on_time_right', 'wrong_att_per100',
                                                 'late_entry', 'blips', 'notes')}
            rec['live'][v]['stats'] = stats_of(obj)
            r = rl[v]
            need = r['rt_nd_live'] + (0.005 if v == 'sung' else 0.0)
            c = {v + ' rt_nd': m['rt_nd_live'] >= need, v + ' harsh': m['harsh'] <= r['harsh'],
                 v + ' fresh': m['fresh_per100'] <= r['fresh_per100']}
            fails += ['G2 ' + k for k, x in c.items() if not x]
            parts.append(f"{v} {m['rt_nd_live']:.4f}/{m['harsh']:.4f}/{m['fresh_per100']:.2f} st "
                         f"{m['stumbles_per100']:.2f}")
        obj, cfg, reg = build(spec)
        out, st = LBR.phone_run(obj, 'dev', 'phone_v11')
        p = out['proxies']
        rec['v11'] = p
        rec['v11_stats'] = stats_of(obj)
        rec['v11_answered'] = out['answered']
        rec['v11_duet_s'] = out['duet_s']
        if p['fresh_heard'] > rl['v11']['fresh_heard']:
            fails.append('G3 fresh_heard')
        if p['harsh_heard'] > rl['v11']['harsh_heard'] + 0.005:
            fails.append('G3 harsh_heard')
        parts.append(f"v11 {p['fresh_heard']}/{p['harsh_heard']:.4f} sung {p['sung']} att {p['attacks']}")
        if not a.no_tw:
            obj, cfg, reg = build(spec)
            out, st = LBR.phone_run(obj, 'dev', 'phone_twinkle')
            m = out['metrics']
            rec['tw'] = {k: m[k] for k in ('rt_nd_live', 'harsh', 'fresh', 'stumbles_per100', 'on_time_right')}
            rec['tw']['sampled_fit'] = out['sampled']['fit']
            parts.append(f"tw {m['rt_nd_live']:.4f}/{m['harsh']:.4f}/{m['fresh']} st {m['stumbles_per100']:.2f}")
        if 'sung' in rec['live'] and 'sung_low' in rec['live']:
            mean = (rec['live']['sung']['rt_nd_live'] + rec['live']['sung_low']['rt_nd_live']) / 2
            fs = rec['live']['sung']['fresh'] + rec['live']['sung_low']['fresh']
            rec['mean_live'], rec['fresh_live_sum'] = mean, fs
            parts.append(f"mean {mean:.4f} fsum {fs}")
        rec['fails'] = fails
        rec['cfg'] = cfg
        rec['reg'] = reg
        rec['secs'] = round(time.time() - t0, 1)
        parts.append('PASS' if not fails else 'FAIL ' + ', '.join(fails))
        print(' | '.join(parts) + f" | {rec['secs']:.0f}s", flush=True)
        with open(LOG, 'a') as fh:
            fh.write(json.dumps(rec, default=LBR._js) + '\n')


if __name__ == '__main__':
    main()
