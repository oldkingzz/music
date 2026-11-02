"""dev_G_v1 (run v5rt_1101, round 4, idea G 'late-hear policy'): dev / tuning harness, not a scored run. VAL ONLY, tuned
on val (this bundle has no test split). Offline: the _NoSound tripwire is installed before any other import; nothing is
played. Scored variants go through the frozen CLIs (live_bench_rt_v2.py, bench_rt.py, attack_metric.py).

Each config is one JSON object {"tag": "...", "over": {overrides on W's CFG (rt_systems.r3_X_regtrkst_v1.CFG)}}; the
system is rt_systems.<--lib>.make_base(...) inside idea D's register adapter with W's REG (r2_mix_v2 and the TRAIN
tables built once per process; a fresh system per cell). Cells: --cells voice@lat,... on the 17 val chorales through the
frozen live bench v2's own functions (live_bench_rt.rendered_song with live_bench_rt_v2's LAT table), compared with W's
phase-1 files live_bench/L4_rt_r3_X_regtrkst_v1_<voice>_<lat>.json (or the round-3 wired files). --brt L: also bench_rt
(+ attack_metric's fresh clashes) on all of val at that L, seed 1 (dev_X_v1.bench, unchanged pieces). Appends one JSON
line per config to runs_mini/v5rt_1101/r4/G/dev_log_G_v1.jsonl.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r4/G/dev_G_v1.py --lib r4_G_lib_v1 \
      --cells sung@bt_cal,sung_low@bt_cal [--brt 0.5] '<json>' ...
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('dev_G_v1 is offline: no sound device may be used (sounddevice.' + name + ')')


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
import live_bench_rt_v2 as LB2  # noqa: E402  (frozen v2: sha checks, tripwire, round-4 LAT settings)

LB1 = LB2.LB1
sys.path.insert(0, os.path.join(RUN, 'r3', 'X'))
import dev_X_v1 as D1  # noqa: E402  (bench(obj, heard, L) and stats_of, unchanged)

from systems import r2_mix_v2  # noqa: E402
from rt_systems import r2_lib_v3 as R3  # noqa: E402
from rt_systems import r3_D_lib_v3 as DL  # noqa: E402
from rt_systems import r3_E_lib_v2 as EL  # noqa: E402
from rt_systems import r3_X_regtrkst_v1 as W  # noqa: E402
from rt_systems.r2_lib_v1 import Prior, load_copy_tab  # noqa: E402

LOG = os.path.join(RUN, 'r4', 'G', 'dev_log_G_v1.jsonl')
KEYS = ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100', 'fresh', 'stumbles_own_per100', 'on_time_right',
        'wrong_att_per100', 'late_entry', 'notes')


def w_file(voice, lat):
    if lat == 'wired':
        p = os.path.join(RUN, 'live_bench', f'R3X_rt_r3_X_regtrkst_v1_{voice}_wired.json')
    else:
        p = os.path.join(RUN, 'live_bench', f'L4_rt_r3_X_regtrkst_v1_{voice}_{lat}.json')
    with open(p) as fh:
        return json.load(fh)['headline']


def live(obj, voice, lat_name):
    lat = LB1.LAT[lat_name]
    acc = LB1.new_acc()
    for s in LB1.bench3.eval_sets('val')['bach']:
        a, r, tr, ex, st = LB1.rendered_song(obj, 'dev', s, voice, lat)
        for k in LB1.ACC_KEYS:
            acc[k] += a[k]
    return LB1.metrics(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', required=True)
    ap.add_argument('--cells', default='sung@bt_cal,sung_low@bt_cal')
    ap.add_argument('--brt', type=float, default=None)
    ap.add_argument('cfgs', nargs='+')
    a = ap.parse_args()
    if not a.lib.startswith('r4_G_lib_'):
        sys.exit('--lib must be an r4_G_lib_* module')
    GL = importlib.import_module('rt_systems.' + a.lib)
    train = LB1.bench3.train_sets()
    mix = r2_mix_v2.build(train)
    prior, tab, hm, trk = Prior(), load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab()
    cells = [tuple(x.split('@')) for x in a.cells.split(',')]
    base_w = {cl: w_file(*cl) for cl in cells}

    def build(over):
        cfg = dict(W.CFG, **over)
        return DL.RegAdapter(GL.make_base(mix, prior, cfg, tab, hm, trk), dict(W.REG)), cfg

    for js in a.cfgs:
        spec = json.loads(js)
        t0 = time.time()
        rec = {'time': time.strftime('%H:%M:%S'), 'lib': a.lib, 'spec': spec, 'cells': {}}
        parts = [spec.get('tag', '')]
        for cl in cells:
            obj, cfg = build(spec.get('over', {}))
            m = live(obj, *cl)
            w = base_w[cl]
            rec['cells']['@'.join(cl)] = {'m': {k: m[k] for k in KEYS}, 'stats': D1.stats_of(obj),
                                          'd': {k: m[k] - w[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100',
                                                                         'stumbles_per100')}}
            k2 = (m['harsh'] <= w['harsh'] + 0.003 and m['fresh_per100'] <= w['fresh_per100'] + 0.5
                  and m['stumbles_per100'] <= w['stumbles_per100'] + 1.0)
            parts.append(f"{cl[0]}@{cl[1]} {m['rt_nd_live']:.4f}({m['rt_nd_live'] - w['rt_nd_live']:+.4f})/"
                         f"{m['harsh']:.4f}({m['harsh'] - w['harsh']:+.4f})/{m['fresh_per100']:.2f}"
                         f"({m['fresh_per100'] - w['fresh_per100']:+.2f})/{m['stumbles_per100']:.2f}"
                         f"({m['stumbles_per100'] - w['stumbles_per100']:+.2f}){'' if k2 else ' K2x'}")
        if a.brt is not None:
            obj, cfg = build(spec.get('over', {}))
            head, rows = D1.bench(obj, D1.B.HEARD1, a.brt)
            rec['brt'] = {'L': a.brt, 'head': head, 'rows': rows, 'stats': D1.stats_of(obj)}
            parts.append(f"brt L{a.brt} {head['rt_nd']:.4f}/{head['harsh']:.4f}/{head['fresh']:.2f}")
        rec['cfg'] = cfg
        rec['secs'] = round(time.time() - t0, 1)
        print(' | '.join(parts) + f" | {rec['secs']:.0f}s", flush=True)
        with open(LOG, 'a') as fh:
            fh.write(json.dumps(rec, default=LB1._js) + '\n')


if __name__ == '__main__':
    main()
