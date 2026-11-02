"""dev_X_v2 (run v5rt_1101, round 3, combine step): dev_X_v1.py (read it; its bench / live / reactive helpers are imported
unchanged) with the combine library chosen by --lib (rt_systems.r3_X_lib_v1 or r3_X_lib_v2, any module with make_base)
and the library recorded in each log line. VAL ONLY, tuned on val (this bundle has no test split). Offline: the _NoSound
tripwire is installed before any other import; nothing is played. Appends to runs_mini/v5rt_1101/r3/X/dev_log_X_v2.jsonl.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/X/dev_X_v2.py --lib r3_X_lib_v2 [--seed2] [--no-bench] [--no-tw]
      '<json>' ...
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('dev_X_v2 is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/r3/X')
import dev_X_v1 as D1  # noqa: E402  (installs the same tripwire; helpers bench, live, reactive, stats_of)

LOG = os.path.join(D1.RUN, 'r3', 'X', 'dev_log_X_v2.jsonl')
B, DS, LBR = D1.B, D1.DS, D1.LBR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', required=True)
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--no-bench', action='store_true')
    ap.add_argument('--no-tw', action='store_true')
    ap.add_argument('cfgs', nargs='+')
    a = ap.parse_args()
    if not a.lib.startswith('r3_X_lib_'):
        sys.exit('--lib must be an r3_X_lib_* module')
    XL = importlib.import_module('rt_systems.' + a.lib)
    train = B.bench3.train_sets()
    mix = D1.r2_mix_v2.build(train)
    prior, tab, hm, trk = D1.Prior(), D1.load_copy_tab(), D1.R3.load_hm_tab(), D1.EL.load_trk_tab()
    rg, rl = D1.reactive()

    def build(spec):
        cfg = dict(importlib.import_module('rt_systems.r3_E_' + spec['e'] + '_v1').CFG, **spec.get('over', {}))
        base = XL.make_base(mix, prior, cfg, tab, hm, trk)
        if spec.get('reg') is None:
            return base, cfg, None
        reg = dict(D1.REG0, **spec['reg'])
        return D1.DL.RegAdapter(base, reg), cfg, reg

    for js in a.cfgs:
        spec = json.loads(js)
        t0 = time.time()
        rec = {'time': time.strftime('%H:%M:%S'), 'lib': a.lib, 'spec': spec}
        parts, fails = [spec.get('tag', '')], []
        if not a.no_bench:
            obj, cfg, reg = build(spec)
            head, rows = D1.bench(obj, B.HEARD1)
            rec['bench'] = {'head': head, 'rows': rows, 'stats': D1.stats_of(obj)}
            c = {'rt_nd +0.005': head['rt_nd'] >= rg['rt_nd'] + 0.005, 'harsh': head['harsh'] <= rg['harsh'],
                 'fresh': head['fresh'] <= D1.FRESH_MAX + 1e-12, 'us': head['us'] < 2000}
            for d in DS:
                c[d + ' harsh'] = rows[d]['harsh'] <= rg['ds_harsh'][d] + 0.005
                c[d + ' rt_nd'] = rows[d]['rt_nd'] >= rg['ds_rt_nd'][d] - 0.01
                c[d + ' fresh'] = rows[d]['fresh'] <= rg['ds_fresh'][d] + 1.0
            txt = f"bench {head['rt_nd']:.4f}/{head['harsh']:.4f}/{head['fresh']:.3f}"
            if a.seed2:
                obj, cfg, reg = build(spec)
                h2, _ = D1.bench(obj, B.heard_seed2)
                B.bench3.B2.heard = B.HEARD1
                rec['bench']['seed2'] = h2['rt_nd']
                c['seed2 +0.003'] = h2['rt_nd'] - rg['seed2'] >= 0.003
                txt += f" s2d {h2['rt_nd'] - rg['seed2']:+.4f}"
            fails += ['G1 ' + k for k, v in c.items() if not v]
            parts.append(txt)
        rec['live'] = {}
        for v in ('sung', 'sung_low'):
            obj, cfg, reg = build(spec)
            m = D1.live(obj, v)
            rec['live'][v] = {k: m[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100',
                                                 'stumbles_own_per100', 'on_time_right', 'wrong_att_per100',
                                                 'late_entry', 'blips', 'notes')}
            rec['live'][v]['stats'] = D1.stats_of(obj)
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
        rec['v11_stats'] = D1.stats_of(obj)
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
