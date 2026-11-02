"""Round 3, idea F tuning harness v5: tune_F_v4.py (read it; same frozen scoring pieces) plus the Twinkle phone file
(live_bench_rt.phone_run(..., 'phone_twinkle'): exact rt_nd_live / harsh / fresh against Twinkle's truth). Default
class r3_F_lib_v5.PlanF5. VAL ONLY, tuned on val (no test split); the phone files are two recordings, not val.
Offline: the _NoSound tripwire is installed before any other import. Appends to
runs_mini/v5rt_1101/r3/F/tune_log_F_v5.jsonl.

  .venv/bin/python runs_mini/v5rt_1101/r3/F/tune_F_v5.py [--lib r3_F_lib_v5 --cls PlanF5] [--no-bench]
      [--voices sung,sung_low] [--no-v11] [--no-tw] [--tag T] '<json>' ['<json>' ...]
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('offline tuning: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r3', 'F'))
import tune_F_v1 as T1  # noqa: E402

LBR = T1.LBR
LOG = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r3', 'F', 'tune_log_F_v5.jsonl')
BASE = dict(T1.BASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', default='r3_F_lib_v5')
    ap.add_argument('--cls', default='PlanF5')
    ap.add_argument('--no-bench', action='store_true')
    ap.add_argument('--voices', default='sung,sung_low')
    ap.add_argument('--no-v11', action='store_true')
    ap.add_argument('--no-tw', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('cfgs', nargs='+')
    a = ap.parse_args()
    Cls = getattr(importlib.import_module('rt_systems.' + a.lib), a.cls)
    train = T1.B.bench3.train_sets()
    mix = T1.r2_mix_v2.build(train)
    prior, tab, hm = T1.Prior(), T1.load_copy_tab(), T1.LIB3.load_hm_tab()
    voices = [v for v in a.voices.split(',') if v]
    for js in a.cfgs:
        cfg = dict(BASE, **json.loads(js))
        t0 = time.time()
        rec = {'time': time.strftime('%H:%M:%S'), 'tag': a.tag, 'lib': a.lib, 'cls': a.cls, 'over': json.loads(js),
               'cfg': cfg}
        parts = []
        if not a.no_bench:
            obj = Cls(mix, prior, cfg, tab, hm)
            head, rows = T1.bench(obj)
            rec['bench'] = {'head': head, 'rows': rows, 'stats': dict(obj.stats)}
            parts.append(f"bench {head['rt_nd']:.4f} h {head['harsh']:.4f} fr {head['fresh']:.2f} | " + ' '.join(
                f"{d} {r['rt_nd']:.4f}/{r['harsh']:.4f}/{r['fresh']:.2f}" for d, r in rows.items()))
        rec['live'] = {}
        for v in voices:
            obj = Cls(mix, prior, cfg, tab, hm)
            m = T1.live(obj, v)
            rec['live'][v] = m
            rec.setdefault('live_stats', {})[v] = dict(obj.stats)
            parts.append(f"{v} {m['rt_nd_live']:.4f} h {m['harsh']:.4f} fr {m['fresh_per100']:.2f} "
                         f"st {m['stumbles_per100']:.2f} wr {m['wrong_att_per100']:.1f}")
        if not a.no_v11:
            obj = Cls(mix, prior, cfg, tab, hm)
            out, st = LBR.phone_run(obj, 'tune', 'phone_v11')
            p = out['proxies']
            rec['v11'] = p
            parts.append(f"v11 {p['fresh_heard']} {p['harsh_heard']:.4f}")
        if not a.no_tw:
            obj = Cls(mix, prior, cfg, tab, hm)
            out, st = LBR.phone_run(obj, 'tune', 'phone_twinkle')
            m = out['metrics']
            rec['twinkle_phone'] = {'metrics': m, 'sampled_fit': out['sampled']['fit']}
            parts.append(f"tw {m['rt_nd_live']:.4f} h {m['harsh']:.4f} fr {m['fresh']} st {m['stumbles_per100']:.2f}")
        rec['secs'] = round(time.time() - t0, 1)
        print(' | '.join(parts) + f" | {rec['secs']:.0f}s | {js}", flush=True)
        with open(LOG, 'a') as fh:
            fh.write(json.dumps(rec, default=LBR._js) + '\n')


if __name__ == '__main__':
    main()
