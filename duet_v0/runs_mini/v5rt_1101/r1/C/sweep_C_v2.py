"""v5rt_1101 r1 idea C: DEV sweep of r1_C_core_v2 settings on val (tuned on val; no test split exists). Not a scored
run. Same as sweep_C_v1 (bench_rt.simulate / score_song / row and attack_metric.song_counts unchanged) plus a
calibration check of every attack plan() makes: predicted P(harsh) vs the actual interval against her TRUE note
sounding at that moment, split by kind ('on' = onset attack, 'rev' = held-note re-voice), with attacks per 100 notes.

  python sweep_C_v2.py <configs.json> --out <new .json under r1/C/> [--L 0.1] [--seed2]
"""
import argparse
import bisect
import importlib.util
import json
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402

_s = importlib.util.spec_from_file_location('attack_metric', os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'tools',
                                                                          'attack_metric.py'))
A = importlib.util.module_from_spec(_s)
_s.loader.exec_module(A)
from rt_systems import r1_C_core_v2 as C  # noqa: E402

KEYS = ('rt_nd', 'harsh', 'nd_first', 'harsh_first', 'on_time_right', 'changes', 'late_entry', 'us_mean')


def calib(song, atr, acc):
    true = sorted(song['sop'], key=lambda n: n[0])
    ons = [n[0] for n in true]
    z = [b['z'] for b in song['beats']]
    for q, kind, h, p_on, bad_on, bad in atr:
        if q < B.J:
            continue
        a = acc.setdefault(kind, dict(n=0, sing=0, pred=0.0, harsh=0, onset=0, hit=0))
        a['n'] += 1
        j = bisect.bisect_right(ons, q) - 1
        if j >= 0 and true[j][0] <= q < true[j][1]:
            m = true[j][2]
            a['sing'] += 1
            a['pred'] += bad
            a['harsh'] += abs(m - h) % 12 in B.HARSH
            k = int(q)
            a['hit'] += k < len(z) and z[k] >= 0 and B.hit(h, m, song['tonic'], z[k])
        jn = bisect.bisect_left(ons, q - 1e-9)
        a['onset'] += jn < len(ons) and ons[jn] < q + 0.01


def run(obj, cfg, val, L, seed2):
    obj.c = dict(cfg)
    heard = B.heard_seed2 if seed2 else B.HEARD1
    rows, cal = {}, {}
    for ds, songs in val.items():
        acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                             'on_time_right', 'changes', 'late_entry'), 0.0)
        acc2 = dict.fromkeys(('notes', 'attacks', 'fresh_clash', 'attack_miss', 'd', 'harsh_fresh', 'harsh_bleed'), 0.0)
        cacc = {}
        up, uh = [], []
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            obj.atr = []
            T, H, a, b = B.simulate(obj, meta, notes, L)
            up += a
            uh += b
            B.score_song(s, T, H, L, acc)
            A.song_counts(s, T, H, L, acc2)
            calib(s, obj.atr, cacc)
            obj.atr = None
        r = B.row(acc, up, uh, len(songs))
        n = acc2['notes'] or 1
        r['fresh_clash_per100'] = 100 * acc2['fresh_clash'] / n
        r['attack_miss_per100'] = 100 * acc2['attack_miss'] / n
        r['attacks_per100'] = 100 * acc2['attacks'] / n
        r['harsh_fresh'] = acc2['harsh_fresh'] / acc2['d']
        rows[ds] = r
        cal[ds] = {k: dict(v, per100=100 * v['n'] / n, pred_mean=v['pred'] / max(v['sing'], 1),
                           harsh_rate=v['harsh'] / max(v['sing'], 1), hit_rate=v['hit'] / max(v['sing'], 1),
                           onset_rate=v['onset'] / max(v['n'], 1)) for k, v in cacc.items()}
    head = {k: float(np.mean([r[k] for r in rows.values()])) for k in KEYS + ('fresh_clash_per100',
                                                                             'attack_miss_per100', 'attacks_per100',
                                                                             'harsh_fresh')}
    return head, rows, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--L', type=float, default=0.1)
    ap.add_argument('--seed2', action='store_true')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert out.startswith(HERE + os.sep) and not os.path.exists(out), out
    with open(a.configs) as fh:
        cfgs = json.load(fh)
    t0 = time.time()
    B.bench3.B2.heard = B.heard_seed2 if a.seed2 else B.HEARD1
    obj = C.build_with(B.bench3.train_sets(), cfgs[0]['cfg'])
    val = B.bench3.eval_sets('val')
    print(f'built in {time.time() - t0:.0f}s', flush=True)
    res = []
    for e in cfgs:
        c0 = time.time()
        head, rows, cal = run(obj, e['cfg'], val, a.L, a.seed2)
        res.append({'name': e['name'], 'cfg': e['cfg'], 'headline': head, 'rows': rows, 'calib': cal})
        print(f"{e['name']:22s} rt_nd {head['rt_nd']:.4f} harsh {head['harsh']:.4f} nd1 {head['nd_first']:.3f} "
              f"ontime {head['on_time_right']:.3f} chg {head['changes']:.3f} fresh {head['fresh_clash_per100']:.2f} "
              f"watt {head['attack_miss_per100']:.1f} | "
              + ' '.join(f"{ds} {r['rt_nd']:.4f}/{r['harsh']:.4f}/{r['fresh_clash_per100']:.2f}"
                         for ds, r in rows.items())
              + f" | us {head['us_mean']:.0f} ({time.time() - c0:.0f}s)", flush=True)
        for kind in ('on', 'rev'):
            if any(kind in cal[ds] for ds in cal):
                print(f"   {kind}: " + ' '.join(
                    f"{ds} n/100 {cal[ds][kind]['per100']:.1f} pred {cal[ds][kind]['pred_mean']:.3f} "
                    f"harsh {cal[ds][kind]['harsh_rate']:.3f} hit {cal[ds][kind]['hit_rate']:.3f} "
                    f"onset {cal[ds][kind]['onset_rate']:.2f};" for ds in cal if kind in cal[ds]), flush=True)
    with open(out, 'x') as fh:
        json.dump({'L': a.L, 'seed': 'seed2' if a.seed2 else 'seed1', 'split': 'val', 'results': res}, fh, indent=1,
                  default=float)


if __name__ == '__main__':
    main()
