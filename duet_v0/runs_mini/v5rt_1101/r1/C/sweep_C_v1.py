"""v5rt_1101 r1 idea C: DEV sweep of r1_C_core_v1 settings on val (tuned on val; no test split exists). Not a scored
run: scored variants are fixed files futures/v5_rt/rt_systems/r1_C_<name>_v<k>.py scored with bench_rt's own CLI.
Builds JointRT once, then for each config of a JSON list runs bench_rt.simulate / score_song / row unchanged and
attack_metric.song_counts unchanged on every val song, and prints rt_nd, harsh, nd_first, on_time_right, changes,
fresh clashes per 100 notes per dataset. Optional --diag: chord / pc forecast accuracy at plan points with an onset.

  python sweep_C_v1.py <configs.json> --out <new .json under r1/C/> [--L 0.1] [--seed2] [--diag]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402  (puts the v3 and duet_beat paths in place)

_s = importlib.util.spec_from_file_location('attack_metric', os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'tools',
                                                                          'attack_metric.py'))
A = importlib.util.module_from_spec(_s)
_s.loader.exec_module(A)
from rt_systems import r1_C_core_v1 as C  # noqa: E402
from rt_systems.r1_C_fnet_v1 import query_feat  # noqa: E402

KEYS = ('rt_nd', 'harsh', 'nd_first', 'harsh_first', 'on_time_right', 'changes', 'late_entry', 'us_mean')


def run(obj, cfg, val, L, seed2, diag=None):
    obj.c = dict(cfg)
    heard = B.heard_seed2 if seed2 else B.HEARD1
    rows = {}
    for ds, songs in val.items():
        acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                             'on_time_right', 'changes', 'late_entry'), 0.0)
        acc2 = dict.fromkeys(('notes', 'attacks', 'fresh_clash', 'attack_miss', 'd', 'harsh_fresh', 'harsh_bleed'), 0.0)
        up, uh = [], []
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            if diag is not None:
                obj.trace = []
            T, H, a, b = B.simulate(obj, meta, notes, L)
            up += a
            uh += b
            B.score_song(s, T, H, L, acc)
            A.song_counts(s, T, H, L, acc2)
            if diag is not None:
                diag_song(s, obj.trace, diag.setdefault(ds, {}))
                obj.trace = None
        r = B.row(acc, up, uh, len(songs))
        n = acc2['notes'] or 1
        r['fresh_clash_per100'] = 100 * acc2['fresh_clash'] / n
        r['attack_miss_per100'] = 100 * acc2['attack_miss'] / n
        r['attacks_per100'] = 100 * acc2['attacks'] / n
        r['harsh_fresh'] = acc2['harsh_fresh'] / acc2['d']
        rows[ds] = r
    head = {k: float(np.mean([r[k] for r in rows.values()])) for k in KEYS + ('fresh_clash_per100',
                                                                             'attack_miss_per100', 'attacks_per100',
                                                                             'harsh_fresh')}
    return head, rows


def diag_song(s, trace, acc):
    """Forecast quality at plan points q >= J where a TRUE note starts in [q, q + 0.01): chord top-1 of the mixed chord
    marginal against the chord of beat int(q); pc top-1 / top-3; mean P(onset) at onset and non-onset points."""
    true = sorted(s['sop'], key=lambda n: n[0])
    ons = np.array([n[0] for n in true])
    z = [b['z'] for b in s['beats']]
    for q, p_on, ppc, pz, p_z in trace:
        if q < B.J:
            continue
        j = int(np.searchsorted(ons, q - 1e-9))
        has = j < len(ons) and ons[j] < q + 0.01
        acc['n_pts'] = acc.get('n_pts', 0) + 1
        acc['p_on_sum_' + ('on' if has else 'off')] = acc.get('p_on_sum_' + ('on' if has else 'off'), 0) + p_on
        acc['n_' + ('on' if has else 'off')] = acc.get('n_' + ('on' if has else 'off'), 0) + 1
        if not has:
            continue
        pc = (true[j][2] - s['tonic']) % 12
        o = np.argsort(-ppc)
        acc['pc1'] = acc.get('pc1', 0) + (o[0] == pc)
        acc['pc3'] = acc.get('pc3', 0) + (pc in o[:3])
        acc['p_true_pc'] = acc.get('p_true_pc', 0) + float(ppc[pc])
        k = int(q)
        if k < len(z) and z[k] >= 0:
            acc['nz'] = acc.get('nz', 0) + 1
            acc['z1'] = acc.get('z1', 0) + (int(np.argmax(pz)) == z[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--L', type=float, default=0.1)
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--diag', action='store_true')
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
        diag = {} if a.diag and e.get('diag') else None
        head, rows = run(obj, e['cfg'], val, a.L, a.seed2, diag)
        res.append({'name': e['name'], 'cfg': e['cfg'], 'headline': head, 'rows': rows, 'diag': diag})
        print(f"{e['name']:28s} rt_nd {head['rt_nd']:.4f} harsh {head['harsh']:.4f} nd1 {head['nd_first']:.3f} "
              f"hr1 {head['harsh_first']:.3f} ontime {head['on_time_right']:.3f} chg {head['changes']:.3f} "
              f"fresh {head['fresh_clash_per100']:.2f} watt {head['attack_miss_per100']:.1f} | "
              + ' '.join(f"{ds} {r['rt_nd']:.4f}/{r['harsh']:.4f}/{r['fresh_clash_per100']:.2f}"
                         for ds, r in rows.items())
              + f" | us {head['us_mean']:.0f} ({time.time() - c0:.0f}s)", flush=True)
        if diag:
            for ds, d in diag.items():
                print(f"   diag {ds}: P(on) at onsets {d['p_on_sum_on'] / d['n_on']:.3f} elsewhere "
                      f"{d['p_on_sum_off'] / max(d['n_off'], 1):.3f}; pc top1 {d['pc1'] / d['n_on']:.3f} top3 "
                      f"{d['pc3'] / d['n_on']:.3f} mean P(true pc) {d['p_true_pc'] / d['n_on']:.3f}; chord top1 "
                      f"{d['z1'] / max(d['nz'], 1):.3f}", flush=True)
    with open(out, 'x') as fh:
        json.dump({'L': a.L, 'seed': 'seed2' if a.seed2 else 'seed1', 'split': 'val', 'results': res}, fh, indent=1,
                  default=float)


if __name__ == '__main__':
    main()
