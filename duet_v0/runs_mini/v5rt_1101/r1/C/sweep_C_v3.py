"""v5rt_1101 r1 idea C: DEV sweep of r1_C_core_v3 settings on val (tuned on val; no test split exists). Not a scored
run. Reuses sweep_C_v2.run (bench_rt.simulate / score_song / row and attack_metric.song_counts unchanged, plus the
attack calibration check) with r1_C_core_v3's JointRT3, and also prints how often the wary ear held a note.

  python sweep_C_v3.py <configs.json> --out <new .json under r1/C/> [--L 0.1] [--seed2]
"""
import argparse
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep_C_v2 as S  # noqa: E402  (puts futures/v5_rt on sys.path, loads bench_rt and attack_metric)
from rt_systems import r1_C_core_v3 as C  # noqa: E402

B = S.B


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
        s0 = obj.n_sus
        head, rows, cal = S.run(obj, e['cfg'], val, a.L, a.seed2)
        nsus = obj.n_sus - s0
        res.append({'name': e['name'], 'cfg': e['cfg'], 'headline': head, 'rows': rows, 'calib': cal, 'n_sus': nsus})
        print(f"{e['name']:22s} rt_nd {head['rt_nd']:.4f} harsh {head['harsh']:.4f} nd1 {head['nd_first']:.3f} "
              f"ontime {head['on_time_right']:.3f} chg {head['changes']:.3f} fresh {head['fresh_clash_per100']:.3f} "
              f"watt {head['attack_miss_per100']:.1f} sus {nsus} | "
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
