"""R2 tuning harness v3 (run v5rt_1101, round 2): tune_R2_v2.py for rt_systems.r2_lib_v3.PlanR2C (calibrated
harmonic memory: the train-only table models/R2_hm_v1.npz is passed in). VAL ONLY, tuned on val (no test split).
Same scoring path (bench_rt simulate / score_song, attack_metric song_counts, unchanged). Appends to
runs_mini/v5rt_1101/r2/tune_log_R2_v3.jsonl.

  python tune_R2_v3.py [--L 0.1] [--seed2] [--sub N] [--tag T] '<json>' ['<json>' ...]
"""
import argparse
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'A'))
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r2'))
from tune_A_v1 import B, r2_mix_v2, run  # noqa: E402
from tune_R2_v2 import BASE as BASE2  # noqa: E402
from rt_systems import r2_lib_v1 as R1  # noqa: E402
from rt_systems import r2_lib_v3 as R3  # noqa: E402

LOG = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r2', 'tune_log_R2_v3.jsonl')
BASE = dict(BASE2, hm_mode='loglin')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=float, default=0.1)
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('cfgs', nargs='+')
    a = ap.parse_args()
    heard = B.heard_seed2 if a.seed2 else B.HEARD1
    B.bench3.B2.heard = heard
    train = B.bench3.train_sets()
    mix = r2_mix_v2.build(train)
    prior = R1.Prior()
    tab = R1.load_copy_tab()
    hm = R3.load_hm_tab()
    sets = B.bench3.eval_sets('val')
    if a.sub:
        sets = {k: v[:a.sub] for k, v in sets.items()}
    for js in a.cfgs:
        cfg = dict(BASE, **json.loads(js))
        obj = R3.PlanR2C(mix, prior, cfg, tab if cfg['copy'] else None, hm)
        t0 = time.time()
        head, rows, ahead, att, n = run(obj, sets, a.L, heard)
        st = {k: round(v / n, 4) for k, v in sorted(obj.stats.items())}
        line = (f"rt_nd {head['rt_nd']:.4f} harsh {head['harsh']:.4f} fresh {ahead['fresh']:.2f} | nd1 {head['nd_first']:.3f} "
                f"hr1 {head['harsh_first']:.3f} ontime {head['on_time_right']:.3f} chg {head['changes']:.3f} "
                f"miss {ahead['miss']:.1f} att {ahead['attacks']:.1f} | "
                + ' '.join(f"{ds} {r['rt_nd']:.4f}/{r['harsh']:.4f}/{att[ds]['fresh']:.2f}" for ds, r in rows.items())
                + f" | us {head['us_mean']:.0f} | {time.time() - t0:.0f}s | {json.dumps(json.loads(js))}")
        print(line, flush=True)
        with open(LOG, 'a') as fh:
            fh.write(json.dumps({'time': time.strftime('%H:%M:%S'), 'tag': a.tag, 'L': a.L, 'seed2': a.seed2,
                                 'sub': a.sub, 'cfg': cfg, 'head': head,
                                 'rows': {d: {k: r[k] for k in ('rt_nd', 'harsh', 'nd_first', 'harsh_first')}
                                          for d, r in rows.items()},
                                 'attack': att, 'attack_head': ahead, 'stats': st}) + '\n')


if __name__ == '__main__':
    main()
