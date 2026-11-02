"""R2 tuning harness v1 (run v5rt_1101, round 2). VAL ONLY, tuned on val (this bundle has no test split).
Runs rt_systems.r2_lib_v1.PlanR2 with several settings through bench_rt's own simulate / score_song and
attack_metric's song_counts (both unchanged, via runs_mini/v5rt_1101/r1/A/tune_A_v1.run), building r2_mix_v2, A's prior
and the copy table once. Not a scored run: scored variants are futures/v5_rt/rt_systems/r2_*_v<k>.py run through
bench_rt.py and attack_metric.py themselves. BASE = r1_A_prior_v1's settings with every round-2 switch off.

  python tune_R2_v1.py [--L 0.1] [--seed2] [--sub N] [--tag T] '<json overrides>' ['<json overrides>' ...]
Prints one line per setting and appends it to runs_mini/v5rt_1101/r2/tune_log_R2_v1.jsonl.
"""
import argparse
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'A'))
from tune_A_v1 import B, r2_mix_v2, run  # noqa: E402
from rt_systems import r2_lib_v1 as R2  # noqa: E402
from rt_systems.r1_A_prior_v1 import CFG as A_CFG  # noqa: E402

LOG = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r2', 'tune_log_R2_v1.jsonl')
BASE = dict(A_CFG, w_next=0.0, w_dyn=0.0, lam_n=1.0, hx=0.0, hedge_post=0, copy=False, copy_rhy=0.0, copy_p=1.0)


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
    prior = R2.Prior()
    tab = R2.load_copy_tab()
    sets = B.bench3.eval_sets('val')
    if a.sub:
        sets = {k: v[:a.sub] for k, v in sets.items()}
    for js in a.cfgs:
        cfg = dict(BASE, **json.loads(js))
        obj = R2.PlanR2(mix, prior, cfg, tab if cfg['copy'] else None)
        t0 = time.time()
        head, rows, ahead, att, n = run(obj, sets, a.L, heard)
        st = {k: round(v / n, 4) for k, v in sorted(obj.stats.items())}
        line = (f"rt_nd {head['rt_nd']:.4f} harsh {head['harsh']:.4f} fresh {ahead['fresh']:.2f} | nd1 {head['nd_first']:.3f} "
                f"hr1 {head['harsh_first']:.3f} ontime {head['on_time_right']:.3f} chg {head['changes']:.3f} "
                f"miss {ahead['miss']:.1f} att {ahead['attacks']:.1f} | "
                + ' '.join(f"{ds} {r['rt_nd']:.4f}/{r['harsh']:.4f}/{att[ds]['fresh']:.2f}" for ds, r in rows.items())
                + f" | us {head['us_mean']:.0f} | {st} | {time.time() - t0:.0f}s | {json.dumps(json.loads(js))}")
        print(line, flush=True)
        with open(LOG, 'a') as fh:
            fh.write(json.dumps({'time': time.strftime('%H:%M:%S'), 'tag': a.tag, 'L': a.L, 'seed2': a.seed2,
                                 'sub': a.sub, 'cfg': cfg, 'head': head,
                                 'rows': {d: {k: r[k] for k in ('rt_nd', 'harsh', 'nd_first', 'harsh_first')}
                                          for d, r in rows.items()},
                                 'attack': att, 'attack_head': ahead, 'stats': st}) + '\n')


if __name__ == '__main__':
    main()
