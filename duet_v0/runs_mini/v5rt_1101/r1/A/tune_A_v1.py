"""Idea A tuning harness (run v5rt_1101, round 1). VAL ONLY, tuned on val (this bundle has no test split).
Runs r1_A_core_v1.PlanA with several settings through bench_rt's own simulate / score_song (unchanged) and
attack_metric's song_counts (unchanged), building r2_mix_v2 and the prior once. Not a scored run: the scored variants
are futures/v5_rt/rt_systems/r1_A_*_v<k>.py run through bench_rt.py itself.

  python tune_A_v1.py [--L 0.1] [--seed2] [--sub N] '<json overrides>' ['<json overrides>' ...]
Prints one line per setting and appends it to runs_mini/v5rt_1101/r1/A/tune_log_A_v1.jsonl.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'tools'))
import bench_rt as B  # noqa: E402
import attack_metric as AM  # noqa: E402
from rt_systems import r1_A_core_v1 as core  # noqa: E402
from systems import r2_mix_v2  # noqa: E402

LOG = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'A', 'tune_log_A_v1.jsonl')
BASE = dict(lam=1.0, kappa=1.0, mu=0.05, beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0, w_x=0.3, gamma=0.0,
            min_cov=0.0, silence=False)


def run(obj, sets, L, heard):
    rows, att, notes_all = {}, {}, 0
    for ds, songs in sets.items():
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
            notes_all += len(notes)
        rows[ds] = B.row(acc, up, uh, len(songs))
        n = aacc['notes'] or 1
        att[ds] = {'fresh': 100 * aacc['fresh_clash'] / n, 'miss': 100 * aacc['attack_miss'] / n,
                   'attacks': 100 * aacc['attacks'] / n, 'harsh_fresh': aacc['harsh_fresh'] / aacc['d'],
                   'harsh_bleed': aacc['harsh_bleed'] / aacc['d']}
    head = {k: sum(r[k] for r in rows.values()) / len(rows) for k in ('rt_nd', 'harsh', 'nd_first', 'harsh_first',
                                                                       'on_time_right', 'changes', 'late_entry', 'us_mean')}
    ahead = {k: sum(r[k] for r in att.values()) / len(att) for k in next(iter(att.values()))}
    return head, rows, ahead, att, notes_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=float, default=0.1)
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('cfgs', nargs='+')
    a = ap.parse_args()
    heard = B.heard_seed2 if a.seed2 else B.HEARD1
    B.bench3.B2.heard = heard
    train = B.bench3.train_sets()
    mix = r2_mix_v2.build(train)
    prior = core.Prior()
    sets = B.bench3.eval_sets('val')
    if a.sub:
        sets = {k: v[:a.sub] for k, v in sets.items()}
    for js in a.cfgs:
        cfg = dict(BASE, **json.loads(js))
        obj = core.PlanA(mix, prior, cfg)
        t0 = time.time()
        head, rows, ahead, att, n = run(obj, sets, a.L, heard)
        st = {k: round(v / n, 4) for k, v in sorted(obj.stats.items())}
        line = (f"rt_nd {head['rt_nd']:.4f} harsh {head['harsh']:.4f} fresh {ahead['fresh']:.2f} | nd1 {head['nd_first']:.3f} "
                f"hr1 {head['harsh_first']:.3f} ontime {head['on_time_right']:.3f} chg {head['changes']:.3f} "
                f"late {head['late_entry']:.4f} miss {ahead['miss']:.1f} | "
                + ' '.join(f"{ds} {r['rt_nd']:.4f}/{r['harsh']:.4f}/{att[ds]['fresh']:.2f}" for ds, r in rows.items())
                + f" | us {head['us_mean']:.0f} | {st} | {time.time() - t0:.0f}s | {json.dumps(json.loads(js))}")
        print(line, flush=True)
        with open(LOG, 'a') as fh:
            fh.write(json.dumps({'time': time.strftime('%H:%M:%S'), 'L': a.L, 'seed2': a.seed2, 'sub': a.sub, 'cfg': cfg,
                                 'head': head, 'rows': {d: {k: r[k] for k in ('rt_nd', 'harsh', 'nd_first', 'harsh_first')}
                                                        for d, r in rows.items()},
                                 'attack': att, 'attack_head': ahead, 'stats': st}) + '\n')


if __name__ == '__main__':
    main()
