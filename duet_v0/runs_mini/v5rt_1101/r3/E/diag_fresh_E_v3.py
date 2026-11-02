"""diag_fresh_E_v3 (run v5rt_1101, round 3, idea E): diag_fresh_E_v2's attribution of live fresh clashes, stumbles and
harsh time to the system's decision branches, for a PlanE config given as JSON (merged over tune_E_v1.BASE) instead of
a system file, and with idea E's branches (trk_keep / trk_react / trk_attack / trk_sil) in the branch priority.
Val only (tuned on val; no test split). Offline: live_bench_rt is imported first (tripwire); nothing is played.

  .venv/bin/python runs_mini/v5rt_1101/r3/E/diag_fresh_E_v3.py '<json>' --voice sung|sung_low|phone_v11 --out <new json>
"""
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402  (tripwire first)
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/r3/E')
import diag_fresh_E_v2 as D2  # noqa: E402
import tune_E_v1 as T  # noqa: E402

D2.PRI = ('trk_sil', 'trk_keep', 'trk_react', 'trk_attack', 'hear_post_attack', 'hear_post_keep', 'hear_post_sil',
          'plan_attack', 'plan_silence', 'hear_enter', 'hear_switch', 'hear_same', 'hear_kept_plan', 'hear_kept_old')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg')
    ap.add_argument('--voice', required=True, choices=('sung', 'sung_low', 'clean', 'phone_v11'))
    ap.add_argument('--songs', type=int, default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(D2.OUT_DIR + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit('--out must be a new .json under ' + D2.OUT_DIR)
    train = T.B.bench3.train_sets()
    cfg = dict(T.BASE, **json.loads(a.cfg))
    obj = T.E.PlanE(T.r2_mix_v2.build(train), T.Prior(), cfg, T.load_copy_tab() if cfg['copy'] else None,
                    T.R3.load_hm_tab(), T.E.load_trk_tab() if cfg['trk'] else None)
    sysw = D2.Logged(obj)
    agg = defaultdict(Counter)
    details = []
    out = {'cfg': cfg, 'voice': a.voice, 'note': 'val only, systems tuned on val; no test split'}
    if a.voice == 'phone_v11':
        prox, rows, plan_rows, duet, her_rows, t0 = D2.phone(sysw, 'cfg', agg, details)
        out.update({'proxies': prox, 'hears': rows, 'plans_named': plan_rows, 'duet': duet, 'her_seg': her_rows,
                    't0': t0})
        print('proxies', json.dumps(prox), flush=True)
        for r in rows:
            print('  hear', r, flush=True)
        for r in plan_rows:
            print('  plan', r, flush=True)
        for r in duet:
            print('  duet', r, flush=True)
        for r in details:
            print('  fresh', r, flush=True)
    else:
        lat = LBR.LAT['wired']
        songs = LBR.bench3.eval_sets('val')['bach']
        if a.songs:
            songs = songs[:a.songs]
        tot = LBR.new_acc()
        for s in songs:
            eng, her, duet, z, acc, t0h, T_b = D2.run_song(sysw, 'cfg', s, a.voice, lat)
            for k in LBR.ACC_KEYS:
                tot[k] += acc[k]
            D2.analyse_song(sysw, eng, her, duet, z, s['tonic'], t0h, T_b, lat, agg, details, s['id'])
        out['metrics'] = LBR.metrics(tot)
        m = out['metrics']
        print(f"{a.voice}: rt_nd_live {m['rt_nd_live']:.4f} harsh {m['harsh']:.4f} fresh/100 "
              f"{m['fresh_per100']:.2f} stumbles/100 {m['stumbles_per100']:.2f} notes {m['notes']}", flush=True)
    out['agg'] = {k: dict(v) for k, v in agg.items()}
    out['fresh_details'] = details
    out['stats'] = dict(obj.stats)
    for k in ('att', 'fresh', 'wrong', 'stumble', 'att_rest', 'fresh_above', 'att_by_tracker', 'fresh_by_tracker',
              'wrong_by_tracker', 'hear_calls', 'time', 'harsh_t', 'hit_t'):
        if k in agg:
            v = agg[k]
            print(f"{k:16s} | " + ', '.join(f"{kk} {round(vv, 2)}" for kk, vv in sorted(v.items(), key=lambda t: -t[1])),
                  flush=True)
    print('stats', dict(obj.stats), flush=True)
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
