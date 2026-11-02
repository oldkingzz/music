"""Round 4 phase 1: print a diag_lat_v1 JSON compactly and check its metrics against the bench files with a float
tolerance (the diagnostic sums songs into one accumulator, the bench sums per-song subtotals: rounding only). Reads
files only; writes nothing. Val only, tuned on val, no test split.

  .venv/bin/python runs_mini/v5rt_1101/r4/B/show_diag_v1.py runs_mini/v5rt_1101/r4/B/diag_W_sung_v1.json
"""
import json
import os
import sys

LBD = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/live_bench'
KEYS = ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100', 'first_100ms_hit', 'on_time_right', 'notes', 'fresh')


def f(x, d=3):
    return 'n/a' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def main():
    with open(sys.argv[1]) as fh:
        res = json.load(fh)
    print(res['spec'], res['voice'])
    for ln, o in res['settings'].items():
        m = o['metrics']
        chk = o.get('bench_check')
        ok = None
        if chk:
            with open(os.path.join(LBD, chk['file'])) as fh:
                hb = json.load(fh)['headline']
            ok = all((abs(hb[k] - m[k]) <= 1e-12) if isinstance(m[k], float) else hb[k] == m[k] for k in KEYS)
        print(f"\n== {ln} (round-trip error {o['round_trip_error_ms']} ms): rt_nd_live {m['rt_nd_live']:.4f} harsh "
              f"{m['harsh']:.4f} fresh/100 {m['fresh_per100']:.2f} stumbles/100 {m['stumbles_per100']:.2f} first100 "
              f"{m['first_100ms_hit']:.3f} | bench file {chk and chk['file']} equal within 1e-12: {ok}")
        ha = o['hear_attacks']
        print(f"  hear attacks {ha['n']} ({ha['per100']:.1f}/100) {ha['by_action']} | sound after her onset p10/med/p90 "
              f"{f(ha['delay_ms']['p10'], 0)}/{f(ha['delay_ms']['median'], 0)}/{f(ha['delay_ms']['p90'], 0)} ms | "
              f"left of her note: {ha['left_of_her_note']} (median {f(ha['left_median'], 2)}) | lands {ha['lands_on']}")
        print(f"     over their sounding time: hit {f(ha['hit_share_new'])} vs replaced {f(ha['hit_share_replaced'])}, "
              f"harsh {f(ha['harsh_share_new'])} vs replaced {f(ha['harsh_share_replaced'])}; net rt_nd "
              f"{ha['net_rt_nd']:+.4f}, net harsh {ha['net_harsh']:+.4f}; fresh {ha['fresh']} ({ha['fresh_per100']:.2f}/100);"
              f" median sounding {f(ha['dur_ms_median'], 0)} ms")
        for k, s in ha['by_left'].items():
            if s['n']:
                print(f"     left {k:8s} n {s['n']:4d}: hit {f(s['hit_share_new'])} vs {f(s['hit_share_replaced'])}, harsh "
                      f"{f(s['harsh_share_new'])} vs {f(s['harsh_share_replaced'])}, net {s['net_rt_nd']:+.4f}, harsh net "
                      f"{s['net_harsh']:+.4f}, fresh {s['fresh']}")
        for k, s in ha['by_land'].items():
            print(f"     lands {k:9s} n {s['n']:4d}: hit {f(s['hit_share_new'])} vs {f(s['hit_share_replaced'])}, harsh "
                  f"{f(s['harsh_share_new'])} vs {f(s['harsh_share_replaced'])}, net {s['net_rt_nd']:+.4f}, harsh net "
                  f"{s['net_harsh']:+.4f}, fresh {s['fresh']}")
        pa = o['plan_attacks']
        print(f"  plan attacks {pa['n']} ({pa['per100']:.1f}/100): hit {f(pa['hit_share_new'])} vs replaced "
              f"{f(pa['hit_share_replaced'])}, harsh {f(pa['harsh_share_new'])} vs {f(pa['harsh_share_replaced'])}, net "
              f"{pa['net_rt_nd']:+.4f}, fresh {pa['fresh']}")
        hd = o['hear_decisions_delay_ms']
        print(f"  all hear decisions (first answer of each counted note) sound after her onset: p10/med/p90 "
              f"{f(hd['p10'], 0)}/{f(hd['median'], 0)}/{f(hd['p90'], 0)} ms (n {hd['n']}, unmatched {o['hear_unmatched']})")
        pp = o['plan_points']
        print(f"  plan points (counted {pp['counted']}, rest {pp['rest']}, uncounted {pp['uncounted']}): not yet heard "
              f"{f(pp['not_yet_heard_share'])}, held notes unheard {f(pp['held_unheard_share_of_held'])}")
        for k, s in pp['by_state'].items():
            print(f"     {k:13s} n {s['n']:4d} ({f(s['share'])}): plan note hit {f(s['plan_hit_share'])} harsh "
                  f"{f(s['plan_harsh_share'])} | actually sounded hit {f(s['actual_hit_share'])} harsh "
                  f"{f(s['actual_harsh_share'])} | window {f(s['window_beats'], 1)} beats")
        st = o['stumbles']
        print(f"  stumbles {st['total']} ({st['per100']:.2f}/100): {st['by_origin']}; hear attack stumbles "
              f"{st['hear_attack_stumbles']}, replaced by a hear {st['replaced_by_hear']}")
        print('  hit share by time since her onset: ' + ', '.join(
            f"{k} {f(v['hit_share'])} (part {v['rt_nd_part']:.4f}, harsh {f(v['harsh_share'])})"
            for k, v in o['bins'].items()) + f" | check {o['bins_check']}")
        if 'loss_vs_wired' in o:
            lw = o['loss_vs_wired']
            print('  loss vs wired: ' + ', '.join(f"{k} {v:+.4f}" for k, v in lw.items() if isinstance(v, float)))


if __name__ == '__main__':
    main()
