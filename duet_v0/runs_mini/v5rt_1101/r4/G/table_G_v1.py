"""table_G_v1 (run v5rt_1101, round 4, idea G): the round-4 rule's checks for the scored G variants, read straight from the
files on disk (live_bench/R4G_*, scores/R4G_*, attack/R4G_*), against W's and reactive's phase-1 / round-3 files. Val
only (17 val chorales; bench_rt all four val datasets), tuned on val; this bundle has no test split. Read only, except
that it writes one NEW json (refuses to overwrite).

K1 mean rt_nd_live over {micbt_cal, bt_cal} x {sung, sung_low} >= W's + 0.005
K2 per cell: harsh <= W + 0.003, fresh_per100 <= W + 0.5, stumbles_per100 <= W + 1.0
K3 v11 fresh_heard <= 4, harsh_heard <= 0.1602
K4 bench_rt L 0.5 vs reactive L 0.5: rt_nd >= +0.005, harsh <=, fresh (attack_metric) <=, seed2 delta >= +0.003
G1 keep rule at L 0.1 (tools/keep_table_v1.row / checks, imported unchanged); G2 wired vs reactive; G3 = K3
use: G1 & G2 & G3 and wired mean (sung, sung_low) >= W's - 0.003

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r4/G/table_G_v1.py --out runs_mini/v5rt_1101/r4/G/table_G_v1.json
"""
import argparse
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
sys.path.insert(0, os.path.join(RUN, 'tools'))
import keep_table_v1 as KT  # noqa: E402  (row / checks / REACT, unchanged)

NAMES = ('r4_G_hedge_v1', 'r4_G_ehedge_v1', 'r4_G_ehedgesafe_v1')
CELLS = (('sung', 'micbt_cal'), ('sung_low', 'micbt_cal'), ('sung', 'bt_cal'), ('sung_low', 'bt_cal'))
COLS = ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100')
DCK = ('hear_switch', 'hear_keep', 'hear_same', 'hear_enter', 'hear_silence', 'plan_switch', 'plan_keep', 'plan_enter',
       'plan_same', 'plan_silence', 'stop_she_stopped')


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def live(prefix, spec_name, voice, lat):
    f = load(f'live_bench/{prefix}_{spec_name}_{voice}_{lat}.json')
    assert f['split'] == 'val' and f['voice'] == voice and f['lat'] == lat and f['songs_n'] == 17, (spec_name, voice, lat)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        sys.exit(f'refusing to overwrite {out}')
    W = {c: live('L4', 'rt_r3_X_regtrkst_v1', *c) for c in CELLS}
    Ww = {v: live('R3X', 'rt_r3_X_regtrkst_v1', v, 'wired')['headline'] for v in ('sung', 'sung_low')}
    Rw = {v: live('L', 'reactive_r2_mix_v2', v, 'wired')['headline'] for v in ('sung', 'sung_low')}
    w_k1 = sum(W[c]['headline']['rt_nd_live'] for c in CELLS) / 4
    w_fs = sum(W[c]['headline']['fresh_per100'] + W[c]['headline']['stumbles_per100'] for c in CELLS)
    w_wired = (Ww['sung']['rt_nd_live'] + Ww['sung_low']['rt_nd_live']) / 2
    r5, r5s2, r5a = (load('scores/L4_reactive_r2_mix_v2_L0.5.json'), load('scores/L4_reactive_r2_mix_v2_L0.5_seed2.json'),
                     load('attack/L4_reactive_r2_mix_v2_L0.5.json'))
    rk = KT.row(KT.REACT)
    res = {'note': 'val only, tuned on val, no test split', 'W': {'k1_mean': w_k1, 'k_fresh_stumbles': w_fs,
                                                                  'wired_mean': w_wired,
                                                                  'cells': {'@'.join(c): {k: W[c]['headline'][k] for k in COLS}
                                                                            for c in CELLS},
                                                                  'decision_counts_bt_cal': {
                                                                      '@'.join(c): {k: W[c]['decision_counts'].get(k, 0)
                                                                                    for k in DCK} for c in CELLS}},
           'reactive_L0.5': {'rt_nd': r5['headline']['rt_nd'], 'harsh': r5['headline']['harsh'],
                             'seed2': r5s2['headline']['rt_nd'], 'fresh': r5a['headline']['fresh_clash_per100']},
           'rows': {}}
    for n in NAMES:
        spec = 'rt_' + n
        G = {c: live('R4G', spec, *c) for c in CELLS}
        fails = []
        cells, txt = {}, []
        for c in CELLS:
            h, w = G[c]['headline'], W[c]['headline']
            cells['@'.join(c)] = {k: h[k] for k in COLS}
            cells['@'.join(c)]['decision_counts'] = {k: G[c]['decision_counts'].get(k, 0) for k in DCK}
            txt.append(f"{c[0]}@{c[1]} {h['rt_nd_live']:.4f}/{h['harsh']:.4f}/{h['fresh_per100']:.2f}/"
                       f"{h['stumbles_per100']:.2f}")
            if h['harsh'] > w['harsh'] + 0.003 + 1e-12:
                fails.append(f"K2 {c[0]}@{c[1]} harsh")
            if h['fresh_per100'] > w['fresh_per100'] + 0.5 + 1e-12:
                fails.append(f"K2 {c[0]}@{c[1]} fresh")
            if h['stumbles_per100'] > w['stumbles_per100'] + 1.0 + 1e-12:
                fails.append(f"K2 {c[0]}@{c[1]} stumbles")
        k1 = sum(G[c]['headline']['rt_nd_live'] for c in CELLS) / 4
        fs = sum(G[c]['headline']['fresh_per100'] + G[c]['headline']['stumbles_per100'] for c in CELLS)
        k1_pass = k1 >= w_k1 + 0.005 - 1e-12
        if not k1_pass:
            fails.append('K1')
        k2_pass = not any(f.startswith('K2') for f in fails)
        v11 = load(f'live_bench/R4G_{spec}_phone_v11.json')['phone']['proxies']
        k3_pass = v11['fresh_heard'] <= 4 and v11['harsh_heard'] <= 0.1602 + 1e-12
        if not k3_pass:
            fails.append('K3')
        s5, s5s2, a5 = (load(f'scores/R4G_{spec}_L0.5.json'), load(f'scores/R4G_{spec}_L0.5_seed2.json'),
                        load(f'attack/R4G_{spec}_L0.5.json'))
        k4 = {'rt_nd': s5['headline']['rt_nd'], 'harsh': s5['headline']['harsh'], 'seed2': s5s2['headline']['rt_nd'],
              'fresh': a5['headline']['fresh_clash_per100']}
        k4c = {'rt_nd +0.005': k4['rt_nd'] >= res['reactive_L0.5']['rt_nd'] + 0.005,
               'harsh <=': k4['harsh'] <= res['reactive_L0.5']['harsh'],
               'fresh <=': k4['fresh'] <= res['reactive_L0.5']['fresh'] + 1e-12,
               'seed2 +0.003': k4['seed2'] - res['reactive_L0.5']['seed2'] >= 0.003}
        k4_pass = all(k4c.values())
        fails += ['K4 ' + k for k, v in k4c.items() if not v]
        g1r = KT.row((f'scores/R4G_{spec}_L0.1.json', f'scores/R4G_{spec}_L0.1_seed2.json', f'attack/R4G_{spec}_L0.1.json'))
        g1c = KT.checks(g1r, rk)
        g1 = all(g1c.values())
        if not g1:
            fails += ['G1 ' + k for k, v in g1c.items() if not v]
        gw = {v: live('R4G', spec, v, 'wired')['headline'] for v in ('sung', 'sung_low')}
        g2c = {'sung rt_nd +0.005': gw['sung']['rt_nd_live'] >= Rw['sung']['rt_nd_live'] + 0.005,
               'sung_low rt_nd': gw['sung_low']['rt_nd_live'] >= Rw['sung_low']['rt_nd_live']}
        for v in ('sung', 'sung_low'):
            g2c[f'{v} harsh'] = gw[v]['harsh'] <= Rw[v]['harsh']
            g2c[f'{v} fresh'] = gw[v]['fresh_per100'] <= Rw[v]['fresh_per100']
        g2 = all(g2c.values())
        if not g2:
            fails += ['G2 ' + k for k, v in g2c.items() if not v]
        wired_mean = (gw['sung']['rt_nd_live'] + gw['sung_low']['rt_nd_live']) / 2
        use_single = g1 and g2 and k3_pass and wired_mean >= w_wired - 0.003 - 1e-12
        res['rows'][n] = {'k1_mean': k1, 'k1_delta_vs_W': k1 - w_k1, 'k_fresh_stumbles': fs, 'cells': cells,
                          'cells_txt': '; '.join(txt), 'k1_pass': k1_pass, 'k2_pass': k2_pass, 'k3_pass': k3_pass,
                          'v11': {'fresh_heard': v11['fresh_heard'], 'harsh_heard': v11['harsh_heard']},
                          'k4_pass': k4_pass, 'k4': k4, 'k4_checks': k4c, 'wired_g1': g1, 'g1_checks': g1c,
                          'g1_row': {k: g1r[k] for k in ('rt_nd', 'harsh', 'seed2', 'fresh', 'us')},
                          'wired_g2': g2, 'g2_checks': g2c, 'wired_g3': k3_pass,
                          'wired': {v: {k: gw[v][k] for k in COLS} for v in ('sung', 'sung_low')},
                          'wired_mean': wired_mean, 'use_single_default': use_single, 'failed': fails}
        print(f"{n}: K1 {k1:.4f} ({k1 - w_k1:+.4f} vs W {w_k1:.4f}) {'PASS' if k1_pass else 'fail'} | K2 "
              f"{'PASS' if k2_pass else 'fail'} | K3 {v11['fresh_heard']}/{v11['harsh_heard']:.4f} "
              f"{'PASS' if k3_pass else 'fail'} | K4 {k4['rt_nd']:.4f}/{k4['harsh']:.4f}/{k4['fresh']:.2f} s2d "
              f"{k4['seed2'] - res['reactive_L0.5']['seed2']:+.4f} {'PASS' if k4_pass else 'fail'} | G1 {g1} G2 {g2} "
              f"G3 {k3_pass} wired mean {wired_mean:.4f} (W {w_wired:.4f}) use_single {use_single} | f+s {fs:.2f} "
              f"(W {w_fs:.2f}) | {'; '.join(txt)} | failed {fails}", flush=True)
    with open(out, 'x') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
