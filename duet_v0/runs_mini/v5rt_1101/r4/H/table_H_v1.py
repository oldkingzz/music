"""Idea H score table (run v5rt_1101, round 4): reads the R4H_ files on disk and checks the round-4 rule (state_R3.json
round4_rule_fixed_now) for every idea-H variant: K1-K4 (K5 is the verifier's causal probe, not run here), the round-3
gates G1 (keep_table_v1.checks, imported unchanged) / G2 / G3 at wired, the wired mean against W's and the rank inside
idea H. Val only, tuned on val (this bundle has no test split). Baselines are the phase-1 L4_ files and the round-3 v1
files (never re-scored). Writes a new JSON (mode 'x').

  table_H_v1.py --out <new json under runs_mini/v5rt_1101/r4/H/>
"""
import argparse
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
sys.path.insert(0, os.path.join(RUN, 'tools'))
import keep_table_v1 as KT  # noqa: E402  (row / checks / REACT, unchanged)

VARS = ('r4_H_lagk15hd_v1', 'r4_H_lagk15_v1', 'r4_H_lagk15hq_v1', 'r4_H_lagk2hd_v1')
POOL_R3 = ('r3_X_regtrkst_v1', 'r3_X_regtrkdk_v1', 'r3_X_regtrkstsafe_v1')
CELLS = (('sung', 'micbt_cal'), ('sung_low', 'micbt_cal'), ('sung', 'bt_cal'), ('sung_low', 'bt_cal'))
KEYS = ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100')


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def live(rel, lat):
    d = load(rel)
    assert d['split'] == 'val' and d['lat'] == lat and d['songs_n'] == 17, rel
    return {k: d['headline'][k] for k in KEYS}


def phone(rel):
    d = load(rel)
    assert d['voice'] == 'phone_v11', rel
    return {'fresh_heard': d['phone']['fresh_heard'], 'harsh_heard': d['phone']['harsh_heard']}


def brt(rel, L, seed):
    d = load(rel)
    assert d['L'] == L and d['seed'] == seed and d['split'] == 'val', rel
    return d['headline']


def atk(rel, L):
    d = load(rel)
    assert d['L'] == L and d['seed'] == 'seed1', rel
    return d['headline']['fresh_clash_per100']


def k_cells(name, prefix):
    return {f'{v}@{l}': live(f'live_bench/{prefix}rt_{name}_{v}_{l}.json', l) for v, l in CELLS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out_path = os.path.abspath(ap.parse_args().out)
    if not out_path.startswith(os.path.join(RUN, 'r4', 'H') + os.sep) or os.path.exists(out_path):
        sys.exit('--out must be a new file under runs_mini/v5rt_1101/r4/H/')
    W = 'r3_X_regtrkst_v1'
    wc = k_cells(W, 'L4_')
    w_k1 = sum(c['rt_nd_live'] for c in wc.values()) / 4
    w_wired = {v: live(f'live_bench/R3X_rt_{W}_{v}_wired.json', 'wired') for v in ('sung', 'sung_low')}
    w_wired_mean = (w_wired['sung']['rt_nd_live'] + w_wired['sung_low']['rt_nd_live']) / 2
    re_wired = {v: live(f'live_bench/L_reactive_r2_mix_v2_{v}_wired.json', 'wired') for v in ('sung', 'sung_low')}
    re_v11 = phone('live_bench/L_reactive_r2_mix_v2_phone_v11.json')
    re5 = brt('scores/L4_reactive_r2_mix_v2_L0.5.json', 0.5, 'seed1')
    re5s2 = brt('scores/L4_reactive_r2_mix_v2_L0.5_seed2.json', 0.5, 'seed2')
    re5f = atk('attack/L4_reactive_r2_mix_v2_L0.5.json', 0.5)
    kt_b = KT.row(KT.REACT)
    ref = {}
    for n in POOL_R3:
        cc = k_cells(n, 'L4_')
        ref[n] = {'k1_mean': sum(c['rt_nd_live'] for c in cc.values()) / 4,
                  'k_fresh_stumbles': sum(c['fresh_per100'] + c['stumbles_per100'] for c in cc.values()),
                  'cells': cc}
    rows = {}
    for n in VARS:
        cc = k_cells(n, 'R4H_')
        k1 = sum(c['rt_nd_live'] for c in cc.values()) / 4
        k2 = {}
        for key, c in cc.items():
            w = wc[key]
            k2[f'{key} harsh <= W+0.003'] = c['harsh'] <= w['harsh'] + 0.003 + 1e-12
            k2[f'{key} fresh <= W+0.5'] = c['fresh_per100'] <= w['fresh_per100'] + 0.5 + 1e-9
            k2[f'{key} stumbles <= W+1.0'] = c['stumbles_per100'] <= w['stumbles_per100'] + 1.0 + 1e-9
        v11 = phone(f'live_bench/R4H_rt_{n}_phone_v11.json')
        k3 = {'v11 fresh_heard <= 4': v11['fresh_heard'] <= 4, 'v11 harsh_heard <= 0.1602': v11['harsh_heard'] <= 0.1602}
        s5 = brt(f'scores/R4H_rt_{n}_L0.5.json', 0.5, 'seed1')
        s5s2 = brt(f'scores/R4H_rt_{n}_L0.5_seed2.json', 0.5, 'seed2')
        f5 = atk(f'attack/R4H_rt_{n}_L0.5.json', 0.5)
        k4 = {'L0.5 rt_nd >= reactive+0.005': s5['rt_nd'] >= re5['rt_nd'] + 0.005,
              'L0.5 harsh <= reactive': s5['harsh'] <= re5['harsh'],
              'L0.5 fresh <= reactive': f5 <= re5f + 1e-12,
              'L0.5 seed2 delta >= +0.003': s5s2['rt_nd'] - re5s2['rt_nd'] >= 0.003}
        kt = KT.row((f'scores/R4H_rt_{n}_L0.1.json', f'scores/R4H_rt_{n}_L0.1_seed2.json',
                     f'attack/R4H_rt_{n}_L0.1.json'))
        g1c = KT.checks(kt, kt_b)
        ww = {v: live(f'live_bench/R4H_rt_{n}_{v}_wired.json', 'wired') for v in ('sung', 'sung_low')}
        g2 = {'sung rt_nd >= reactive+0.005': ww['sung']['rt_nd_live'] >= re_wired['sung']['rt_nd_live'] + 0.005,
              'sung_low rt_nd >= reactive': ww['sung_low']['rt_nd_live'] >= re_wired['sung_low']['rt_nd_live']}
        for v in ('sung', 'sung_low'):
            g2[f'{v} harsh <= reactive'] = ww[v]['harsh'] <= re_wired[v]['harsh']
            g2[f'{v} fresh <= reactive'] = ww[v]['fresh_per100'] <= re_wired[v]['fresh_per100']
        g3 = dict(k3)
        wired_mean = (ww['sung']['rt_nd_live'] + ww['sung_low']['rt_nd_live']) / 2
        failed = ([k for k, v in k2.items() if not v] + [k for k, v in k3.items() if not v]
                  + [k for k, v in k4.items() if not v] + ([] if k1 >= w_k1 + 0.005 else ['K1'])
                  + [f'G1 {k}' for k, v in g1c.items() if not v] + [f'G2 {k}' for k, v in g2.items() if not v]
                  + ([] if wired_mean >= w_wired_mean - 0.003 else ['wired mean >= W-0.003']))
        rows[n] = {'k1_mean': k1, 'k1_delta_vs_W': k1 - w_k1, 'k1_pass': k1 >= w_k1 + 0.005,
                   'k2_pass': all(k2.values()), 'k3_pass': all(k3.values()), 'k4_pass': all(k4.values()),
                   'g1_pass': all(g1c.values()), 'g1_n': f'{sum(g1c.values())}/{len(g1c)}', 'g2_pass': all(g2.values()),
                   'g3_pass': all(g3.values()), 'wired_mean': wired_mean, 'wired_mean_ok': wired_mean >= w_wired_mean - 0.003,
                   'k_fresh_stumbles': sum(c['fresh_per100'] + c['stumbles_per100'] for c in cc.values()),
                   'cells': cc, 'k2': k2, 'v11': v11, 'k4': {'rt_nd': s5['rt_nd'], 'seed2': s5s2['rt_nd'],
                                                            'seed2_delta': s5s2['rt_nd'] - re5s2['rt_nd'],
                                                            'harsh': s5['harsh'], 'fresh_per100': f5, 'checks': k4},
                   'g1': {'rt_nd': kt['rt_nd'], 'seed2_delta': kt['seed2'] - kt_b['seed2'], 'checks': g1c},
                   'wired': ww, 'g2': g2, 'failed': failed}
    passing = [n for n, r in rows.items() if r['k1_pass'] and r['k2_pass'] and r['k3_pass'] and r['k4_pass']]
    top = max((rows[n]['k1_mean'] for n in passing), default=None)
    band = [n for n in passing if top is not None and rows[n]['k1_mean'] >= top - 0.003]
    pick = min(band, key=lambda n: rows[n]['k_fresh_stumbles']) if band else None
    use = None
    if pick:
        r = rows[pick]
        use = ('replaces W as the single default' if r['g1_pass'] and r['g2_pass'] and r['g3_pass'] and r['wired_mean_ok']
               else 'page uses it only when the calibrated round trip >= 120 ms, W below')
    res = {'note': 'val only, tuned on val (no test split); K5 (causal probe) is the verifier\'s',
           'W': {'k1_mean': w_k1, 'cells': wc, 'wired_mean': w_wired_mean, 'wired': w_wired},
           'reactive': {'wired': re_wired, 'v11': re_v11, 'L0.5': {'rt_nd': re5['rt_nd'], 'seed2': re5s2['rt_nd'],
                                                                 'harsh': re5['harsh'], 'fresh_per100': re5f}},
           'round3_pool': ref, 'variants': rows, 'h_pass_K1_K4': passing, 'h_band': band, 'h_pick': pick, 'h_use': use}
    with open(out_path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print(f"W K1 {w_k1:.4f}  wired mean {w_wired_mean:.4f}")
    for n, r in ref.items():
        print(f"{n:24s} K1 {r['k1_mean']:.4f}  fresh+stum {r['k_fresh_stumbles']:.2f}")
    for n, r in rows.items():
        cs = '  '.join(f"{k}: {c['rt_nd_live']:.4f}/{c['harsh']:.4f}/{c['fresh_per100']:.2f}/{c['stumbles_per100']:.2f}"
                       for k, c in r['cells'].items())
        print(f"{n:20s} K1 {r['k1_mean']:.4f} ({r['k1_delta_vs_W']:+.4f}) K1 {r['k1_pass']} K2 {r['k2_pass']} "
              f"K3 {r['k3_pass']} K4 {r['k4_pass']} G1 {r['g1_n']} G2 {r['g2_pass']} G3 {r['g3_pass']} "
              f"wired {r['wired_mean']:.4f} fresh+stum {r['k_fresh_stumbles']:.2f} failed {r['failed']}\n    {cs}\n"
              f"    K4 L0.5 {r['k4']['rt_nd']:.4f} seed2 d {r['k4']['seed2_delta']:+.4f} harsh {r['k4']['harsh']:.4f} "
              f"fresh {r['k4']['fresh_per100']:.2f}; G1 L0.1 {r['g1']['rt_nd']:.4f} seed2 d {r['g1']['seed2_delta']:+.4f}")
    print('pass K1-K4:', passing, 'band:', band, 'pick:', pick, '->', use)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
