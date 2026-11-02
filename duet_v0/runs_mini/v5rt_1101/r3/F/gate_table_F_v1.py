"""Round 3, idea F: the round-3 selection rule (state_R2.json, round3_selection_rule_fixed_now) applied to the idea-F
variants, read straight from the files on disk. Val only, tuned on val (no test split).
G1 = runs_mini/v5rt_1101/tools/keep_table_v1.py's checks (its logic copied, the file is not edited) vs
reactive:r2_mix_v2 at L = 0.1; G2 = live bench wired, val Bach headline vs RTDuet(reactive:r2_mix_v2) on the same audio
(the phase-1 files); G3 = phone_v11 proxies. Also: mean of rt_nd_live on sung and sung_low (the rank), live fresh
clashes (sum of both voices, the tie-break), live stumbles, bench stumbles (diag_attacks_v1). Phase-1 systems are listed
for reference with their G2 / G3 numbers.

  .venv/bin/python runs_mini/v5rt_1101/r3/F/gate_table_F_v1.py --out <new json under runs_mini/v5rt_1101/r3/F/>
"""
import argparse
import json
import os

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
DS = ('bach', 'folk', 'pop', 'pop909')
REACT = ('scores/B_reactive_r2_mix_v2_L0.1.json', 'scores/B_reactive_r2_mix_v2_L0.1_seed2.json',
         'attack/reactive_r2_mix_v2_L0.1.json')
LIVE_REACT = 'live_bench/L_reactive_r2_mix_v2_{}.json'
NEW = ('r3_F_guard_v1', 'r3_F_calm_v1', 'r3_F_bold_v1', 'r3_F_lowsafe_v1')
OLD = {'r2_copyhmbold_v1': 'live_bench/L_rt_r2_copyhmbold_v1_{}.json', 'r2_copyhm_v1': 'live_bench/L_rt_r2_copyhm_v1_{}.json',
       'r2_copy_v1': 'live_bench/L_rt_r2_copy_v1_{}.json', 'r2_copysafe_v1': 'live_bench/L_rt_r2_copysafe_v1_{}.json',
       'r1_A_prior_v1': 'live_bench/L_rt_r1_A_prior_v1_{}.json', 'r1_B_cons_v1': 'live_bench/L_rt_r1_B_cons_v1_{}.json',
       'r1_C_revoice_v2': 'live_bench/L_rt_r1_C_revoice_v2_{}.json'}


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def g1_row(paths):
    s, s2, a = (load(p) for p in paths)
    assert s['L'] == 0.1 and s['seed'] == 'seed1' and s2['seed'] == 'seed2' and s['split'] == 'val', paths
    assert a['L'] == 0.1 and a['seed'] == 'seed1', paths
    return {'rt_nd': s['headline']['rt_nd'], 'harsh': s['headline']['harsh'], 'seed2': s2['headline']['rt_nd'],
            'us': s['headline']['us_mean'], 'on_time_right': s['headline']['on_time_right'],
            'fresh': a['headline']['fresh_clash_per100'], 'miss': a['headline']['attack_miss_per100'],
            'ds_rt_nd': {d: s['rows'][d]['rt_nd'] for d in DS}, 'ds_harsh': {d: s['rows'][d]['harsh'] for d in DS},
            'ds_fresh': {d: a['rows'][d]['fresh_clash_per100'] for d in DS}, 'files': list(paths),
            'bench_sha256': s['bench_sha256']}


def g1_checks(r, b):
    c = {'rt_nd +0.005': r['rt_nd'] >= b['rt_nd'] + 0.005, 'harsh <= reactive': r['harsh'] <= b['harsh'],
         'seed2 +0.003': r['seed2'] - b['seed2'] >= 0.003, 'us < 2000': r['us'] < 2000,
         'fresh <= reactive': r['fresh'] <= b['fresh'] + 1e-12}
    for d in DS:
        c[f'{d} harsh <= +0.005'] = r['ds_harsh'][d] <= b['ds_harsh'][d] + 0.005
        c[f'{d} rt_nd >= -0.01'] = r['ds_rt_nd'][d] >= b['ds_rt_nd'][d] - 0.01
        c[f'{d} fresh <= +1.0'] = r['ds_fresh'][d] <= b['ds_fresh'][d] + 1.0
    return c


def live_row(pat):
    out = {}
    for v in ('sung_wired', 'sung_low_wired'):
        h = load(pat.format(v))
        assert h['split'] == 'val' and h['lat'] == 'wired', pat
        out[v.replace('_wired', '')] = {k: h['headline'][k] for k in ('rt_nd_live', 'harsh', 'fresh_per100',
                                                                       'stumbles_per100', 'stumbles_own_per100',
                                                                       'wrong_att_per100', 'on_time_right',
                                                                       'late_entry', 'fresh')}
        out[v.replace('_wired', '')]['code'] = h['code']
    p = load(pat.format('phone_v11'))['phone']['proxies']
    out['v11'] = {'fresh_heard': p['fresh_heard'], 'harsh_heard': p['harsh_heard'], 'attacks': p['attacks']}
    out['mean'] = (out['sung']['rt_nd_live'] + out['sung_low']['rt_nd_live']) / 2
    out['fresh_sum'] = out['sung']['fresh_per100'] + out['sung_low']['fresh_per100']
    return out


def g2_checks(r, b):
    c = {'sung rt_nd +0.005': r['sung']['rt_nd_live'] >= b['sung']['rt_nd_live'] + 0.005,
         'sung_low rt_nd >= reactive': r['sung_low']['rt_nd_live'] >= b['sung_low']['rt_nd_live']}
    for v in ('sung', 'sung_low'):
        c[f'{v} harsh <= reactive'] = r[v]['harsh'] <= b[v]['harsh'] + 1e-12
        c[f'{v} fresh <= reactive'] = r[v]['fresh_per100'] <= b[v]['fresh_per100'] + 1e-12
    return c


def g3_checks(r, b):
    return {'v11 fresh_heard <= reactive': r['v11']['fresh_heard'] <= b['v11']['fresh_heard'],
            'v11 harsh_heard <= reactive + 0.005': r['v11']['harsh_heard'] <= b['v11']['harsh_heard'] + 0.005 + 1e-12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    path = os.path.abspath(ap.parse_args().out)
    if not path.startswith(os.path.join(RUN, 'r3', 'F') + os.sep) or os.path.exists(path):
        raise SystemExit('--out must be a new file under runs_mini/v5rt_1101/r3/F/')
    b1 = g1_row(REACT)
    bl = live_row(LIVE_REACT)
    out = {'note': 'val only, tuned on val (no test split)', 'reactive': {'g1': b1, 'live': bl}, 'systems': {}}
    hdr = (f"{'system':18s} {'bench':>7s} {'seed2d':>7s} {'fresh':>5s} {'sung':>7s} {'s_low':>7s} {'mean':>7s} "
           f"{'fr_s':>5s} {'fr_l':>5s} {'h_s':>6s} {'h_l':>6s} {'st_s':>5s} {'st_l':>5s} {'v11f':>4s} {'v11h':>6s} G1 G2 G3")
    print(hdr)
    print(f"{'reactive':18s} {b1['rt_nd']:7.4f} {'':7s} {b1['fresh']:5.2f} {bl['sung']['rt_nd_live']:7.4f} "
          f"{bl['sung_low']['rt_nd_live']:7.4f} {bl['mean']:7.4f} {bl['sung']['fresh_per100']:5.2f} "
          f"{bl['sung_low']['fresh_per100']:5.2f} {bl['sung']['harsh']:6.4f} {bl['sung_low']['harsh']:6.4f} "
          f"{bl['sung']['stumbles_per100']:5.2f} {bl['sung_low']['stumbles_per100']:5.2f} {bl['v11']['fresh_heard']:4d} "
          f"{bl['v11']['harsh_heard']:6.4f}")
    for name in NEW:
        g1 = g1_row((f'scores/R3F_rt_{name}_L0.1.json', f'scores/R3F_rt_{name}_L0.1_seed2.json',
                     f'attack/R3F_rt_{name}_L0.1.json'))
        lv = live_row('live_bench/R3F_rt_' + name + '_{}.json')
        c1, c2, c3 = g1_checks(g1, b1), g2_checks(lv, bl), g3_checks(lv, bl)
        dg = load(f'diag/R3F_attacks_rt_{name}_L0.1.json')
        tw = load(f'live_bench/R3F_rt_{name}_phone_twinkle.json')['phone']
        r = {'g1': g1, 'live': lv, 'g1_checks': c1, 'g2_checks': c2, 'g3_checks': c3,
             'g1_pass': all(c1.values()), 'g2_pass': all(c2.values()), 'g3_pass': all(c3.values()),
             'failed': [k for c in (c1, c2, c3) for k, v in c.items() if not v],
             'bench_stumble': dg['headline']['stumble'], 'bench_stumble_bach': dg['rows']['bach']['stumble'],
             'phone_twinkle': {k: tw['metrics'][k] for k in ('rt_nd_live', 'harsh', 'fresh', 'stumbles_per100',
                                                             'on_time_right')}}
        r['phone_twinkle']['sampled_fit'] = tw['sampled']['fit']
        out['systems'][name] = r
        print(f"{name:18s} {g1['rt_nd']:7.4f} {g1['seed2'] - b1['seed2']:+7.4f} {g1['fresh']:5.2f} "
              f"{lv['sung']['rt_nd_live']:7.4f} {lv['sung_low']['rt_nd_live']:7.4f} {lv['mean']:7.4f} "
              f"{lv['sung']['fresh_per100']:5.2f} {lv['sung_low']['fresh_per100']:5.2f} {lv['sung']['harsh']:6.4f} "
              f"{lv['sung_low']['harsh']:6.4f} {lv['sung']['stumbles_per100']:5.2f} {lv['sung_low']['stumbles_per100']:5.2f} "
              f"{lv['v11']['fresh_heard']:4d} {lv['v11']['harsh_heard']:6.4f} {sum(c1.values())}/{len(c1)} "
              f"{sum(c2.values())}/{len(c2)} {sum(c3.values())}/{len(c3)} {', '.join(r['failed']) or '-'}")
    out['phase1'] = {}
    for name, pat in OLD.items():
        lv = live_row(pat)
        c2, c3 = g2_checks(lv, bl), g3_checks(lv, bl)
        out['phase1'][name] = {'live': lv, 'g2_pass': all(c2.values()), 'g3_pass': all(c3.values()),
                               'failed': [k for c in (c2, c3) for k, v in c.items() if not v]}
        print(f"  phase1 {name:18s} sung {lv['sung']['rt_nd_live']:.4f} low {lv['sung_low']['rt_nd_live']:.4f} "
              f"mean {lv['mean']:.4f} fresh {lv['fresh_sum']:.2f} st {lv['sung']['stumbles_per100']:.2f} "
              f"G2 {all(c2.values())} G3 {all(c3.values())} {', '.join(out['phase1'][name]['failed']) or '-'}")
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
