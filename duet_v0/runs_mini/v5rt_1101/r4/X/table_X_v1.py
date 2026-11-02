"""Round-4 select step (run v5rt_1101): the FULL round-4 table. Reads only JSON files on disk (no system is run or
re-scored) and applies the round-4 rule fixed in runs_mini/v5rt_1101/state_R3.json (round4_rule_fixed_now) to the whole
pool: W = r3_X_regtrkst_v1, r3_X_regtrkdk_v1, r3_X_regtrkstsafe_v1 and every round-4 variant (idea G, idea H and the X
merges of this step). Val only, tuned on val (this bundle has no test split).

Files per system (prefix P):
  K1 / K2 cells (micbt_cal, bt_cal) x (sung, sung_low): live_bench/<P>rt_<name>_<voice>_<lat>.json, P = L4_ (phase 1)
      for the three round-3 systems, R4G_ / R4H_ / R4X_ for the round-4 variants.
  K3 / G3 (v11 at its native latency): live_bench/<P>rt_<name>_phone_v11.json, P = R3X_ for the round-3 systems.
  K4 (bench_rt L 0.5 seed1 / seed2, attack_metric L 0.5): W from phase 1 (L4_), r3_X_regtrkdk_v1 / r3_X_regtrkstsafe_v1
      from this step (R4X_), round-4 variants from their own runs.
  G1 (bench_rt L 0.1 seed1 / seed2, attack L 0.1; keep_table_v1.row / checks imported unchanged): R3X_ for round 3.
  G2 (live bench wired, sung / sung_low): R3X_ for the round-3 systems.
  reactive:r2_mix_v2: wired and v11 from round 3 (L_), the 4 cells and L 0.5 from phase 1 (L4_), L 0.1 = KT.REACT.
Rule (state_R3.json):
  K1  mean rt_nd_live over the 4 cells >= W's + 0.005 (W is checked against itself, so K1 fails by definition).
  K2  in every cell harsh <= W + 0.003, fresh_per100 <= W + 0.5, stumbles_per100 <= W + 1.0 (W against itself: passes).
  K3  v11 fresh_heard <= 4 and harsh_heard <= 0.1602.
  K4  bench_rt L 0.5 vs reactive L 0.5: rt_nd >= reactive + 0.005, harsh <= reactive, attack fresh clashes per 100 <=
      reactive, seed2 delta (seed2 rt_nd minus reactive's seed2 rt_nd) >= +0.003.
  K5  the verifier's causal probe: not run here (reported as not run).
  Rank among K1-K4: highest K1; within 0.003 of it, fewer fresh + stumbles summed over the 4 cells.
  Use: the pick also passes G1 (17/17 keep checks) & G2 & G3 at wired and its wired mean >= W's - 0.003 -> 'single
       default'; otherwise 'above 120 ms' (page: calibrated round trip >= 120 ms, W below); no pick -> 'none' (W).
Every live file is checked for spec, voice, lat, split, 17 songs and the frozen code hashes (v1 or v2 bench); every
bench_rt file for spec, L, seed, split and the frozen bench_rt sha when the file records it.

  table_X_v1.py --out <new json under runs_mini/v5rt_1101/r4/X/>
"""
import argparse
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
sys.path.insert(0, os.path.join(RUN, 'tools'))
import keep_table_v1 as KT  # noqa: E402  (row / checks / REACT, unchanged)

BENCH_SHA = 'f1eb524aa939b5ba54704532ac7b5eb536659ceaaef32c3701934f9cf7d3e772'
CODE_V1 = {'live_bench_rt.py': '46b43f91bf4bfa03f04c07ecdb543368394b6279f198cc8a21c435a8ab2bd8fd',
           'voice_v1.py': 'ed395a5acb7c4a1dbf341c22f41b3b772f74feaeb492880ff93ae8799220e2f9',
           'bench_rt.py': BENCH_SHA}
CODE_V2 = dict(CODE_V1, **{'live_bench_rt_v2.py': 'ffeb1ab0b6585719afcb6baf974cf1fc0a373852822eb12b32875572ed442946'})
RS = 'reactive:r2_mix_v2'
W = 'r3_X_regtrkst_v1'
R3 = ('r3_X_regtrkst_v1', 'r3_X_regtrkdk_v1', 'r3_X_regtrkstsafe_v1')
R4 = {'R4G_': ('r4_G_ehedgesafe_v1', 'r4_G_ehedge_v1', 'r4_G_hedge_v1'),
      'R4H_': ('r4_H_lagk15hd_v1', 'r4_H_lagk15_v1', 'r4_H_lagk15hq_v1', 'r4_H_lagk2hd_v1'),
      'R4X_': ('r4_X_hgsafe_v1', 'r4_X_hdgsafe_v1', 'r4_X_hgeh_v1')}
WHAT = {
    'r3_X_regtrkst_v1': 'W (round-3 page default): register adapter + tracker-aware hearing + stumble guards + doubt rule',
    'r3_X_regtrkdk_v1': 'round 3: W without the stumble guards (top round-3 wired mean)',
    'r3_X_regtrkstsafe_v1': 'round 3: most cautious W variant',
    'r4_G_ehedgesafe_v1': 'idea G: W + lag regime from call order + lag-aware hedge (floor 0.25, next-note harsh 0.3) '
                          '+ E hedge + late override; keep guard off in the large regime',
    'r4_G_ehedge_v1': 'idea G: as ehedgesafe with next-note harsh weight 0.2, keep guard unchanged',
    'r4_G_hedge_v1': 'idea G: lag-aware hedge (floor 0.15, harsh 0.2) + late override; no E hedge',
    'r4_H_lagk15hd_v1': 'idea H: lag-aware plan table, kappa_L 1.5, kappa_st_L 0, late_keep 2, hedge_L 0.5',
    'r4_H_lagk15_v1': 'idea H: as lagk15hd without the lag-aware hedge',
    'r4_H_lagk15hq_v1': 'idea H: as lagk15hd with hedge_L 0.25',
    'r4_H_lagk2hd_v1': 'idea H: kappa_L 2.0, hedge_L 0.5',
    'r4_X_hgsafe_v1': 'merge: H lagk15 plan + late_keep 2 (no H hedge) + G ehedgesafe hear (hedge, E hedge, late '
                      'override, keep guard off)',
    'r4_X_hdgsafe_v1': 'merge of the two idea picks: H lagk15hd (hedge_L 0.5) + G ehedgesafe hear',
    'r4_X_hgeh_v1': 'merge: H lagk15 plan + late_keep 2 (no H hedge) + G ehedge hear (next-note harsh 0.2, keep '
                    'guard on)'}
CELLS = (('sung', 'micbt_cal'), ('sung_low', 'micbt_cal'), ('sung', 'bt_cal'), ('sung_low', 'bt_cal'))
KEYS = ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100')
FILE_NOTES = []


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def live(rel, spec, voice, lat):
    d = load(rel)
    assert d['spec'] == spec and d['voice'] == voice and d['lat'] == lat and d['split'] == 'val', (rel, d['spec'])
    assert d['code'] in (CODE_V1, CODE_V2), (rel, d['code'])
    if lat not in ('wired', 'bt'):
        assert d['code'] == CODE_V2, rel                 # the round-4 settings exist only in the frozen v2 bench
    if voice == 'phone_v11':
        p = d['phone']['proxies']
        return {'fresh_heard': p['fresh_heard'], 'harsh_heard': p['harsh_heard'], 'digest': d['digest'], 'file': rel}
    assert d['songs_n'] == 17 and d['count_in'] == 4 and d['bpm'] == 90.0, rel
    h = d['headline']
    return dict({k: h[k] for k in KEYS}, digest=d['digest'], file=rel)


def brt(rel, spec, L, seed):
    d = load(rel)
    assert d['spec'] == spec and d['L'] == L and d['seed'] == seed and d['split'] == 'val', rel
    sha = d.get('bench_sha256')
    if sha is None:
        FILE_NOTES.append(f'{rel}: no bench_sha256 recorded')
    else:
        assert sha == BENCH_SHA, rel
    return d['headline']


def atk(rel, spec, L):
    d = load(rel)
    assert d['spec'] == spec and d['L'] == L and d['seed'] == 'seed1' and d['split'] == 'val', rel
    return d['headline']['fresh_clash_per100']


def prefixes(name):
    if name in R3:
        return {'cells': 'L4_', 'v11': 'R3X_', 'L05': 'L4_' if name == W else 'R4X_', 'L01': 'R3X_', 'wired': 'R3X_'}
    p = next(p for p, ns in R4.items() if name in ns)
    return {k: p for k in ('cells', 'v11', 'L05', 'L01', 'wired')}


def source(name):
    return os.path.join(ABS, 'futures', 'v5_rt', 'rt_systems', name + '.py')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out_path = os.path.abspath(ap.parse_args().out)
    if not out_path.startswith(os.path.join(RUN, 'r4', 'X') + os.sep) or os.path.exists(out_path):
        sys.exit('--out must be a new file under runs_mini/v5rt_1101/r4/X/')
    re_cells = {f'{v}@{l}': live(f'live_bench/L4_reactive_r2_mix_v2_{v}_{l}.json', RS, v, l) for v, l in CELLS}
    re_wired = {v: live(f'live_bench/L_reactive_r2_mix_v2_{v}_wired.json', RS, v, 'wired') for v in ('sung', 'sung_low')}
    re_v11 = live('live_bench/L_reactive_r2_mix_v2_phone_v11.json', RS, 'phone_v11', 'wired')
    re5 = brt('scores/L4_reactive_r2_mix_v2_L0.5.json', RS, 0.5, 'seed1')
    re5s2 = brt('scores/L4_reactive_r2_mix_v2_L0.5_seed2.json', RS, 0.5, 'seed2')
    re5f = atk('attack/L4_reactive_r2_mix_v2_L0.5.json', RS, 0.5)
    kt_b = KT.row(KT.REACT)
    pool = list(R3) + [n for ns in R4.values() for n in ns]
    rows = {}
    wc = w_wired = w_v11 = w_kt = None
    for n in pool:                                       # W first: every K1 / K2 bar is W's
        P = prefixes(n)
        spec = f'rt:{n}'
        cc = {f'{v}@{l}': live(f"live_bench/{P['cells']}rt_{n}_{v}_{l}.json", spec, v, l) for v, l in CELLS}
        v11 = live(f"live_bench/{P['v11']}rt_{n}_phone_v11.json", spec, 'phone_v11', 'wired')
        f05 = (f"scores/{P['L05']}rt_{n}_L0.5.json", f"scores/{P['L05']}rt_{n}_L0.5_seed2.json",
               f"attack/{P['L05']}rt_{n}_L0.5.json")
        s5, s5s2, f5 = brt(f05[0], spec, 0.5, 'seed1'), brt(f05[1], spec, 0.5, 'seed2'), atk(f05[2], spec, 0.5)
        f01 = (f"scores/{P['L01']}rt_{n}_L0.1.json", f"scores/{P['L01']}rt_{n}_L0.1_seed2.json",
               f"attack/{P['L01']}rt_{n}_L0.1.json")
        brt(f01[0], spec, 0.1, 'seed1')
        brt(f01[1], spec, 0.1, 'seed2')
        atk(f01[2], spec, 0.1)
        kt = KT.row(f01)
        ww = {v: live(f"live_bench/{P['wired']}rt_{n}_{v}_wired.json", spec, v, 'wired') for v in ('sung', 'sung_low')}
        if n == W:
            wc, w_wired, w_v11, w_kt = cc, ww, v11, kt
        k1 = sum(c['rt_nd_live'] for c in cc.values()) / 4
        w_k1 = sum(c['rt_nd_live'] for c in wc.values()) / 4
        k2, k2_slack = {}, {}
        for key, c in cc.items():
            w = wc[key]
            for m, cap, tol in (('harsh', 0.003, 1e-12), ('fresh_per100', 0.5, 1e-9), ('stumbles_per100', 1.0, 1e-9)):
                k2[f'{key} {m} <= W+{cap}'] = c[m] <= w[m] + cap + tol
                k2_slack[f'{key} {m}'] = w[m] + cap - c[m]
        k3 = {'v11 fresh_heard <= 4': v11['fresh_heard'] <= 4, 'v11 harsh_heard <= 0.1602': v11['harsh_heard'] <= 0.1602}
        k4 = {'L0.5 rt_nd >= reactive+0.005': s5['rt_nd'] >= re5['rt_nd'] + 0.005,
              'L0.5 harsh <= reactive': s5['harsh'] <= re5['harsh'],
              'L0.5 fresh <= reactive': f5 <= re5f + 1e-12,
              'L0.5 seed2 delta >= +0.003': s5s2['rt_nd'] - re5s2['rt_nd'] >= 0.003}
        g1c = KT.checks(kt, kt_b)
        g2 = {'sung rt_nd >= reactive+0.005': ww['sung']['rt_nd_live'] >= re_wired['sung']['rt_nd_live'] + 0.005,
              'sung_low rt_nd >= reactive': ww['sung_low']['rt_nd_live'] >= re_wired['sung_low']['rt_nd_live']}
        for v in ('sung', 'sung_low'):
            g2[f'{v} harsh <= reactive'] = ww[v]['harsh'] <= re_wired[v]['harsh']
            g2[f'{v} fresh <= reactive'] = ww[v]['fresh_per100'] <= re_wired[v]['fresh_per100']
        g3 = {'v11 fresh_heard <= reactive': v11['fresh_heard'] <= re_v11['fresh_heard'],
              'v11 harsh_heard <= reactive+0.005': v11['harsh_heard'] <= re_v11['harsh_heard'] + 0.005}
        wired_mean = (ww['sung']['rt_nd_live'] + ww['sung_low']['rt_nd_live']) / 2
        w_wired_mean = (w_wired['sung']['rt_nd_live'] + w_wired['sung_low']['rt_nd_live']) / 2
        k1_pass = k1 >= w_k1 + 0.005
        failed = (([] if k1_pass else ['K1' + (' (W against itself: fails by definition)' if n == W else '')])
                  + [f'K2 {k}' for k, v in k2.items() if not v] + [f'K3 {k}' for k, v in k3.items() if not v]
                  + [f'K4 {k}' for k, v in k4.items() if not v] + [f'G1 {k}' for k, v in g1c.items() if not v]
                  + [f'G2 {k}' for k, v in g2.items() if not v] + [f'G3 {k}' for k, v in g3.items() if not v]
                  + ([] if wired_mean >= w_wired_mean - 0.003 else ['wired mean >= W-0.003']))
        same = {'wired digests': all(ww[v]['digest'] == w_wired[v]['digest'] for v in ww),
                'micbt_cal digests': all(cc[f'{v}@micbt_cal']['digest'] == wc[f'{v}@micbt_cal']['digest']
                                         for v in ('sung', 'sung_low')),
                'bt_cal digests': all(cc[f'{v}@bt_cal']['digest'] == wc[f'{v}@bt_cal']['digest']
                                      for v in ('sung', 'sung_low')),
                'v11 digest': v11['digest'] == w_v11['digest'],
                'bench_rt L0.1 (rt_nd, harsh, seed2, fresh)': all(kt[k] == w_kt[k] for k in ('rt_nd', 'harsh', 'seed2',
                                                                                             'fresh'))}
        rows[n] = {'what': WHAT[n], 'source': source(n),
                   'k1_mean': k1, 'k1_delta_vs_W': k1 - w_k1, 'k1_pass': k1_pass,
                   'k2_pass': all(k2.values()), 'k3_pass': all(k3.values()), 'k4_pass': all(k4.values()),
                   'k5': 'not run here (the verifier\'s causal probe)',
                   'k_fresh_stumbles': sum(c['fresh_per100'] + c['stumbles_per100'] for c in cc.values()),
                   'cells': {k: {m: c[m] for m in KEYS} for k, c in cc.items()},
                   'k2': k2, 'k2_min_slack': {m: min(s for k, s in k2_slack.items() if k.endswith(m))
                                              for m in ('harsh', 'fresh_per100', 'stumbles_per100')},
                   'k3': k3, 'v11': {'fresh_heard': v11['fresh_heard'], 'harsh_heard': v11['harsh_heard']},
                   'k4': {'rt_nd': s5['rt_nd'], 'seed2': s5s2['rt_nd'], 'seed2_delta': s5s2['rt_nd'] - re5s2['rt_nd'],
                          'harsh': s5['harsh'], 'fresh_per100': f5, 'checks': k4},
                   'g1_pass': all(g1c.values()), 'g1_n': f'{sum(g1c.values())}/{len(g1c)}',
                   'g1': {'rt_nd': kt['rt_nd'], 'seed2_delta': kt['seed2'] - kt_b['seed2'], 'fresh': kt['fresh'],
                          'checks': g1c},
                   'g2_pass': all(g2.values()), 'g2': g2, 'g3_pass': all(g3.values()), 'g3': g3,
                   'wired': {v: {m: ww[v][m] for m in KEYS} for v in ww}, 'wired_mean': wired_mean,
                   'wired_mean_ok': wired_mean >= w_wired_mean - 0.003, 'same_as_W': same, 'failed': failed,
                   'files': sorted([c['file'] for c in cc.values()] + [v11['file']] + list(f05) + list(f01)
                                   + [ww[v]['file'] for v in ww])}
    passing = [n for n, r in rows.items() if r['k1_pass'] and r['k2_pass'] and r['k3_pass'] and r['k4_pass']]
    top = max((rows[n]['k1_mean'] for n in passing), default=None)
    band = sorted((n for n in passing if rows[n]['k1_mean'] >= top - 0.003 - 1e-12),
                  key=lambda n: (rows[n]['k_fresh_stumbles'], -rows[n]['k1_mean'])) if passing else []
    pick = band[0] if band else None

    def use_of(p):
        if p is None:
            return 'none'
        r = rows[p]
        return 'single default' if r['g1_pass'] and r['g2_pass'] and r['g3_pass'] and r['wired_mean_ok'] else 'above 120 ms'

    use = use_of(pick)
    alt = {}                                             # the same rule on smaller pools (sensitivity, report only)
    for tag, keep in (('without X merges', lambda n: not n.startswith('r4_X_')),
                      ('idea G only (+ round 3)', lambda n: not n.startswith(('r4_X_', 'r4_H_'))),
                      ('idea H only (+ round 3)', lambda n: not n.startswith(('r4_X_', 'r4_G_')))):
        ps = [n for n in passing if keep(n)]
        t = max((rows[n]['k1_mean'] for n in ps), default=None)
        b = sorted((n for n in ps if rows[n]['k1_mean'] >= t - 0.003 - 1e-12),
                   key=lambda n: (rows[n]['k_fresh_stumbles'], -rows[n]['k1_mean'])) if ps else []
        alt[tag] = {'pass_K1_K4': ps, 'band': b, 'pick': b[0] if b else None, 'use': use_of(b[0] if b else None)}
    schema_rows = []
    for n, r in rows.items():
        c = r['cells']
        schema_rows.append({
            'name': n, 'source': r['source'], 'what': r['what'],
            'k1_mean': round(r['k1_mean'], 4), 'k_fresh_stumbles': round(r['k_fresh_stumbles'], 2),
            'k1_pass': r['k1_pass'], 'k2_pass': r['k2_pass'], 'k3_pass': r['k3_pass'], 'k4_pass': r['k4_pass'],
            'wired_g1': r['g1_pass'], 'wired_g2': r['g2_pass'], 'wired_g3': r['g3_pass'],
            'wired_mean': round(r['wired_mean'], 4), 'failed': '; '.join(r['failed']) or 'none',
            'micbt_cal_sung': round(c['sung@micbt_cal']['rt_nd_live'], 4),
            'micbt_cal_low': round(c['sung_low@micbt_cal']['rt_nd_live'], 4),
            'bt_cal_sung': round(c['sung@bt_cal']['rt_nd_live'], 4),
            'bt_cal_low': round(c['sung_low@bt_cal']['rt_nd_live'], 4),
            'cells': '; '.join(f"{k} " + '/'.join(f'{x[m]:.4f}' if m in ('rt_nd_live', 'harsh') else f'{x[m]:.2f}'
                                                  for m in KEYS) for k, x in c.items()),
            'v11_fresh_heard': r['v11']['fresh_heard'], 'v11_harsh_heard': round(r['v11']['harsh_heard'], 4),
            'bench05_rt_nd': round(r['k4']['rt_nd'], 4), 'files': ', '.join(r['files'])})
    w_k1 = rows[W]['k1_mean']
    res = {'note': 'val only, tuned on val (no test split); every number read from the files on disk; K5 (causal probe) '
                   'is the verifier\'s and is not run here; W is checked against itself (K1 fails by definition)',
           'W': {'k1_mean': w_k1, 'cells': rows[W]['cells'], 'wired_mean': rows[W]['wired_mean']},
           'reactive': {'cells': {k: {m: c[m] for m in KEYS} for k, c in re_cells.items()},
                        'k1_mean': sum(c['rt_nd_live'] for c in re_cells.values()) / 4,
                        'wired': {v: {m: re_wired[v][m] for m in KEYS} for v in re_wired},
                        'v11': {'fresh_heard': re_v11['fresh_heard'], 'harsh_heard': re_v11['harsh_heard']},
                        'L0.5': {'rt_nd': re5['rt_nd'], 'seed2': re5s2['rt_nd'], 'harsh': re5['harsh'],
                                 'fresh_per100': re5f},
                        'L0.1': {k: kt_b[k] for k in ('rt_nd', 'harsh', 'seed2', 'fresh')}},
           'bars': {'K1': w_k1 + 0.005, 'K4 rt_nd': re5['rt_nd'] + 0.005, 'K4 harsh': re5['harsh'], 'K4 fresh': re5f,
                    'K4 seed2 rt_nd': re5s2['rt_nd'] + 0.003, 'wired mean': rows[W]['wired_mean'] - 0.003},
           'rows': rows, 'pass_K1_K4': passing, 'top_k1': top, 'band': band, 'pick': pick, 'use': use,
           'sensitivity': alt, 'schema_rows': schema_rows, 'file_notes': FILE_NOTES}
    with open(out_path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print(f"W K1 {w_k1:.4f} (bar {w_k1 + 0.005:.4f})  W wired mean {rows[W]['wired_mean']:.4f}  "
          f"reactive L0.5 {re5['rt_nd']:.4f} s2 {re5s2['rt_nd']:.4f} harsh {re5['harsh']:.4f} fresh {re5f:.2f}")
    for n, r in sorted(rows.items(), key=lambda t: -t[1]['k1_mean']):
        cs = '  '.join(f"{k}: {c['rt_nd_live']:.4f}/{c['harsh']:.4f}/{c['fresh_per100']:.2f}/{c['stumbles_per100']:.2f}"
                       for k, c in r['cells'].items())
        print(f"{n:22s} K1 {r['k1_mean']:.4f} ({r['k1_delta_vs_W']:+.4f}) K1 {r['k1_pass']} K2 {r['k2_pass']} "
              f"K3 {r['k3_pass']} K4 {r['k4_pass']} | G1 {r['g1_n']} G2 {r['g2_pass']} G3 {r['g3_pass']} wired "
              f"{r['wired_mean']:.4f} | f+s {r['k_fresh_stumbles']:.2f} | same_as_W {r['same_as_W']}\n    {cs}\n"
              f"    v11 {r['v11']}; K4 L0.5 {r['k4']['rt_nd']:.4f} seed2 d {r['k4']['seed2_delta']:+.4f} "
              f"harsh {r['k4']['harsh']:.4f} fresh {r['k4']['fresh_per100']:.2f}; G1 L0.1 {r['g1']['rt_nd']:.4f} "
              f"seed2 d {r['g1']['seed2_delta']:+.4f} fresh {r['g1']['fresh']:.2f}; K2 min slack "
              f"{ {m: round(s, 4) for m, s in r['k2_min_slack'].items()} }\n    failed: {r['failed'] or 'none'}")
    print('pass K1-K4:', passing)
    print('top K1:', top, 'band:', band, 'pick:', pick, '->', use)
    print('sensitivity:', json.dumps(alt))
    print('file notes:', FILE_NOTES)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
