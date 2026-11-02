"""table_X_v1 (run v5rt_1101, round 3, combine step): the FULL round-3 candidate table and the binding selection rule
(state_R2.json 'round3_selection_rule_fixed_now'), read straight from the score / attack / live_bench JSON files on disk
(not from any agent's report). VAL ONLY: every system was tuned on val; this bundle has no test split. Read-only on its
inputs; writes one new JSON.

Pool = the round-1/2 systems scored live in phase 1 + every round-3 variant (ideas D, E, F and this combine step X).
G1 = runs_mini/v5rt_1101/tools/keep_table_v1.py's row / checks (logic copied, that file is not edited) vs
     reactive:r2_mix_v2 at L = 0.1: headline rt_nd >= reactive + 0.005; harsh <= reactive; no dataset harsh >
     reactive + 0.005; no dataset rt_nd < reactive - 0.01; seed2 delta >= +0.003; us < 2000; fresh clashes <=
     1.8100634866373206 and no dataset above reactive + 1.0.
G2 = runs_mini/v5rt_1101/tools/LB_select_v1.py's checks (copied): live bench, wired, val Bach headline: sung rt_nd_live >=
     reactive + 0.005; sung_low rt_nd_live >= reactive; on both voices harsh <= reactive and fresh_per100 <= reactive
     (reactive = the phase-1 files L_reactive_r2_mix_v2_*).
G3 = phone_v11 fresh_heard <= reactive's and harsh_heard <= reactive's + 0.005.
Rank (LB_select_v1.pick, copied): among G1 & G2 & G3 the highest mean of rt_nd_live on sung and sung_low; within 0.003 of
it the one with the fewest live fresh clashes (count, sum of both voices; ties: higher mean). If none passes G3: the best
of G1 & G2, flagged. If none passes G2: none (the page keeps the round-2 systems).
The comparisons are strict as in LB_select_v1 (no tolerance); the script also re-applies them with a 1e-12 tolerance and
reports whether any verdict would change.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/X/table_X_v1.py --out runs_mini/v5rt_1101/r3/X/table_v1.json
"""
import argparse
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
SYS = os.path.join(ABS, 'futures', 'v5_rt', 'rt_systems')
DS = ('bach', 'folk', 'pop', 'pop909')
FRESH_MAX = 1.8100634866373206
REACT = ('scores/B_reactive_r2_mix_v2_L0.1.json', 'scores/B_reactive_r2_mix_v2_L0.1_seed2.json',
         'attack/reactive_r2_mix_v2_L0.1.json')
PHASE1_G1 = {                                          # keep_table_v1.SYSTEMS (copied)
    'r2_copyhmbold_v1': ('scores/R2_rt_r2_copyhmbold_v1_L0.1.json', 'scores/R2_rt_r2_copyhmbold_v1_L0.1_seed2.json',
                         'attack/R2_rt_r2_copyhmbold_v1_L0.1.json'),
    'r2_copyhm_v1': ('scores/R2_rt_r2_copyhm_v1_L0.1.json', 'scores/R2_rt_r2_copyhm_v1_L0.1_seed2.json',
                     'attack/R2_rt_r2_copyhm_v1_L0.1.json'),
    'r2_copy_v1': ('scores/R2_rt_r2_copy_v1_L0.1.json', 'scores/R2_rt_r2_copy_v1_L0.1_seed2.json',
                   'attack/R2_rt_r2_copy_v1_L0.1.json'),
    'r2_copysafe_v1': ('scores/R2_rt_r2_copysafe_v1_L0.1.json', 'scores/R2_rt_r2_copysafe_v1_L0.1_seed2.json',
                       'attack/R2_rt_r2_copysafe_v1_L0.1.json'),
    'r1_A_prior_v1': ('scores/A_rt_r1_A_prior_v1_L0.1.json', 'scores/A_rt_r1_A_prior_v1_L0.1_seed2.json',
                      'attack/A_rt_r1_A_prior_v1_L0.1.json'),
    'r1_B_cons_v1': ('scores/r1_B_cons_v1_L0.1.json', 'scores/r1_B_cons_v1_L0.1_seed2.json',
                     'attack/rt_r1_B_cons_v1_L0.1.json'),
    'r1_C_revoice_v2': ('scores/C_rt_r1_C_revoice_v2_L0.1.json', 'scores/C_rt_r1_C_revoice_v2_L0.1_seed2.json',
                        'attack/C_rt_r1_C_revoice_v2_L0.1.json'),
}
ROUND3 = {
    'D': ('r3_D_reghm_v1', 'r3_D_regsafe_v1', 'r3_D_regguard_v1', 'r3_D_regguardsafe_v1'),
    'E': ('r3_E_trk_v1', 'r3_E_trkhold_v1', 'r3_E_trkholdk_v1', 'r3_E_trksafe_v1'),
    'F': ('r3_F_guard_v1', 'r3_F_calm_v1', 'r3_F_bold_v1', 'r3_F_lowsafe_v1'),
    'X': ('r3_X_regtrkdk_v1', 'r3_X_regtrkst_v1', 'r3_X_regtrkstsafe_v1', 'r3_X_regtrkstfree_v1'),
}
WHAT = {
    'r2_copyhmbold_v1': 'round-2 champion: melody prior + copy model + harmonic memory, lower attack cost (kappa 0.75)',
    'r2_copyhm_v1': 'round 2: melody prior + copy model + harmonic memory',
    'r2_copy_v1': 'round 2: melody prior + copy model of the song repeats + hedged hear',
    'r2_copysafe_v1': 'round 2: r2_copy_v1 with a 1.5x attack cost on unheard notes',
    'r1_A_prior_v1': 'round 1 idea A: melody prior, expected-value plan, posterior hear',
    'r1_B_cons_v1': 'round 1 idea B: conservative look-ahead',
    'r1_C_revoice_v2': 'round 1 idea C: learned forecaster, revoicing (close to reactive)',
    'r3_D_reghm_v1': 'D: register adapter around r2_copyhm_v1, no guard',
    'r3_D_regsafe_v1': 'D: register adapter around r2_copysafe_v1, no guard',
    'r3_D_regguard_v1': 'D: register adapter around r2_copy_v1 + out-of-key hearing guard (hold_sil)',
    'r3_D_regguardsafe_v1': 'D: register adapter around r2_copysafe_v1 + out-of-key hearing guard (hold_sil)',
    'r3_E_trk_v1': 'E: tracker-aware hear on r2_copyhm_v1 (tau_t 0.15, harsh_max 0.2)',
    'r3_E_trkhold_v1': 'E: r3_E_trk_v1 + tracker posterior for her held note (hold_trk)',
    'r3_E_trkholdk_v1': 'E: r3_E_trkhold_v1 with kappa_hold 0.25',
    'r3_E_trksafe_v1': 'E: stricter tracker guard (tau_t 0.12, harsh_max 0.1, sil_v -0.1)',
    'r3_F_guard_v1': 'F: r2_copyhm_v1 + stumble guards + cautious posterior hear, w_x 0.2',
    'r3_F_calm_v1': 'F: guard_v1 with a hop block (hop_bad 0.5), w_x 0.25',
    'r3_F_bold_v1': 'F: guard_v1 guards on r2_copyhmbold_v1, w_x 0.25',
    'r3_F_lowsafe_v1': 'F: calm_v1 with w_x 0.4',
    'r3_X_regtrkdk_v1': 'X: D register adapter + E tracker-aware hear (trkholdk) + key-aware doubt rule (0.7, keep); '
                        'no stumble guards',
    'r3_X_regtrkst_v1': 'X: D register adapter + E tracker-aware hear (trkholdk) + F stumble guards (g_mu 0.1) + '
                        'key-aware doubt rule (0.7, noatt)',
    'r3_X_regtrkstsafe_v1': 'X: r3_X_regtrkst_v1 with the doubt rule widened (out-of-key and P(misheard) >= 0.15, noatt)',
    'r3_X_regtrkstfree_v1': 'X: r3_X_regtrkst_v1 without the doubt rule (shows its cost; fails G3)',
}


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def exists(rel):
    return os.path.exists(os.path.join(RUN, rel))


def g1_row(paths):                                     # keep_table_v1.row (copied)
    s, s2, a = (load(p) for p in paths)
    assert s['L'] == 0.1 and s['seed'] == 'seed1' and s2['seed'] == 'seed2' and s['split'] == 'val', paths
    assert a['L'] == 0.1 and a['seed'] == 'seed1', paths
    return {'rt_nd': s['headline']['rt_nd'], 'harsh': s['headline']['harsh'], 'seed2': s2['headline']['rt_nd'],
            'us': s['headline']['us_mean'], 'on_time_right': s['headline']['on_time_right'],
            'fresh': a['headline']['fresh_clash_per100'], 'miss': a['headline']['attack_miss_per100'],
            'ds_rt_nd': {d: s['rows'][d]['rt_nd'] for d in DS}, 'ds_harsh': {d: s['rows'][d]['harsh'] for d in DS},
            'ds_fresh': {d: a['rows'][d]['fresh_clash_per100'] for d in DS}, 'files': list(paths),
            'bench_sha256': s['bench_sha256']}


def g1_checks(r, b, tol=0.0):                          # keep_table_v1.checks (copied; tol = 0 there, 1e-12 on fresh)
    c = {'rt_nd +0.005': r['rt_nd'] >= b['rt_nd'] + 0.005 - tol, 'harsh <= reactive': r['harsh'] <= b['harsh'] + tol,
         'seed2 +0.003': r['seed2'] - b['seed2'] >= 0.003 - tol, 'us < 2000': r['us'] < 2000,
         'fresh <= reactive': r['fresh'] <= b['fresh'] + max(tol, 1e-12)}
    for d in DS:
        c[f'{d} harsh <= +0.005'] = r['ds_harsh'][d] <= b['ds_harsh'][d] + 0.005 + tol
        c[f'{d} rt_nd >= -0.01'] = r['ds_rt_nd'][d] >= b['ds_rt_nd'][d] - 0.01 - tol
        c[f'{d} fresh <= +1.0'] = r['ds_fresh'][d] <= b['ds_fresh'][d] + 1.0 + tol
    return c


def live_files(prefix, name):
    base = f'live_bench/{prefix}rt_{name}_'
    return {'sung': base + 'sung_wired.json', 'sung_low': base + 'sung_low_wired.json', 'v11': base + 'phone_v11.json',
            'twinkle_phone': base + 'phone_twinkle.json', 'clean': base + 'clean_wired.json', 'sung_bt': base + 'sung_bt.json'}


def live_row(files):
    out = {}
    for v in ('sung', 'sung_low', 'clean', 'sung_bt'):
        if not exists(files[v]):
            continue
        d = load(files[v])
        assert d['split'] == 'val' and d['lat'] == ('bt' if v == 'sung_bt' else 'wired'), files[v]
        assert d['voice'] == ('sung' if v == 'sung_bt' else v), files[v]
        m = d['headline']
        out[v] = {k: m[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100',
                                     'stumbles_own_per100', 'on_time_right', 'wrong_att_per100', 'late_entry', 'notes')}
        out[v]['twinkle_rt_nd_live'] = d['twinkle']['rt_nd_live']
        out[v]['digest'] = d['digest']
        out[v]['code'] = d['code']
    d = load(files['v11'])
    assert d['voice'] == 'phone_v11', files['v11']
    out['v11'] = dict(d['phone']['proxies'], digest=d['digest'])
    if exists(files['twinkle_phone']):
        d = load(files['twinkle_phone'])
        m = d['phone']['metrics']
        out['twinkle_phone'] = {k: m[k] for k in ('rt_nd_live', 'harsh', 'fresh', 'stumbles_per100', 'on_time_right')}
        out['twinkle_phone']['sampled_fit'] = d['phone']['sampled']['fit']
    return out


def g2_checks(L, R, tol=0.0):                          # LB_select_v1's G2 (copied; tol = 0 there)
    s, sl, rs, rsl = L['sung'], L['sung_low'], R['sung'], R['sung_low']
    return {'sung rt_nd_live >= reactive + 0.005': s['rt_nd_live'] >= rs['rt_nd_live'] + 0.005 - tol,
            'sung_low rt_nd_live >= reactive': sl['rt_nd_live'] >= rsl['rt_nd_live'] - tol,
            'sung harsh <= reactive': s['harsh'] <= rs['harsh'] + tol,
            'sung_low harsh <= reactive': sl['harsh'] <= rsl['harsh'] + tol,
            'sung fresh_per100 <= reactive': s['fresh_per100'] <= rs['fresh_per100'] + tol,
            'sung_low fresh_per100 <= reactive': sl['fresh_per100'] <= rsl['fresh_per100'] + tol}


def g3_checks(L, R, tol=0.0):                          # LB_select_v1's G3 (copied; tol = 0 there)
    v, rv = L['v11'], R['v11']
    return {'v11 fresh_heard <= reactive': v['fresh_heard'] <= rv['fresh_heard'],
            'v11 harsh_heard <= reactive + 0.005': v['harsh_heard'] <= rv['harsh_heard'] + 0.005 + tol}


def pick(pool, names):                                 # LB_select_v1.pick (copied)
    if not names:
        return None
    names = sorted(names, key=lambda s: -pool[s]['mean_live'])
    top = pool[names[0]]['mean_live']
    near = [s for s in names if top - pool[s]['mean_live'] <= 0.003]
    return min(near, key=lambda s: (pool[s]['fresh_live_sum'], -pool[s]['mean_live']))


def verdict(pool, key_g1='G1', key_g2='G2', key_g3='G3'):
    full = [s for s in pool if pool[s][key_g1] and pool[s][key_g2] and pool[s][key_g3]]
    g12 = [s for s in pool if pool[s][key_g1] and pool[s][key_g2]]
    if full:
        v = {'selected': pick(pool, full), 'flag': None}
    elif g12:
        v = {'selected': pick(pool, g12), 'flag': 'none passes G3: best of G1 & G2'}
    else:
        v = {'selected': None, 'flag': 'none passes G2: the page keeps the round-2 systems'}
    v['passing_all'] = sorted(full, key=lambda s: -pool[s]['mean_live'])
    v['passing_G1_G2'] = sorted(g12, key=lambda s: -pool[s]['mean_live'])
    if full:
        top = max(pool[s]['mean_live'] for s in full)
        v['top_mean'] = top
        v['band_0.003'] = sorted([s for s in full if top - pool[s]['mean_live'] <= 0.003],
                                 key=lambda s: -pool[s]['mean_live'])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    path = os.path.abspath(ap.parse_args().out)
    if not path.startswith(os.path.join(RUN, 'r3', 'X') + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit('--out must be a new .json under runs_mini/v5rt_1101/r3/X/')
    b = g1_row(REACT)
    R = live_row({'sung': 'live_bench/L_reactive_r2_mix_v2_sung_wired.json',
                  'sung_low': 'live_bench/L_reactive_r2_mix_v2_sung_low_wired.json',
                  'v11': 'live_bench/L_reactive_r2_mix_v2_phone_v11.json',
                  'twinkle_phone': 'live_bench/L_reactive_r2_mix_v2_phone_twinkle.json',
                  'clean': 'live_bench/L_reactive_r2_mix_v2_clean_wired.json',
                  'sung_bt': 'live_bench/L_reactive_r2_mix_v2_sung_bt.json'})
    entries = [(n, 'phase1', PHASE1_G1[n], live_files('L_', n)) for n in PHASE1_G1]
    for letter, names in ROUND3.items():
        for n in names:
            p = f'R3{letter}_rt_{n}_L0.1'
            entries.append((n, 'round3_' + letter, (f'scores/{p}.json', f'scores/{p}_seed2.json', f'attack/{p}.json'),
                            live_files(f'R3{letter}_', n)))
    pool = {}
    for n, origin, g1p, lf in entries:
        g = g1_row(g1p)
        L = live_row(lf)
        c1, c2, c3 = g1_checks(g, b), g2_checks(L, R), g3_checks(L, R)
        t1, t2, t3 = g1_checks(g, b, 1e-12), g2_checks(L, R, 1e-12), g3_checks(L, R, 1e-12)
        rec = {'name': n, 'origin': origin, 'source': os.path.join(SYS, n + '.py'), 'what': WHAT[n],
               'G1': all(c1.values()), 'G2': all(c2.values()), 'G3': all(c3.values()),
               'G1_tol': all(t1.values()), 'G2_tol': all(t2.values()), 'G3_tol': all(t3.values()),
               'failed': [k for k, v in {**c1, **c2, **c3}.items() if not v],
               'g1': {k: g[k] for k in ('rt_nd', 'harsh', 'seed2', 'us', 'fresh', 'ds_rt_nd', 'ds_harsh', 'ds_fresh')},
               'seed2_delta': g['seed2'] - b['seed2'], 'g1_checks': c1, 'g2_checks': c2, 'g3_checks': c3,
               'live': L, 'mean_live': (L['sung']['rt_nd_live'] + L['sung_low']['rt_nd_live']) / 2,
               'fresh_live_sum': L['sung']['fresh'] + L['sung_low']['fresh'],
               'margins': {'sung_rt_nd_vs_bar': L['sung']['rt_nd_live'] - (R['sung']['rt_nd_live'] + 0.005),
                           'sung_low_rt_nd_vs_bar': L['sung_low']['rt_nd_live'] - R['sung_low']['rt_nd_live'],
                           'bench_fresh_vs_limit': FRESH_MAX - g['fresh'],
                           'v11_fresh_heard_vs_limit': R['v11']['fresh_heard'] - L['v11']['fresh_heard'],
                           'v11_harsh_heard_vs_limit': R['v11']['harsh_heard'] + 0.005 - L['v11']['harsh_heard']},
               'files': {'g1': [os.path.join('runs_mini/v5rt_1101', p) for p in g1p],
                         'live': {k: os.path.join('runs_mini/v5rt_1101', p) for k, p in lf.items() if exists(p)}}}
        pool[n] = rec
    v = verdict(pool)
    vt = verdict(pool, 'G1_tol', 'G2_tol', 'G3_tol')
    out = {'note': 'val only; every system tuned on val; this bundle has no test split. G3 is one real recording (v11, '
                   '44 answered notes, no ground truth).',
           'reactive': {'g1': b, 'live': R, 'mean_live': (R['sung']['rt_nd_live'] + R['sung_low']['rt_nd_live']) / 2,
                        'fresh_live_sum': R['sung']['fresh'] + R['sung_low']['fresh']},
           'rule': 'state_R2.json round3_selection_rule_fixed_now (G1 keep_table_v1 logic, G2 / G3 / rank LB_select_v1 '
                   'logic, both copied)',
           'verdict': v, 'verdict_with_1e-12_tolerance': vt, 'tolerance_changes_verdict': v != vt,
           'pool': pool}
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1)
    print(f"reactive: bench {b['rt_nd']:.4f}/{b['harsh']:.4f}/{b['fresh']:.2f} seed2 {b['seed2']:.4f} | sung "
          f"{R['sung']['rt_nd_live']:.4f}/{R['sung']['harsh']:.4f}/{R['sung']['fresh_per100']:.2f} st "
          f"{R['sung']['stumbles_per100']:.2f} | low {R['sung_low']['rt_nd_live']:.4f}/{R['sung_low']['harsh']:.4f}/"
          f"{R['sung_low']['fresh_per100']:.2f} st {R['sung_low']['stumbles_per100']:.2f} | v11 {R['v11']['fresh_heard']}/"
          f"{R['v11']['harsh_heard']:.4f}")
    print(f"{'name':22s} G1 G2 G3 {'bench':>6s} {'s2d':>7s} {'bfr':>5s} | {'sung':>6s} {'harsh':>6s} {'fr':>5s} {'st':>5s} | "
          f"{'low':>6s} {'harsh':>6s} {'fr':>5s} {'st':>5s} | v11 fh hh     | {'mean':>6s} fsum | failed")
    for n in sorted(pool, key=lambda s: -pool[s]['mean_live']):
        r = pool[n]
        g, L = r['g1'], r['live']
        print(f"{n:22s} {'Y' if r['G1'] else '-':2s} {'Y' if r['G2'] else '-':2s} {'Y' if r['G3'] else '-':2s} "
              f"{g['rt_nd']:.4f} {r['seed2_delta']:+.4f} {g['fresh']:5.2f} | {L['sung']['rt_nd_live']:.4f} "
              f"{L['sung']['harsh']:.4f} {L['sung']['fresh_per100']:5.2f} {L['sung']['stumbles_per100']:5.2f} | "
              f"{L['sung_low']['rt_nd_live']:.4f} {L['sung_low']['harsh']:.4f} {L['sung_low']['fresh_per100']:5.2f} "
              f"{L['sung_low']['stumbles_per100']:5.2f} | {L['v11']['fresh_heard']:2d} {L['v11']['harsh_heard']:.4f} | "
              f"{r['mean_live']:.4f} {r['fresh_live_sum']:4d} | {', '.join(r['failed']) or '-'}")
    print('verdict', json.dumps(v))
    print('tolerance changes verdict:', out['tolerance_changes_verdict'])
    print('wrote', path)


if __name__ == '__main__':
    main()
