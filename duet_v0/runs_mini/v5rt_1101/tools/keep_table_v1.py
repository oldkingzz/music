"""Orchestrator's own keep-rule table for the round 1-2 systems, read straight from the score and attack files on disk.

Keep rule (state_R0 + state_R1), vs reactive:r2_mix_v2 at L = 0.1, seed 1, val (tuned on val, no test split):
headline rt_nd >= reactive + 0.005; harsh headline <= reactive; no dataset harsh > reactive + 0.005; no dataset
rt_nd < reactive - 0.01; seed2 delta >= +0.003; mean us < 2000; fresh clashes <= reactive's 1.8100634866373206
per 100 notes and no dataset above reactive + 1.0.

  .venv/bin/python runs_mini/v5rt_1101/tools/keep_table_v1.py --out <new json>
"""
import argparse
import json
import os

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
DS = ('bach', 'folk', 'pop', 'pop909')
REACT = ('scores/B_reactive_r2_mix_v2_L0.1.json', 'scores/B_reactive_r2_mix_v2_L0.1_seed2.json',
         'attack/reactive_r2_mix_v2_L0.1.json')
SYSTEMS = {
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


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def row(paths):
    s, s2, a = (load(p) for p in paths)
    assert s['L'] == 0.1 and s['seed'] == 'seed1' and s2['seed'] == 'seed2' and s['split'] == 'val', paths
    assert a['L'] == 0.1 and a['seed'] == 'seed1', paths
    return {'rt_nd': s['headline']['rt_nd'], 'harsh': s['headline']['harsh'], 'seed2': s2['headline']['rt_nd'],
            'us': s['headline']['us_mean'], 'on_time_right': s['headline']['on_time_right'],
            'fresh': a['headline']['fresh_clash_per100'], 'miss': a['headline']['attack_miss_per100'],
            'ds_rt_nd': {d: s['rows'][d]['rt_nd'] for d in DS}, 'ds_harsh': {d: s['rows'][d]['harsh'] for d in DS},
            'ds_fresh': {d: a['rows'][d]['fresh_clash_per100'] for d in DS}, 'files': list(paths)}


def checks(r, b):
    c = {'rt_nd +0.005': r['rt_nd'] >= b['rt_nd'] + 0.005, 'harsh <= reactive': r['harsh'] <= b['harsh'],
         'seed2 +0.003': r['seed2'] - b['seed2'] >= 0.003, 'us < 2000': r['us'] < 2000,
         'fresh <= reactive': r['fresh'] <= b['fresh'] + 1e-12}
    for d in DS:
        c[f'{d} harsh <= +0.005'] = r['ds_harsh'][d] <= b['ds_harsh'][d] + 0.005
        c[f'{d} rt_nd >= -0.01'] = r['ds_rt_nd'][d] >= b['ds_rt_nd'][d] - 0.01
        c[f'{d} fresh <= +1.0'] = r['ds_fresh'][d] <= b['ds_fresh'][d] + 1.0
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out_path = ap.parse_args().out
    b = row(REACT)
    out = {'reactive': b, 'systems': {}}
    print(f"{'system':18s} {'rt_nd':>7s} {'d':>7s} {'harsh':>7s} {'seed2 d':>8s} {'fresh':>6s} {'miss':>6s} "
          f"{'ontime':>7s} {'us':>5s}  pass  failed")
    print(f"{'reactive r2_mix_v2':18s} {b['rt_nd']:7.4f} {'':7s} {b['harsh']:7.4f} {'':8s} {b['fresh']:6.2f} "
          f"{b['miss']:6.1f} {b['on_time_right']:7.4f} {b['us']:5.0f}")
    for name, paths in SYSTEMS.items():
        r = row(paths)
        c = checks(r, b)
        bad = [k for k, v in c.items() if not v]
        r.update({'checks': c, 'n_pass': sum(c.values()), 'n_checks': len(c), 'passes': not bad})
        out['systems'][name] = r
        print(f"{name:18s} {r['rt_nd']:7.4f} {r['rt_nd'] - b['rt_nd']:+7.4f} {r['harsh']:7.4f} "
              f"{r['seed2'] - b['seed2']:+8.4f} {r['fresh']:6.2f} {r['miss']:6.1f} {r['on_time_right']:7.4f} "
              f"{r['us']:5.0f}  {r['n_pass']}/{r['n_checks']}  {', '.join(bad) or '-'}")
    for name in ('r2_copyhmbold_v1', 'r2_copyhm_v1', 'r2_copysafe_v1'):
        r = out['systems'][name]
        print(name, 'fresh by dataset', {d: round(r['ds_fresh'][d], 2) for d in DS},
              'reactive', {d: round(b['ds_fresh'][d], 2) for d in DS})
    with open(out_path, 'x') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
