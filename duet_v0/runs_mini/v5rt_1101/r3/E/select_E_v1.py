"""select_E_v1 (run v5rt_1101, round 3, idea E): the round-3 selection rule (state_R2.json, binding) applied to the
idea-E variants, read straight from the files on disk. VAL ONLY, tuned on val (this bundle has no test split).

G1 = runs_mini/v5rt_1101/tools/keep_table_v1.py's checks (its logic copied, not imported: that file runs nothing on
     import but its SYSTEMS table is fixed) vs reactive:r2_mix_v2 at L = 0.1, seed 1 (+ seed2), incl. fresh clashes.
G2 = live bench, wired, val Bach headline: sung rt_nd_live >= reactive + 0.005; sung_low rt_nd_live >= reactive; on both
     voices harsh <= reactive and fresh_per100 <= reactive (reactive = phase-1 files L_reactive_r2_mix_v2_*).
G3 = phone_v11: fresh_heard <= reactive's and harsh_heard <= reactive's + 0.005.
Rank among G1 & G2 & G3 (this script's pool: the idea-E variants plus the phase-1 live pool): highest mean of
rt_nd_live on sung and sung_low; within 0.003 the one with fewer live fresh clashes (sum of both voices).

  .venv/bin/python runs_mini/v5rt_1101/r3/E/select_E_v1.py --out runs_mini/v5rt_1101/r3/E/select_E_v1.json
"""
import argparse
import json
import os

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
DS = ('bach', 'folk', 'pop', 'pop909')
REACT = ('scores/B_reactive_r2_mix_v2_L0.1.json', 'scores/B_reactive_r2_mix_v2_L0.1_seed2.json',
         'attack/reactive_r2_mix_v2_L0.1.json')
E_NAMES = ('r3_E_trk_v1', 'r3_E_trkhold_v1', 'r3_E_trkholdk_v1', 'r3_E_trksafe_v1')
PHASE1 = ('r2_copyhmbold_v1', 'r2_copyhm_v1', 'r2_copy_v1', 'r2_copysafe_v1', 'r1_A_prior_v1', 'r1_B_cons_v1',
          'r1_C_revoice_v2')


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


def live(spec_file):
    d = load('live_bench/' + spec_file)
    return d


def live_row(prefix):
    out = {}
    for v in ('sung', 'sung_low', 'clean'):
        p = os.path.join(RUN, 'live_bench', f'{prefix}_{v}_wired.json')
        if os.path.exists(p):
            d = json.load(open(p))
            m = d['headline']
            out[v] = {k: m[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100', 'on_time_right',
                                         'wrong_att_per100', 'fresh', 'late_entry', 'first_100ms_hit',
                                         'first_100ms_harsh', 'blips')}
            out[v]['twinkle_rt_nd_live'] = d['twinkle']['rt_nd_live']
            out[v]['twinkle_fresh_per100'] = d['twinkle']['fresh_per100']
            out[v]['digest'] = d['digest']
            out[v]['code'] = d['code']
    for v in ('phone_v11', 'phone_twinkle'):
        p = os.path.join(RUN, 'live_bench', f'{prefix}_{v}.json')
        if os.path.exists(p):
            d = json.load(open(p))
            ph = d['phone']
            out[v] = {'proxies': ph['proxies'], 'digest': d['digest']}
            if 'metrics' in ph:
                out[v]['metrics'] = {k: ph['metrics'][k] for k in ('rt_nd_live', 'harsh', 'fresh_per100',
                                                                    'stumbles_per100', 'on_time_right',
                                                                    'wrong_att_per100')}
                out[v]['sampled_fit'] = ph['sampled']['fit']
    return out


def g23(L, R):
    s, sl, v = L['sung'], L['sung_low'], L['phone_v11']['proxies']
    rs, rsl, rv = R['sung'], R['sung_low'], R['phone_v11']['proxies']
    g2 = {'sung rt_nd_live >= reactive + 0.005': s['rt_nd_live'] >= rs['rt_nd_live'] + 0.005,
          'sung_low rt_nd_live >= reactive': sl['rt_nd_live'] >= rsl['rt_nd_live'],
          'sung harsh <= reactive': s['harsh'] <= rs['harsh'],
          'sung_low harsh <= reactive': sl['harsh'] <= rsl['harsh'],
          'sung fresh <= reactive': s['fresh_per100'] <= rs['fresh_per100'] + 1e-12,
          'sung_low fresh <= reactive': sl['fresh_per100'] <= rsl['fresh_per100'] + 1e-12}
    g3 = {'v11 fresh_heard <= reactive': v['fresh_heard'] <= rv['fresh_heard'],
          'v11 harsh_heard <= reactive + 0.005': v['harsh_heard'] <= rv['harsh_heard'] + 0.005}
    return g2, g3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if os.path.exists(path):
        raise SystemExit('refusing to overwrite ' + path)
    b = g1_row(REACT)
    R = live_row('L_reactive_r2_mix_v2')
    out = {'note': 'val only, tuned on val; no test split', 'reactive': {'g1': b, 'live': R}, 'systems': {}}
    keep_orch = load('verify/keep_table_orch_v1.json')
    rows = []
    for name in E_NAMES + PHASE1:
        if name in E_NAMES:
            paths = (f'scores/R3E_rt_{name}_L0.1.json', f'scores/R3E_rt_{name}_L0.1_seed2.json',
                     f'attack/R3E_rt_{name}_L0.1.json')
            r = g1_row(paths)
            c1 = g1_checks(r, b)
        else:
            r = keep_orch['systems'][name]
            c1 = r['checks']
        L = live_row(f'R3E_rt_{name}' if name in E_NAMES else f'L_rt_{name}')
        g2, g3 = g23(L, R)
        fails = [k for k, v in list(c1.items()) + list(g2.items()) + list(g3.items()) if not v]
        mean = (L['sung']['rt_nd_live'] + L['sung_low']['rt_nd_live']) / 2
        fsum = L['sung']['fresh_per100'] + L['sung_low']['fresh_per100']
        rec = {'g1': {k: r[k] for k in ('rt_nd', 'harsh', 'seed2', 'us', 'fresh', 'ds_rt_nd', 'ds_harsh', 'ds_fresh')},
               'g1_checks': c1, 'g2_checks': g2, 'g3_checks': g3, 'G1': all(c1.values()), 'G2': all(g2.values()),
               'G3': all(g3.values()), 'failed': fails, 'mean_live': mean, 'fresh_sum_live': fsum, 'live': L,
               'seed2_delta': r['seed2'] - b['seed2'], 'idea_E': name in E_NAMES}
        out['systems'][name] = rec
        rows.append((name, rec))
    ok = [(n, r) for n, r in rows if r['G1'] and r['G2'] and r['G3']]
    rank = []
    if ok:
        top = max(r['mean_live'] for _, r in ok)
        band = [(n, r) for n, r in ok if r['mean_live'] >= top - 0.003]
        best = min(band, key=lambda t: (t[1]['fresh_sum_live'], -t[1]['mean_live']))[0]
        rank = sorted(ok, key=lambda t: -t[1]['mean_live'])
        out['pick'] = best
        out['pick_rule'] = (f'highest mean {top:.4f}; within 0.003 of it: {[n for n, _ in band]}; the one with the fewest '
                            f'live fresh clashes: {best}')
    out['passing_by_mean'] = [[n, round(r['mean_live'], 4), round(r['fresh_sum_live'], 2)] for n, r in rank]
    print(f"reactive: bench rt_nd {b['rt_nd']:.4f} harsh {b['harsh']:.4f} seed2 {b['seed2']:.4f} fresh {b['fresh']:.4f}; "
          f"live sung {R['sung']['rt_nd_live']:.4f}/{R['sung']['harsh']:.4f}/{R['sung']['fresh_per100']:.2f} "
          f"sung_low {R['sung_low']['rt_nd_live']:.4f}/{R['sung_low']['harsh']:.4f}/{R['sung_low']['fresh_per100']:.2f} "
          f"v11 {R['phone_v11']['proxies']['fresh_heard']}/{R['phone_v11']['proxies']['harsh_heard']:.4f}")
    for n, r in rows:
        L = r['live']
        print(f"{n:18s} G1 {r['G1']!s:5s} G2 {r['G2']!s:5s} G3 {r['G3']!s:5s} bench {r['g1']['rt_nd']:.4f}/"
              f"{r['g1']['harsh']:.4f}/{r['g1']['fresh']:.4f} s2d {r['seed2_delta']:+.4f} | sung "
              f"{L['sung']['rt_nd_live']:.4f}/{L['sung']['harsh']:.4f}/{L['sung']['fresh_per100']:.2f}/"
              f"{L['sung']['stumbles_per100']:.2f} low {L['sung_low']['rt_nd_live']:.4f}/{L['sung_low']['harsh']:.4f}/"
              f"{L['sung_low']['fresh_per100']:.2f}/{L['sung_low']['stumbles_per100']:.2f} v11 "
              f"{L['phone_v11']['proxies']['fresh_heard']}/{L['phone_v11']['proxies']['harsh_heard']:.4f} mean "
              f"{r['mean_live']:.4f} | failed: {', '.join(r['failed']) or '-'}")
    print('pick:', out.get('pick'), '|', out.get('pick_rule'))
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
