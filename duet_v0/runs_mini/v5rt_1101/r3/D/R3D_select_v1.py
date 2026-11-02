"""Round 3, idea D: the binding round-3 selection rule (state_R2.json) applied to the four idea-D variants, read straight
from their score / attack / live_bench files, plus the pool's phase-1 systems (G1 from verify/keep_table_orch_v1.json, G2 /
G3 from live_bench/L_select_R3p1_v1.json) and a few diagnostics from the live run JSONs. The G1 logic is copied from
runs_mini/v5rt_1101/tools/keep_table_v1.py and the G2 / G3 / rank logic from tools/LB_select_v1.py (neither edited).
Val only (tuned on val, no test split) and the two phone files. Read-only on inputs; writes one new JSON.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/D/R3D_select_v1.py --out runs_mini/v5rt_1101/live_bench/<new>.json
"""
import argparse
import json
import os
import sys
from collections import Counter

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
LB = os.path.join(RUN, 'live_bench')
DS = ('bach', 'folk', 'pop', 'pop909')
J = 4
REACT = ('scores/B_reactive_r2_mix_v2_L0.1.json', 'scores/B_reactive_r2_mix_v2_L0.1_seed2.json',
         'attack/reactive_r2_mix_v2_L0.1.json')
VARIANTS = ('r3_D_reghm_v1', 'r3_D_regsafe_v1', 'r3_D_regguard_v1', 'r3_D_regguardsafe_v1')
BASES = {'r3_D_reghm_v1': 'r2_copyhm_v1', 'r3_D_regsafe_v1': 'r2_copysafe_v1', 'r3_D_regguard_v1': 'r2_copy_v1',
         'r3_D_regguardsafe_v1': 'r2_copysafe_v1'}
HARSH = {1, 2, 6, 10, 11}


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def g1_row(paths):                                   # keep_table_v1.row, copied
    s, s2, a = (load(p) for p in paths)
    assert s['L'] == 0.1 and s['seed'] == 'seed1' and s2['seed'] == 'seed2' and s['split'] == 'val', paths
    assert a['L'] == 0.1 and a['seed'] == 'seed1', paths
    return {'rt_nd': s['headline']['rt_nd'], 'harsh': s['headline']['harsh'], 'seed2': s2['headline']['rt_nd'],
            'us': s['headline']['us_mean'], 'on_time_right': s['headline']['on_time_right'],
            'fresh': a['headline']['fresh_clash_per100'], 'miss': a['headline']['attack_miss_per100'],
            'ds_rt_nd': {d: s['rows'][d]['rt_nd'] for d in DS}, 'ds_harsh': {d: s['rows'][d]['harsh'] for d in DS},
            'ds_fresh': {d: a['rows'][d]['fresh_clash_per100'] for d in DS}, 'files': list(paths),
            'bench_sha256': s['bench_sha256']}


def g1_checks(r, b):                                 # keep_table_v1.checks, copied
    c = {'rt_nd +0.005': r['rt_nd'] >= b['rt_nd'] + 0.005, 'harsh <= reactive': r['harsh'] <= b['harsh'],
         'seed2 +0.003': r['seed2'] - b['seed2'] >= 0.003, 'us < 2000': r['us'] < 2000,
         'fresh <= reactive': r['fresh'] <= b['fresh'] + 1e-12}
    for d in DS:
        c[f'{d} harsh <= +0.005'] = r['ds_harsh'][d] <= b['ds_harsh'][d] + 0.005
        c[f'{d} rt_nd >= -0.01'] = r['ds_rt_nd'][d] >= b['ds_rt_nd'][d] - 0.01
        c[f'{d} fresh <= +1.0'] = r['ds_fresh'][d] <= b['ds_fresh'][d] + 1.0
    return c


def lb(prefix, spec, voice, lat):
    base = prefix + spec.replace(':', '_') + '_' + voice + ('_' + lat if lat else '') + '.json'
    with open(os.path.join(LB, base)) as fh:
        return json.load(fh), 'runs_mini/v5rt_1101/live_bench/' + base


def time_shares(res):
    """Share of her counted time (nominal notes from beat 4 on labelled beats, beats) where Duet is silent / harsh / a
    hit, from each song's saved Duet step function (approximate diagnostic: nominal note spans, not the jittered ones)."""
    from theory import CONSONANT, CT                 # noqa: E402
    good = set(CONSONANT) - {0}
    tot = sil = bad = hit = 0.0
    sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
    import bench3                                    # noqa: E402
    songs = {s['id']: s for s in bench3.eval_sets('val')['bach']}
    for r in res['songs']:
        s = songs[r['id']]
        z = [b['z'] for b in s['beats']]
        ton = s['tonic']
        duet = sorted(r['duet'])
        shift = -12 if res['voice'] == 'sung_low' else 0
        for on, off, m in sorted(s['sop']):
            m = m + shift
            for k in range(max(J, int(on)), int(off) + 1):
                a, e = max(on, k), min(off, k + 1)
                if e - a <= 1e-9 or k >= len(z) or z[k] < 0:
                    continue
                pts = sorted({a, e} | {x for d in duet for x in (d[0], d[1]) if a < x < e})
                for x0, x1 in zip(pts, pts[1:]):
                    mid = 0.5 * (x0 + x1)
                    h = -1
                    for d in duet:
                        if d[0] <= mid < d[1]:
                            h = d[2]
                    w = x1 - x0
                    tot += w
                    if h < 0:
                        sil += w
                    else:
                        iv = abs(m - h) % 12
                        bad += w * (iv in HARSH)
                        hit += w * (iv in good and (h - ton) % 12 in CT[z[k]])
    return {'silent': sil / tot, 'harsh': bad / tot, 'hit': hit / tot, 'counted_beats': tot}


def clash_ivs(res):
    c = Counter()
    for r in res['songs']:
        for x in r['fresh_clashes']:
            c[x[3]] += 1
    return dict(sorted(c.items()))


def duet_range(res):
    ms = [d[2] for r in res['songs'] for d in r['duet']]
    ms.sort()
    return {'min': ms[0], 'p10': ms[len(ms) // 10], 'median': ms[len(ms) // 2], 'p90': ms[9 * len(ms) // 10],
            'max': ms[-1]} if ms else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if os.path.exists(path) or not path.endswith('.json'):
        sys.exit('refusing: ' + path)
    sys.path.insert(0, os.path.join(ABS, 'duet_beat'))
    out = {'note': 'val only, tuned on val, no test split; G1 bench_rt L = 0.1 seed 1 / seed 2 + fresh clashes; G2 live '
                   'bench wired val Bach headline; G3 phone_v11 proxies', 'variants': {}}
    b = g1_row(REACT)
    out['reactive_G1'] = b
    re_s, _ = lb('L_', 'reactive:r2_mix_v2', 'sung', 'wired')
    re_l, _ = lb('L_', 'reactive:r2_mix_v2', 'sung_low', 'wired')
    re_v, _ = lb('L_', 'reactive:r2_mix_v2', 'phone_v11', None)
    rs, rl, rv = re_s['headline'], re_l['headline'], re_v['phone']['proxies']
    out['reactive_live'] = {'sung': rs, 'sung_low': rl, 'phone_v11': rv}
    for v in VARIANTS:
        g1 = g1_row((f'scores/R3D_rt_{v}_L0.1.json', f'scores/R3D_rt_{v}_L0.1_seed2.json', f'attack/R3D_rt_{v}_L0.1.json'))
        c1 = g1_checks(g1, b)
        cs_res, fs = lb('R3D_', 'rt:' + v, 'sung', 'wired')
        cl_res, fl = lb('R3D_', 'rt:' + v, 'sung_low', 'wired')
        cv_res, fv = lb('R3D_', 'rt:' + v, 'phone_v11', None)
        tw_res, ft = lb('R3D_', 'rt:' + v, 'phone_twinkle', None)
        cs, cl, cv = cs_res['headline'], cl_res['headline'], cv_res['phone']['proxies']
        g2 = {'sung rt_nd_live >= reactive + 0.005': cs['rt_nd_live'] >= rs['rt_nd_live'] + 0.005,
              'sung_low rt_nd_live >= reactive': cl['rt_nd_live'] >= rl['rt_nd_live'],
              'sung harsh <= reactive': cs['harsh'] <= rs['harsh'],
              'sung_low harsh <= reactive': cl['harsh'] <= rl['harsh'],
              'sung fresh_per100 <= reactive': cs['fresh_per100'] <= rs['fresh_per100'],
              'sung_low fresh_per100 <= reactive': cl['fresh_per100'] <= rl['fresh_per100']}
        g3 = {'v11 fresh_heard <= reactive': cv['fresh_heard'] <= rv['fresh_heard'],
              'v11 harsh_heard <= reactive + 0.005': cv['harsh_heard'] <= rv['harsh_heard'] + 0.005}
        code = {k: cs_res['code'][k] for k in cs_res['code']}
        base_s, _ = lb('L_', 'rt:' + BASES[v], 'sung', 'wired')
        base_l, _ = lb('L_', 'rt:' + BASES[v], 'sung_low', 'wired')
        base_v, _ = lb('L_', 'rt:' + BASES[v], 'phone_v11', None)
        out['variants'][v] = {
            'base': BASES[v], 'G1': all(c1.values()), 'G2': all(g2.values()), 'G3': all(g3.values()),
            'failed': [k for k, x in {**c1, **g2, **g3}.items() if not x],
            'G1_row': g1, 'G1_checks': c1, 'G2_checks': g2, 'G3_checks': g3,
            'sung': {k: cs[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100',
                                        'stumbles_own_per100', 'on_time_right', 'wrong_att_per100', 'late_entry',
                                        'double', 'blips')},
            'sung_low': {k: cl[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100',
                                            'stumbles_own_per100', 'on_time_right', 'wrong_att_per100', 'late_entry',
                                            'double', 'blips')},
            'phone_v11': cv, 'phone_twinkle': tw_res['phone']['metrics'],
            'phone_twinkle_sampled_fit': tw_res['phone']['sampled']['fit'],
            'twinkle_rendered': {'sung': cs_res['twinkle']['rt_nd_live'], 'sung_low': cl_res['twinkle']['rt_nd_live']},
            'mean_sung_sunglow': (cs['rt_nd_live'] + cl['rt_nd_live']) / 2,
            'fresh_live_sum': cs['fresh'] + cl['fresh'],
            'base_live': {'sung': base_s['headline']['rt_nd_live'], 'sung_low': base_l['headline']['rt_nd_live'],
                          'sung_low_harsh': base_l['headline']['harsh'], 'sung_low_fresh': base_l['headline']['fresh_per100'],
                          'v11_fresh_heard': base_v['phone']['proxies']['fresh_heard'],
                          'v11_harsh_heard': base_v['phone']['proxies']['harsh_heard']},
            'diag': {'sung_low_time': time_shares(cl_res), 'sung_time': time_shares(cs_res),
                     'sung_low_clash_iv': clash_ivs(cl_res), 'sung_clash_iv': clash_ivs(cs_res),
                     'sung_low_duet_range': duet_range(cl_res), 'sung_duet_range': duet_range(cs_res),
                     'sung_low_decisions': cl_res['decision_counts'], 'sung_decisions': cs_res['decision_counts']},
            'files': {'sung': fs, 'sung_low': fl, 'phone_v11': fv, 'phone_twinkle': ft}, 'code_sha256': code}
    # diagnostics for the references on the same audio
    refs = {}
    for spec in ('reactive:r2_mix_v2', 'rt:r2_copyhm_v1', 'rt:r2_copysafe_v1', 'rt:r2_copy_v1'):
        r_l, _ = lb('L_', spec, 'sung_low', 'wired')
        r_s, _ = lb('L_', spec, 'sung', 'wired')
        refs[spec] = {'sung_low_time': time_shares(r_l), 'sung_time': time_shares(r_s),
                      'sung_low_clash_iv': clash_ivs(r_l), 'sung_clash_iv': clash_ivs(r_s),
                      'sung_low_duet_range': duet_range(r_l), 'sung_low_decisions': r_l['decision_counts']}
    out['reference_diag'] = refs
    # the pool: phase-1 systems + the idea-D variants (other round-3 letters are added by the orchestrator)
    p1 = json.load(open(os.path.join(LB, 'L_select_R3p1_v1.json')))
    pool = {}
    for s, x in p1['selection'].items():
        pool[s] = {'G1': x['G1'], 'G2': x['G2'], 'G3': x['G3'], 'mean': x['mean_sung_sunglow'],
                   'fresh_sum': x['fresh_live_sum']}
    for v, x in out['variants'].items():
        pool[v] = {'G1': x['G1'], 'G2': x['G2'], 'G3': x['G3'], 'mean': x['mean_sung_sunglow'],
                   'fresh_sum': x['fresh_live_sum']}

    def pick(names):                                 # LB_select_v1.pick, copied
        if not names:
            return None
        names = sorted(names, key=lambda s: -pool[s]['mean'])
        top = pool[names[0]]['mean']
        near = [s for s in names if top - pool[s]['mean'] <= 0.003]
        return min(near, key=lambda s: (pool[s]['fresh_sum'], -pool[s]['mean']))

    full = [s for s in pool if pool[s]['G1'] and pool[s]['G2'] and pool[s]['G3']]
    g12 = [s for s in pool if pool[s]['G1'] and pool[s]['G2']]
    out['pool'] = pool
    out['verdict_phase1_plus_D'] = {'passing_all': sorted(full, key=lambda s: -pool[s]['mean']),
                                    'selected': pick(full), 'passing_G1_G2': sorted(g12, key=lambda s: -pool[s]['mean'])}
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1)
    print(f"reactive live: sung {rs['rt_nd_live']:.4f}/{rs['harsh']:.4f}/{rs['fresh_per100']:.2f}  sung_low "
          f"{rl['rt_nd_live']:.4f}/{rl['harsh']:.4f}/{rl['fresh_per100']:.2f}  v11 {rv['fresh_heard']}/{rv['harsh_heard']:.4f}")
    for v, x in out['variants'].items():
        g = x['G1_row']
        print(f"{v:22s} G1 {x['G1']} G2 {x['G2']} G3 {x['G3']} | bench rt_nd {g['rt_nd']:.4f} harsh {g['harsh']:.4f} "
              f"seed2 d {g['seed2'] - b['seed2']:+.4f} fresh {g['fresh']:.2f} | sung {x['sung']['rt_nd_live']:.4f}/"
              f"{x['sung']['harsh']:.4f}/{x['sung']['fresh_per100']:.2f} st {x['sung']['stumbles_per100']:.2f} | sung_low "
              f"{x['sung_low']['rt_nd_live']:.4f}/{x['sung_low']['harsh']:.4f}/{x['sung_low']['fresh_per100']:.2f} st "
              f"{x['sung_low']['stumbles_per100']:.2f} | v11 {x['phone_v11']['fresh_heard']}/"
              f"{x['phone_v11']['harsh_heard']:.4f} | mean {x['mean_sung_sunglow']:.4f} fresh_sum {x['fresh_live_sum']} "
              f"| failed {x['failed']}")
        print(f"{'':22s} diag sung_low time {x['diag']['sung_low_time']} clash iv {x['diag']['sung_low_clash_iv']} "
              f"range {x['diag']['sung_low_duet_range']}")
    for spec, d in refs.items():
        print(f"ref {spec:22s} sung_low time {d['sung_low_time']} clash iv {d['sung_low_clash_iv']} range "
              f"{d['sung_low_duet_range']}")
    print('pool verdict', json.dumps(out['verdict_phase1_plus_D']))
    print('wrote', path)


if __name__ == '__main__':
    main()
