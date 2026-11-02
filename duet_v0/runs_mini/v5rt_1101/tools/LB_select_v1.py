"""Round 3 phase 1: the baseline table of the live bench and the round-3 selection rule's G2 / G3 (state_R2.json,
binding) applied to the round-1/2 systems; G1 is read from runs_mini/v5rt_1101/verify/keep_table_orch_v1.json. Also checks
that the determinism repeat (D_repeat_*.json) equals its L_ file in every field but the timings. Val only (17 Bach
chorales + Twinkle, tuned on val, no test split) and the two phone files. Read-only on inputs; writes one new JSON.

G2: wired, val Bach headline: 'sung' rt_nd_live >= reactive + 0.005; 'sung_low' rt_nd_live >= reactive; on both voices
    harsh <= reactive and fresh_per100 <= reactive (reactive = RTDuet(reactive:r2_mix_v2) on the same audio).
G3: phone_v11 fresh_heard count <= reactive's and harsh_heard <= reactive's + 0.005.
Rank: among G1 & G2 & G3 the highest mean of rt_nd_live on 'sung' and 'sung_low'; within 0.003 the one with fewer live
fresh clashes (sum of both voices). If none passes G3: best of G1 & G2, flagged. If none passes G2: the page keeps the
round-2 systems.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/tools/LB_select_v1.py --out runs_mini/v5rt_1101/live_bench/<new>.json
"""
import argparse
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
LB = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'live_bench')
KEEP = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'verify', 'keep_table_orch_v1.json')
REACTIVE = 'reactive:r2_mix_v2'
SYSTEMS = ('r2_copyhmbold_v1', 'r2_copyhm_v1', 'r2_copy_v1', 'r2_copysafe_v1', 'r1_A_prior_v1', 'r1_B_cons_v1',
           'r1_C_revoice_v2')
ENGINES = (REACTIVE,) + tuple('rt:' + s for s in SYSTEMS)
RUNS = [('clean', 'wired'), ('sung', 'wired'), ('sung_low', 'wired'), ('phone_twinkle', None), ('phone_v11', None),
        ('sung', 'bt')]


def fname(spec, voice, lat):
    base = 'L_' + spec.replace(':', '_') + '_' + voice
    return base + ('_' + lat if lat else '') + '.json'


def load(spec, voice, lat):
    p = os.path.join(LB, fname(spec, voice, lat))
    return (json.load(open(p)), p) if os.path.exists(p) else (None, p)


def strip(d):
    return {k: v for k, v in d.items() if k not in ('timing', 'time')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(LB + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit('--out must be a new .json under ' + LB)
    res = {}
    rows = []
    for spec in ENGINES:
        for voice, lat in RUNS:
            r, p = load(spec, voice, lat)
            if r is None:
                continue
            row = {'engine': spec, 'voice': voice, 'lat': lat or 'phone (Li 0, Lo 10 ms)', 'file': os.path.relpath(p, ABS),
                   'digest': r['digest']}
            if voice in ('clean', 'sung', 'sung_low'):
                m = r['headline']
                row.update({k: m[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100',
                                              'stumbles_own_per100', 'on_time_right', 'wrong_att_per100', 'late_entry',
                                              'double', 'first_100ms_hit', 'first_100ms_harsh', 'blips', 'notes')})
                row['twinkle_rt_nd_live'] = r['twinkle']['rt_nd_live']
                row['twinkle_fresh_per100'] = r['twinkle']['fresh_per100']
                row['tracker_pc_right'] = r['tracker']['bach']['pc_right']
            elif voice == 'phone_twinkle':
                m = r['phone']['metrics']
                row.update({k: m[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100',
                                              'on_time_right', 'wrong_att_per100', 'late_entry', 'blips', 'notes')})
                row['sampled_fit'] = r['phone']['sampled']['fit']
                row['sampled_harsh'] = r['phone']['sampled']['harsh']
                row['fresh_heard'] = r['phone']['proxies']['fresh_heard']
                row['harsh_heard'] = r['phone']['proxies']['harsh_heard']
            else:
                pr = r['phone']['proxies']
                row.update({k: pr[k] for k in ('answered', 'sung', 'attacks', 'cons_heard', 'changes', 'on_time_cons',
                                               'fresh_heard', 'harsh_heard', 'blips')})
            row['us_mean'] = r['timing']['us_mean']
            row['seconds'] = r['timing']['total_s']
            rows.append(row)
    res['rows'] = rows
    get = {(r['engine'], r['voice'], r['lat']): r for r in rows}
    W = 'wired'
    P = 'phone (Li 0, Lo 10 ms)'
    re_s, re_l, re_v = get[(REACTIVE, 'sung', W)], get[(REACTIVE, 'sung_low', W)], get[(REACTIVE, 'phone_v11', P)]
    keep = json.load(open(KEEP))
    sel = {}
    for s in SYSTEMS:
        e = 'rt:' + s
        cs, cl, cv = get[(e, 'sung', W)], get[(e, 'sung_low', W)], get[(e, 'phone_v11', P)]
        g2 = {'sung rt_nd_live >= reactive + 0.005': cs['rt_nd_live'] >= re_s['rt_nd_live'] + 0.005,
              'sung_low rt_nd_live >= reactive': cl['rt_nd_live'] >= re_l['rt_nd_live'],
              'sung harsh <= reactive': cs['harsh'] <= re_s['harsh'],
              'sung_low harsh <= reactive': cl['harsh'] <= re_l['harsh'],
              'sung fresh_per100 <= reactive': cs['fresh_per100'] <= re_s['fresh_per100'],
              'sung_low fresh_per100 <= reactive': cl['fresh_per100'] <= re_l['fresh_per100']}
        g3 = {'v11 fresh_heard <= reactive': cv['fresh_heard'] <= re_v['fresh_heard'],
              'v11 harsh_heard <= reactive + 0.005': cv['harsh_heard'] <= re_v['harsh_heard'] + 0.005}
        sel[s] = {'G1': bool(keep['systems'][s]['passes']), 'G2': all(g2.values()), 'G3': all(g3.values()),
                  'G2_checks': g2, 'G3_checks': g3,
                  'mean_sung_sunglow': (cs['rt_nd_live'] + cl['rt_nd_live']) / 2,
                  'fresh_live_sum': cs['fresh'] + cl['fresh'],
                  'deltas_vs_reactive': {'sung_rt_nd': cs['rt_nd_live'] - re_s['rt_nd_live'],
                                         'sung_low_rt_nd': cl['rt_nd_live'] - re_l['rt_nd_live'],
                                         'sung_harsh': cs['harsh'] - re_s['harsh'],
                                         'sung_low_harsh': cl['harsh'] - re_l['harsh'],
                                         'sung_fresh_per100': cs['fresh_per100'] - re_s['fresh_per100'],
                                         'sung_low_fresh_per100': cl['fresh_per100'] - re_l['fresh_per100'],
                                         'v11_fresh_heard': cv['fresh_heard'] - re_v['fresh_heard'],
                                         'v11_harsh_heard': cv['harsh_heard'] - re_v['harsh_heard']}}
    res['reactive'] = {'sung': {k: re_s[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh')},
                       'sung_low': {k: re_l[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh')},
                       'phone_v11': {k: re_v[k] for k in ('fresh_heard', 'harsh_heard')},
                       'mean_sung_sunglow': (re_s['rt_nd_live'] + re_l['rt_nd_live']) / 2}
    res['selection'] = sel

    def pick(names):
        if not names:
            return None
        names = sorted(names, key=lambda s: -sel[s]['mean_sung_sunglow'])
        top = sel[names[0]]['mean_sung_sunglow']
        near = [s for s in names if top - sel[s]['mean_sung_sunglow'] <= 0.003]
        return min(near, key=lambda s: (sel[s]['fresh_live_sum'], -sel[s]['mean_sung_sunglow']))

    full = [s for s in SYSTEMS if sel[s]['G1'] and sel[s]['G2'] and sel[s]['G3']]
    g12 = [s for s in SYSTEMS if sel[s]['G1'] and sel[s]['G2']]
    if full:
        verdict = {'selected': pick(full), 'flag': None, 'passing_all': full}
    elif g12:
        verdict = {'selected': pick(g12), 'flag': 'none passes G3: best of G1 & G2', 'passing_all': []}
    else:
        verdict = {'selected': None, 'flag': 'none passes G2: the page keeps the round-2 systems', 'passing_all': []}
    verdict['passing_G1_G2'] = g12
    verdict['ranking_all_by_mean'] = sorted(SYSTEMS, key=lambda s: -sel[s]['mean_sung_sunglow'])
    res['verdict'] = verdict
    # determinism repeat
    det = {}
    for fn in sorted(os.listdir(LB)):
        if fn.startswith('D_repeat_') and fn.endswith('.json'):
            twin = 'L_' + fn[len('D_repeat_'):]
            d1, d2 = json.load(open(os.path.join(LB, fn))), json.load(open(os.path.join(LB, twin)))
            det[fn] = {'twin': twin, 'same_digest': d1['digest'] == d2['digest'],
                       'same_except_timing': strip(d1) == strip(d2)}
    res['determinism'] = det
    # compact text table
    lines = ['engine                 voice/lat      rt_nd_live  harsh  fresh/100 stumbles/100 on_time_right wrong_att/100'
             '  twinkle_rt_nd']
    for r in rows:
        if r['voice'] in ('clean', 'sung', 'sung_low', 'phone_twinkle'):
            vl = r['voice'] + '/' + (r['lat'] if r['voice'] != 'phone_twinkle' else 'phone')
            tw = f"{r['twinkle_rt_nd_live']:.4f}" if 'twinkle_rt_nd_live' in r else f"(sampled fit {r['sampled_fit']:.3f})"
            lines.append(f"{r['engine']:22s} {vl:14s} {r['rt_nd_live']:.4f}     {r['harsh']:.4f} {r['fresh_per100']:6.2f}"
                         f"    {r['stumbles_per100']:6.2f}       {r['on_time_right']:.4f}      {r['wrong_att_per100']:6.1f}"
                         f"       {tw}")
    lines.append('phone_v11 proxies (no truth): engine answered sung attacks cons_heard changes on_time_cons fresh_heard '
                 'harsh_heard')
    for r in rows:
        if r['voice'] == 'phone_v11':
            lines.append(f"  {r['engine']:22s} {r['answered']} {r['sung']} {r['attacks']} {r['cons_heard']} "
                         f"{r['changes']} {r['on_time_cons']} {r['fresh_heard']} {r['harsh_heard']:.4f}")
    res['table_text'] = '\n'.join(lines)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print(res['table_text'])
    print()
    for s in SYSTEMS:
        x = sel[s]
        print(f"{s:18s} G1 {x['G1']} G2 {x['G2']} G3 {x['G3']} mean {x['mean_sung_sunglow']:.4f} fresh_sum "
              f"{x['fresh_live_sum']} failed: {[k for k, v in {**x['G2_checks'], **x['G3_checks']}.items() if not v]}")
    print('verdict', json.dumps(verdict))
    print('determinism', json.dumps(det))
    print('wrote', path)


if __name__ == '__main__':
    main()
