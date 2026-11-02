"""Orchestrator's own check of the round-3 selection, read straight from the files on disk (no agent code reused
except keep_table_v1's G1 checks, which the orchestrator wrote in round 2).

Rule (state_R2.json, round3_selection_rule_fixed_now), vs reactive:r2_mix_v2; val only, tuned on val, no test split:
  G1: bench_rt keep rule 17/17 at L = 0.1 (keep_table_v1.checks).
  G2: live bench, wired, val Bach headline: sung rt_nd_live >= reactive + 0.005; sung_low rt_nd_live >= reactive;
      on both voices harsh <= reactive and fresh_per100 <= reactive.
  G3: phone_v11 proxies: fresh_heard <= reactive's; harsh_heard <= reactive's + 0.005.
  Rank: among G1 & G2 & G3, highest mean(sung, sung_low) rt_nd_live; within 0.003 of the top, fewer live fresh
        clashes (count, sung + sung_low).
Every live file is checked for the frozen code hashes, spec, voice, lat, split and song count.

  .venv/bin/python runs_mini/v5rt_1101/tools/r3_gates_orch_v1.py --out <new json>
"""
import argparse
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
sys.path.insert(0, os.path.join(RUN, 'tools'))
import keep_table_v1 as KT  # noqa: E402

CODE = {'live_bench_rt.py': '46b43f91bf4bfa03f04c07ecdb543368394b6279f198cc8a21c435a8ab2bd8fd',
        'voice_v1.py': 'ed395a5acb7c4a1dbf341c22f41b3b772f74feaeb492880ff93ae8799220e2f9',
        'bench_rt.py': 'f1eb524aa939b5ba54704532ac7b5eb536659ceaaef32c3701934f9cf7d3e772'}
OLD = {'r1_A_prior_v1': ('A_rt_', ''), 'r1_B_cons_v1': ('', 'rt_'), 'r1_C_revoice_v2': ('C_rt_', ''),
       'r2_copy_v1': ('R2_rt_', ''), 'r2_copyhm_v1': ('R2_rt_', ''), 'r2_copyhmbold_v1': ('R2_rt_', ''),
       'r2_copysafe_v1': ('R2_rt_', '')}
NEW = {'D': ('regguard', 'regguardsafe', 'reghm', 'regsafe'), 'E': ('trk', 'trkhold', 'trkholdk', 'trksafe'),
       'F': ('bold', 'calm', 'guard', 'lowsafe'), 'X': ('regtrkdk', 'regtrkst', 'regtrkstfree', 'regtrkstsafe')}
VOICES = ('sung_wired', 'sung_low_wired', 'phone_v11', 'clean_wired', 'sung_bt', 'phone_twinkle')


def bench_paths(name):
    if name in OLD:
        p, a = OLD[name]
        if name == 'r1_B_cons_v1':
            return (f'scores/{name}_L0.1.json', f'scores/{name}_L0.1_seed2.json', f'attack/rt_{name}_L0.1.json')
        return (f'scores/{p}{name}_L0.1.json', f'scores/{p}{name}_L0.1_seed2.json', f'attack/{p}{name}_L0.1.json')
    p = f'R3{name[3]}_rt_'
    return (f'scores/{p}{name}_L0.1.json', f'scores/{p}{name}_L0.1_seed2.json', f'attack/{p}{name}_L0.1.json')


def live_path(name, voice):
    if name == 'reactive':
        return os.path.join(RUN, 'live_bench', f'L_reactive_r2_mix_v2_{voice}.json')
    p = 'L_rt_' if name in OLD else f'R3{name[3]}_rt_'
    return os.path.join(RUN, 'live_bench', f'{p}{name}_{voice}.json')


def live(name, voice):
    path = live_path(name, voice)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    spec = 'reactive:r2_mix_v2' if name == 'reactive' else f'rt:{name}'
    v, lat = (voice.rsplit('_', 1) if voice.endswith(('_wired', '_bt')) else (voice, 'wired'))
    assert d['spec'] == spec and d['voice'] == v and d['lat'] == lat and d['split'] == 'val', (path, d['spec'])
    assert d['code'] == CODE, (path, d['code'])
    if voice == 'phone_v11':
        return {'fresh_heard': d['phone']['proxies']['fresh_heard'],
                'harsh_heard': d['phone']['proxies']['harsh_heard'], 'digest': d['digest']}
    if voice == 'phone_twinkle':
        h = d['headline']
        return {'rt_nd_live': h['rt_nd_live'], 'harsh': h['harsh'], 'fresh': h['fresh'],
                'stumbles_per100': h['stumbles_per100'], 'digest': d['digest']}
    assert d['songs_n'] == 17 and d['count_in'] == 4 and d['bpm'] == 90.0, path
    h = d['headline']
    return {k: h[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100', 'on_time_right',
                              'late_entry', 'notes')} | {'digest': d['digest']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out_path = ap.parse_args().out
    pool = list(OLD) + [f'r3_{k}_{v}_v1' for k, vs in NEW.items() for v in vs]
    b1 = KT.row(KT.REACT)
    rb = {v: live('reactive', v) for v in VOICES}
    rows = {}
    for name in pool:
        r1 = KT.row(bench_paths(name))
        c1 = KT.checks(r1, b1)
        L = {v: live(name, v) for v in VOICES}
        s, sl, v11 = L['sung_wired'], L['sung_low_wired'], L['phone_v11']
        bs, bsl, bv = rb['sung_wired'], rb['sung_low_wired'], rb['phone_v11']
        c2 = {'sung rt_nd +0.005': s['rt_nd_live'] >= bs['rt_nd_live'] + 0.005,
              'sung_low rt_nd >= reactive': sl['rt_nd_live'] >= bsl['rt_nd_live'],
              'sung harsh <= reactive': s['harsh'] <= bs['harsh'],
              'sung_low harsh <= reactive': sl['harsh'] <= bsl['harsh'],
              'sung fresh <= reactive': s['fresh_per100'] <= bs['fresh_per100'],
              'sung_low fresh <= reactive': sl['fresh_per100'] <= bsl['fresh_per100']}
        c3 = {'v11 fresh_heard <= reactive': v11['fresh_heard'] <= bv['fresh_heard'],
              'v11 harsh_heard <= reactive + 0.005': v11['harsh_heard'] <= bv['harsh_heard'] + 0.005}
        rows[name] = {'G1': all(c1.values()), 'G2': all(c2.values()), 'G3': all(c3.values()),
                      'failed': [k for c in (c1, c2, c3) for k, ok in c.items() if not ok],
                      'mean_live': (s['rt_nd_live'] + sl['rt_nd_live']) / 2,
                      'live_fresh_count': s['fresh'] + sl['fresh'], 'bench': {k: r1[k] for k in
                      ('rt_nd', 'harsh', 'seed2', 'fresh', 'us', 'ds_rt_nd', 'ds_fresh')}, 'live': L}
    ok = {n: r for n, r in rows.items() if r['G1'] and r['G2'] and r['G3']}
    top = max(r['mean_live'] for r in ok.values())
    band = sorted((n for n, r in ok.items() if top - r['mean_live'] <= 0.003),
                  key=lambda n: (rows[n]['live_fresh_count'], -rows[n]['mean_live']))
    out = {'reactive_bench': b1, 'reactive_live': rb, 'rows': rows, 'n_pool': len(pool), 'n_pass': len(ok),
           'top_mean': top, 'band': band, 'winner': band[0],
           'note': 'val only (17 Bach chorales live; bench_rt val sets); systems tuned on val; no test split'}
    print(f"pool {len(pool)}, pass G1&G2&G3 {len(ok)}, top mean {top:.5f}, band {band}, WINNER {band[0]}")
    print(f"{'system':24s} G1 G2 G3  mean   sung   s_low  fresh#  stmb   v11f  v11h   bench  b_fresh  failed")
    for n in sorted(rows, key=lambda n: -rows[n]['mean_live']):
        r = rows[n]
        L = r['live']
        print(f"{n:24s} {'Y' if r['G1'] else '.'}  {'Y' if r['G2'] else '.'}  {'Y' if r['G3'] else '.'}  "
              f"{r['mean_live']:.4f} {L['sung_wired']['rt_nd_live']:.4f} {L['sung_low_wired']['rt_nd_live']:.4f} "
              f"{r['live_fresh_count']:5d}  {L['sung_wired']['stumbles_per100']:5.2f}  "
              f"{L['phone_v11']['fresh_heard']:3d}  {L['phone_v11']['harsh_heard']:.4f}  {r['bench']['rt_nd']:.4f} "
              f"{r['bench']['fresh']:6.2f}  {'; '.join(r['failed'])[:90]}")
    print('reactive live', {v: {k: (round(x, 4) if isinstance(x, float) else x) for k, x in d.items()
                                if k != 'digest'} for v, d in rb.items() if d})
    w = rows[band[0]]['live']
    print('winner live', {v: {k: (round(x, 4) if isinstance(x, float) else x) for k, x in d.items()
                              if k != 'digest'} for v, d in w.items() if d})
    with open(out_path, 'x') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
