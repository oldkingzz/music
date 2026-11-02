"""Verifier (run v5rt_1101): compare the verifier's own re-scores (scores/VER_*, attack/VER_*) with the files the
round-2 author reported (scores/R2_*, attack/R2_*) and the phase-1 baselines (scores/B_*, attack/*reactive*), then judge
bench_rt's KEEP RULE plus the orchestrator's fresh-clash check from the verifier's numbers only. Val only, tuned on val
(this bundle has no test split). Read-only: prints, writes nothing.

  .venv/bin/python runs_mini/v5rt_1101/verify/compare_v1.py
"""
import json
import os

ABS = '/Users/san/claude_sandbox/music/duet_v0'
SC = ABS + '/runs_mini/v5rt_1101/scores/'
AT = ABS + '/runs_mini/v5rt_1101/attack/'
DS = ('bach', 'folk', 'pop', 'pop909')
SHARES = ('rt_nd', 'harsh', 'double', 'nd_first', 'harsh_first', 'on_time_right', 'changes', 'late_entry')
ATK = ('fresh_clash_per100', 'attack_miss_per100', 'attacks_per100', 'harsh_fresh', 'harsh_bleed')
CH = 'rt:r2_copyhmbold_v1'
RE = 'reactive:r2_mix_v2'


def load(p):
    with open(p) as fh:
        return json.load(fh)


def same(a, b, keys, label):
    """Exact float equality of every share, headline and per dataset."""
    bad = []
    for k in keys:
        if a['headline'][k] != b['headline'][k]:
            bad.append(('headline', k, a['headline'][k], b['headline'][k]))
        for ds in DS:
            if a['rows'][ds][k] != b['rows'][ds][k]:
                bad.append((ds, k, a['rows'][ds][k], b['rows'][ds][k]))
    print(f"{'EQUAL ' if not bad else 'DIFFER'} {label}: {len(keys) * 5 - len(bad)}/{len(keys) * 5} values identical")
    for x in bad[:12]:
        print('   ', x)
    return not bad


mine = {
    ('ch', 0.1, 1): SC + 'VER_rt_r2_copyhmbold_v1_L0.1.json', ('re', 0.1, 1): SC + 'VER_reactive_r2_mix_v2_L0.1.json',
    ('ch', 0.1, 2): SC + 'VER_rt_r2_copyhmbold_v1_L0.1_seed2.json',
    ('re', 0.1, 2): SC + 'VER_reactive_r2_mix_v2_L0.1_seed2.json',
    ('ch', 0.05, 1): SC + 'VER_rt_r2_copyhmbold_v1_L0.05.json', ('re', 0.05, 1): SC + 'VER_reactive_r2_mix_v2_L0.05.json',
    ('ch', 0.15, 1): SC + 'VER_rt_r2_copyhmbold_v1_L0.15.json', ('re', 0.15, 1): SC + 'VER_reactive_r2_mix_v2_L0.15.json'}
theirs = {
    ('ch', 0.1, 1): SC + 'R2_rt_r2_copyhmbold_v1_L0.1.json', ('re', 0.1, 1): SC + 'B_reactive_r2_mix_v2_L0.1.json',
    ('ch', 0.1, 2): SC + 'R2_rt_r2_copyhmbold_v1_L0.1_seed2.json', ('re', 0.1, 2): SC + 'B_reactive_r2_mix_v2_L0.1_seed2.json',
    ('ch', 0.05, 1): SC + 'R2_rt_r2_copyhmbold_v1_L0.05.json', ('re', 0.05, 1): SC + 'B_reactive_r2_mix_v2_L0.05.json',
    ('ch', 0.15, 1): SC + 'R2_rt_r2_copyhmbold_v1_L0.15.json', ('re', 0.15, 1): SC + 'B_reactive_r2_mix_v2_L0.15.json'}
amine = {
    ('ch', 0.1, 1): AT + 'VER_rt_r2_copyhmbold_v1_L0.1.json', ('re', 0.1, 1): AT + 'VER_reactive_r2_mix_v2_L0.1.json',
    ('ch', 0.1, 2): AT + 'VER_rt_r2_copyhmbold_v1_L0.1_seed2.json',
    ('re', 0.1, 2): AT + 'VER_reactive_r2_mix_v2_L0.1_seed2.json',
    ('ch', 0.05, 1): AT + 'VER_rt_r2_copyhmbold_v1_L0.05.json', ('re', 0.05, 1): AT + 'VER_reactive_r2_mix_v2_L0.05.json',
    ('ch', 0.15, 1): AT + 'VER_rt_r2_copyhmbold_v1_L0.15.json', ('re', 0.15, 1): AT + 'VER_reactive_r2_mix_v2_L0.15.json'}
atheirs = {
    ('ch', 0.1, 1): AT + 'R2_rt_r2_copyhmbold_v1_L0.1.json', ('re', 0.1, 1): AT + 'reactive_r2_mix_v2_L0.1.json',
    ('ch', 0.1, 2): AT + 'R2_rt_r2_copyhmbold_v1_L0.1_seed2.json', ('re', 0.1, 2): AT + 'R2_reactive_r2_mix_v2_L0.1_seed2.json',
    ('ch', 0.05, 1): AT + 'R2_rt_r2_copyhmbold_v1_L0.05.json', ('re', 0.05, 1): AT + 'C_reactive_r2_mix_v2_L0.05.json',
    ('ch', 0.15, 1): AT + 'R2_rt_r2_copyhmbold_v1_L0.15.json', ('re', 0.15, 1): AT + 'C_reactive_r2_mix_v2_L0.15.json'}

M = {k: load(p) for k, p in mine.items()}
T = {k: load(p) for k, p in theirs.items() if os.path.exists(p)}
AM = {k: load(p) for k, p in amine.items()}
AT_ = {k: load(p) for k, p in atheirs.items() if os.path.exists(p)}

print('== 0. provenance of the verifier files')
for k, o in M.items():
    want = CH if k[0] == 'ch' else RE
    assert o['spec'] == want and o['L'] == k[1] and o['seed'] == f'seed{k[2]}' and o['split'] == 'val', (k, o['spec'])
    print(f"   {os.path.basename(mine[k]):45s} spec {o['spec']:22s} L {o['L']:<5g} {o['seed']} sha {o['bench_sha256'][:12]}")
for k, o in AM.items():
    want = CH if k[0] == 'ch' else RE
    assert o['spec'] == want and o['L'] == k[1] and o['seed'] == f'seed{k[2]}', (k, o['spec'])
print('   all', len(set(o['bench_sha256'] for o in M.values())), 'distinct bench sha256 among verifier score files')

print('== 1. reproduction (exact float equality, every share, headline + 4 datasets)')
for k in M:
    if k in T:
        same(M[k], T[k], SHARES, f"{os.path.basename(mine[k])} vs {os.path.basename(theirs[k])}")
    else:
        print('    missing reported file', theirs[k])
for k in AM:
    if k in AT_:
        same(AM[k], AT_[k], ATK, f"{os.path.basename(amine[k])} vs {os.path.basename(atheirs[k])}")
    else:
        print('    missing reported attack file', atheirs[k])

print('== 2. numbers quoted in the champion report vs the verifier (4 decimals)')
q = {'rt_nd': 0.6315, 'harsh': 0.0313, 'on_time_right': 0.2945, 'changes': 0.5662, 'nd_first': 0.391,
     'harsh_first': 0.160}
h = M[('ch', 0.1, 1)]['headline']
for k, v in q.items():
    print(f"   {k:14s} reported {v:.4f}  verifier {h[k]:.6f}  {'ok' if abs(h[k] - v) < 6e-4 else 'MISMATCH'}")
qd = {'bach': (0.6790, 0.0315, 2.29), 'folk': (0.6797, 0.0364, 1.56), 'pop': (0.5675, 0.0318, 1.05),
      'pop909': (0.6000, 0.0254, 1.28)}
for ds, (nd, hr, fc) in qd.items():
    r = M[('ch', 0.1, 1)]['rows'][ds]
    f = AM[('ch', 0.1, 1)]['rows'][ds]['fresh_clash_per100']
    ok = abs(r['rt_nd'] - nd) < 6e-5 and abs(r['harsh'] - hr) < 6e-5 and abs(f - fc) < 6e-3
    print(f"   {ds:7s} reported {nd:.4f}/{hr:.4f}/{fc:.2f}  verifier {r['rt_nd']:.4f}/{r['harsh']:.4f}/{f:.2f}  "
          f"{'ok' if ok else 'MISMATCH'}")
print(f"   L=0.05 reported 0.6480 verifier {M[('ch', 0.05, 1)]['headline']['rt_nd']:.4f}; "
      f"L=0.15 reported 0.6151 verifier {M[('ch', 0.15, 1)]['headline']['rt_nd']:.4f}; "
      f"seed2 reported 0.6371 verifier {M[('ch', 0.1, 2)]['headline']['rt_nd']:.4f}")
print(f"   fresh reported 1.55 / seed2 1.62 / L0.05 1.55 / L0.15 1.53; verifier "
      f"{AM[('ch', 0.1, 1)]['headline']['fresh_clash_per100']:.2f} / {AM[('ch', 0.1, 2)]['headline']['fresh_clash_per100']:.2f}"
      f" / {AM[('ch', 0.05, 1)]['headline']['fresh_clash_per100']:.2f} / {AM[('ch', 0.15, 1)]['headline']['fresh_clash_per100']:.2f}")

print('== 3. KEEP RULE at L = 0.1 (verifier numbers only)')
c, r, c2, r2 = M[('ch', 0.1, 1)], M[('re', 0.1, 1)], M[('ch', 0.1, 2)], M[('re', 0.1, 2)]
ac, ar = AM[('ch', 0.1, 1)], AM[('re', 0.1, 1)]
rows = []
ch_, rh_ = c['headline'], r['headline']
rows.append(('headline rt_nd >= reactive + 0.005', ch_['rt_nd'] >= rh_['rt_nd'] + 0.005,
             f"{ch_['rt_nd']:.6f} vs {rh_['rt_nd']:.6f} (delta {ch_['rt_nd'] - rh_['rt_nd']:+.4f})"))
rows.append(('headline harsh <= reactive', ch_['harsh'] <= rh_['harsh'],
             f"{ch_['harsh']:.6f} vs {rh_['harsh']:.6f} (delta {ch_['harsh'] - rh_['harsh']:+.4f})"))
for ds in DS:
    a, b = c['rows'][ds]['harsh'], r['rows'][ds]['harsh']
    rows.append((f'{ds} harsh <= reactive + 0.005', a <= b + 0.005, f'{a:.4f} vs {b:.4f} (delta {a - b:+.4f})'))
for ds in DS:
    a, b = c['rows'][ds]['rt_nd'], r['rows'][ds]['rt_nd']
    rows.append((f'{ds} rt_nd >= reactive - 0.01', a >= b - 0.01, f'{a:.4f} vs {b:.4f} (delta {a - b:+.4f})'))
d2 = c2['headline']['rt_nd'] - r2['headline']['rt_nd']
rows.append(('seed2 headline delta >= +0.003', d2 >= 0.003,
             f"{c2['headline']['rt_nd']:.4f} vs {r2['headline']['rt_nd']:.4f} (delta {d2:+.4f})"))
rows.append(('mean us per decision < 2000', ch_['us_mean'] < 2000, f"{ch_['us_mean']:.0f} us"))
fa, fr = ac['headline']['fresh_clash_per100'], ar['headline']['fresh_clash_per100']
rows.append(('fresh clashes headline <= reactive', fa <= fr, f'{fa:.3f} vs {fr:.3f} per 100 notes'))
for ds in DS:
    a, b = ac['rows'][ds]['fresh_clash_per100'], ar['rows'][ds]['fresh_clash_per100']
    rows.append((f'{ds} fresh <= reactive + 1.0', a <= b + 1.0, f'{a:.3f} vs {b:.3f} (delta {a - b:+.3f})'))
for name, ok, detail in rows:
    print(f"   {'PASS' if ok else 'FAIL'}  {name:38s} {detail}")
print(f"   KEEP RULE + fresh-clash check: {'ALL PASS' if all(x[1] for x in rows) else 'FAILS'} "
      f"({sum(not x[1] for x in rows)} of {len(rows)} conditions fail)")

print('== 4. other latencies / seeds: champion minus reactive (verifier numbers)')
for L, s in ((0.05, 1), (0.1, 1), (0.15, 1), (0.1, 2)):
    a, b = M[('ch', L, s)], M[('re', L, s)]
    fa, fb = AM[('ch', L, s)], AM[('re', L, s)]
    print(f"   L={L:<4g} seed{s}: rt_nd {a['headline']['rt_nd']:.4f} vs {b['headline']['rt_nd']:.4f} "
          f"({a['headline']['rt_nd'] - b['headline']['rt_nd']:+.4f}); harsh {a['headline']['harsh']:.4f} vs "
          f"{b['headline']['harsh']:.4f} ({a['headline']['harsh'] - b['headline']['harsh']:+.4f}); fresh/100 "
          f"{fa['headline']['fresh_clash_per100']:.2f} vs {fb['headline']['fresh_clash_per100']:.2f}; wrong attacks/100 "
          f"{fa['headline']['attack_miss_per100']:.1f} vs {fb['headline']['attack_miss_per100']:.1f}")
    for ds in DS:
        x, y = a['rows'][ds], b['rows'][ds]
        u, v = fa['rows'][ds], fb['rows'][ds]
        print(f"      {ds:7s} rt_nd {x['rt_nd']:.4f} vs {y['rt_nd']:.4f} ({x['rt_nd'] - y['rt_nd']:+.4f})  harsh "
              f"{x['harsh']:.4f} vs {y['harsh']:.4f} ({x['harsh'] - y['harsh']:+.4f})  fresh {u['fresh_clash_per100']:.2f}"
              f" vs {v['fresh_clash_per100']:.2f} ({u['fresh_clash_per100'] - v['fresh_clash_per100']:+.2f})  wrong-att "
              f"{u['attack_miss_per100']:.1f} vs {v['attack_miss_per100']:.1f}")
