"""v5rt_1101: bench_rt's KEEP RULE (see futures/v5_rt/bench_rt.py docstring) for one candidate, from bench_rt JSON files.
Val only (tuned on val). Prints every condition with its numbers; exits 0 if all pass, 1 otherwise.

  python keep_rt_v1.py <candidate L0.1 seed1> <reactive:r2_mix_v2 L0.1 seed1> <candidate L0.1 seed2> <reactive seed2>
"""
import json
import sys

DS = ('bach', 'folk', 'pop', 'pop909')


def load(p):
    with open(p) as fh:
        return json.load(fh)


c, r, c2, r2 = (load(p) for p in sys.argv[1:5])
assert c['L'] == r['L'] == c2['L'] == r2['L'] == 0.1, 'keep rule is at L = 0.1'
assert c['seed'] == r['seed'] == 'seed1' and c2['seed'] == r2['seed'] == 'seed2', 'seed1 / seed2 order'
assert r['spec'] == r2['spec'] == 'reactive:r2_mix_v2' and c['spec'] == c2['spec'], 'reference and candidate specs'
assert len({x['bench_sha256'] for x in (c, r, c2, r2)}) == 1, 'all four files must come from the same bench_rt'
rows = []
ch, rh = c['headline'], r['headline']
rows.append(('headline rt_nd >= reactive + 0.005', ch['rt_nd'] >= rh['rt_nd'] + 0.005,
             f"{ch['rt_nd']:.4f} vs {rh['rt_nd']:.4f} (delta {ch['rt_nd'] - rh['rt_nd']:+.4f})"))
rows.append(('headline harsh <= reactive', ch['harsh'] <= rh['harsh'],
             f"{ch['harsh']:.4f} vs {rh['harsh']:.4f} (delta {ch['harsh'] - rh['harsh']:+.4f})"))
for ds in DS:
    a, b = c['rows'][ds]['harsh'], r['rows'][ds]['harsh']
    rows.append((f'{ds} harsh <= reactive + 0.005', a <= b + 0.005, f'{a:.4f} vs {b:.4f} (delta {a - b:+.4f})'))
for ds in DS:
    a, b = c['rows'][ds]['rt_nd'], r['rows'][ds]['rt_nd']
    rows.append((f'{ds} rt_nd >= reactive - 0.01', a >= b - 0.01, f'{a:.4f} vs {b:.4f} (delta {a - b:+.4f})'))
d2 = c2['headline']['rt_nd'] - r2['headline']['rt_nd']
rows.append(('seed2 headline delta >= +0.003', d2 >= 0.003,
             f"{c2['headline']['rt_nd']:.4f} vs {r2['headline']['rt_nd']:.4f} (delta {d2:+.4f})"))
rows.append(('mean us per decision < 2000', ch['us_mean'] < 2000, f"{ch['us_mean']:.0f} us"))
for name, ok, detail in rows:
    print(f"{'PASS' if ok else 'FAIL'}  {name:36s} {detail}")
ok = all(x[1] for x in rows)
print(f"{c['spec']}: {'KEEP' if ok else 'NOT KEPT'} ({sum(not x[1] for x in rows)} of {len(rows)} conditions fail)")
sys.exit(0 if ok else 1)
