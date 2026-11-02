"""v5rt_1101 phase 1 sanity: bench_rt at L = 0 against bench3/rescore_nd (v4_1101 nd files), oracle against reactive,
and seed2 against v4's seed2 scorer. Prints the largest differences; exits 1 if any exceeds 1e-9.

  python sanity_compare_v1.py <reactive r2_mix_v2 L0 json> <reactive m1_filter_v5lo L0 json> <oracle r2_mix_v2 L0 json>
                              <reactive r2_mix_v2 L0 seed2 json>
"""
import json
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
ND = ABS + '/runs_mini/v4_1101/nd/{0}/results/nd_{0}.json'
S2 = ABS + '/runs_mini/v4_1101/seed2/{0}.json'
DS = ('bach', 'folk', 'pop', 'pop909')
TOL = 1e-9
bad = []


def load(p):
    with open(p) as fh:
        return json.load(fh)


def vs_nd(path, ref_path, label):
    out, ref = load(path), load(ref_path)
    assert out['L'] == 0.0, out['L']
    worst = 0.0
    for ds in DS:
        a, b = out['rows'][ds]['rt_nd'], ref[ds]['music_nd']
        c, d = out['rows'][ds]['double'], ref[ds]['double']
        worst = max(worst, abs(a - b), abs(c - d))
        print(f'  {label} {ds:7s} rt_nd {a:.15f} music_nd {b:.15f} diff {a - b:+.1e} | double diff {c - d:+.1e}')
    a, b = out['headline']['rt_nd'], ref['headline_nd']
    worst = max(worst, abs(a - b))
    print(f'  {label} headline rt_nd {a:.15f} headline_nd {b:.15f} diff {a - b:+.1e}  -> worst {worst:.1e} '
          f"{'OK' if worst <= TOL else 'MISMATCH'}")
    if worst > TOL:
        bad.append(label)


r2, m1, orc, s2 = sys.argv[1:5]
print('reactive at L = 0 vs rescore_nd music_nd and double (v4_1101 nd files):')
vs_nd(r2, ND.format('r2_mix_v2'), 'reactive:r2_mix_v2')
vs_nd(m1, ND.format('m1_filter_v5lo'), 'reactive:m1_filter_v5lo')
print('reactive at L = 0 with --seed2 vs futures/v4_selfiter/seed2.py (v4_1101 seed2 file):')
vs_nd(s2, S2.format('r2_mix_v2'), 'reactive:r2_mix_v2 seed2')
print('oracle:r2_mix_v2 vs reactive:r2_mix_v2, both at L = 0, every share:')
o, r = load(orc), load(r2)
worst = 0.0
for key in ('rt_nd', 'harsh', 'double', 'nd_first', 'harsh_first', 'on_time_right', 'changes', 'late_entry'):
    for ds in DS + ('headline',):
        a = o['headline'][key] if ds == 'headline' else o['rows'][ds][key]
        b = r['headline'][key] if ds == 'headline' else r['rows'][ds][key]
        if a is None or b is None:
            if (a is None) != (b is None):
                worst = float('inf')
            continue
        worst = max(worst, abs(a - b))
print(f"  worst difference {worst:.1e} {'OK' if worst <= TOL else 'MISMATCH'}; oracle notes answered at a plan point: "
      + ', '.join(f"{ds} {o['rows'][ds]['oracle_notes']} (blocked {o['rows'][ds]['oracle_blocked']})" for ds in DS))
if worst > TOL:
    bad.append('oracle vs reactive')
print('ALL OK' if not bad else f'MISMATCH: {bad}')
sys.exit(1 if bad else 0)
