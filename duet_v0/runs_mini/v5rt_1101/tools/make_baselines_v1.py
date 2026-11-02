"""v5rt_1101 phase 1: collect every bench_rt baseline number (val split only, tuned on val; no test split exists) into
runs_mini/v5rt_1101/baselines.json (mode 'x', never overwritten) and print a compact table. Checks that every file was
scored by the frozen bench (sha in BENCH_SHA.txt) and that spec / L / seed match its label."""
import json
import os
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
SC = os.path.join(RUN, 'scores')
DS = ('bach', 'folk', 'pop', 'pop909')
SHARES = ('rt_nd', 'harsh', 'double', 'nd_first', 'harsh_first', 'on_time_right', 'changes', 'late_entry')
RUNS = [('reactive:r2_mix_v2', 0.1, 'seed1', 'B_reactive_r2_mix_v2_L0.1.json'),
        ('reactive:m1_filter_v5lo', 0.1, 'seed1', 'B_reactive_m1_filter_v5lo_L0.1.json'),
        ('reactive:third_below', 0.1, 'seed1', 'B_reactive_third_below_L0.1.json'),
        ('oracle:r2_mix_v2', 0.1, 'seed1', 'B_oracle_r2_mix_v2_L0.1.json'),
        ('rt:r0_predict_live_v1', 0.1, 'seed1', 'B_rt_r0_predict_live_v1_L0.1.json'),
        ('reactive:r2_mix_v2', 0.05, 'seed1', 'B_reactive_r2_mix_v2_L0.05.json'),
        ('rt:r0_predict_live_v1', 0.05, 'seed1', 'B_rt_r0_predict_live_v1_L0.05.json'),
        ('reactive:r2_mix_v2', 0.15, 'seed1', 'B_reactive_r2_mix_v2_L0.15.json'),
        ('rt:r0_predict_live_v1', 0.15, 'seed1', 'B_rt_r0_predict_live_v1_L0.15.json'),
        ('reactive:r2_mix_v2', 0.1, 'seed2', 'B_reactive_r2_mix_v2_L0.1_seed2.json'),
        ('rt:r0_predict_live_v1', 0.1, 'seed2', 'B_rt_r0_predict_live_v1_L0.1_seed2.json')]
SANITY = [('reactive:r2_mix_v2', 0.0, 'seed1', 'sanity_reactive_r2_mix_v2_L0.json'),
          ('reactive:m1_filter_v5lo', 0.0, 'seed1', 'sanity_reactive_m1_filter_v5lo_L0.json'),
          ('oracle:r2_mix_v2', 0.0, 'seed1', 'sanity_oracle_r2_mix_v2_L0.json'),
          ('reactive:r2_mix_v2', 0.0, 'seed2', 'sanity_reactive_r2_mix_v2_L0_seed2.json')]


def load(p):
    with open(p) as fh:
        return json.load(fh)


with open(os.path.join(RUN, 'BENCH_SHA.txt')) as fh:
    SHA = fh.read().split()[0]


def entry(spec, L, seed, name):
    d = load(os.path.join(SC, name))
    assert (d['spec'], d['L'], d['seed'], d['bench_sha256']) == (spec, L, seed, SHA), (name, d['spec'], d['L'], d['seed'])
    return {'file': 'runs_mini/v5rt_1101/scores/' + name, 'spec': spec, 'L': L, 'seed': seed,
            'headline': d['headline'], 'rows': d['rows'], 'scored': d['time']}


def label(spec, L, seed):
    return f'{spec} L={L:g}' + (' seed2' if seed == 'seed2' else '')


runs = {label(*r[:3]): entry(*r) for r in RUNS}
sanity = {label(*r[:3]): entry(*r) for r in SANITY}

# L = 0 checks: reactive = rescore_nd music_nd (v4_1101 nd files, and v4 seed2 scorer); oracle = reactive.
checks = {}
for name, key, ref in (('r2_mix_v2', 'reactive:r2_mix_v2 L=0', 'runs_mini/v4_1101/nd/r2_mix_v2/results/nd_r2_mix_v2.json'),
                       ('m1_filter_v5lo', 'reactive:m1_filter_v5lo L=0',
                        'runs_mini/v4_1101/nd/m1_filter_v5lo/results/nd_m1_filter_v5lo.json'),
                       ('r2_mix_v2 seed2', 'reactive:r2_mix_v2 L=0 seed2', 'runs_mini/v4_1101/seed2/r2_mix_v2.json')):
    ref_d, s = load(os.path.join(ABS, ref)), sanity[key]
    diff = max([abs(s['rows'][ds]['rt_nd'] - ref_d[ds]['music_nd']) for ds in DS]
               + [abs(s['rows'][ds]['double'] - ref_d[ds]['double']) for ds in DS]
               + [abs(s['headline']['rt_nd'] - ref_d['headline_nd'])])
    checks[f'{key} vs {ref}'] = {'max_abs_diff': diff, 'ok': diff <= 1e-9}
o, r = sanity['oracle:r2_mix_v2 L=0'], sanity['reactive:r2_mix_v2 L=0']
diff = max(abs(o['rows'][ds][k] - r['rows'][ds][k]) for ds in DS for k in SHARES if r['rows'][ds][k] is not None)
diff = max(diff, max(abs(o['headline'][k] - r['headline'][k]) for k in SHARES if r['headline'][k] is not None))
checks['oracle:r2_mix_v2 L=0 vs reactive:r2_mix_v2 L=0, every share'] = {'max_abs_diff': diff, 'ok': diff <= 1e-9}

# keep rule for r0 (same conditions as tools/keep_rt_v1.py)
c, rr = runs['rt:r0_predict_live_v1 L=0.1'], runs['reactive:r2_mix_v2 L=0.1']
c2, r2 = runs['rt:r0_predict_live_v1 L=0.1 seed2'], runs['reactive:r2_mix_v2 L=0.1 seed2']
keep = {'headline rt_nd >= reactive + 0.005': c['headline']['rt_nd'] >= rr['headline']['rt_nd'] + 0.005,
        'headline harsh <= reactive': c['headline']['harsh'] <= rr['headline']['harsh']}
for ds in DS:
    keep[f'{ds} harsh <= reactive + 0.005'] = c['rows'][ds]['harsh'] <= rr['rows'][ds]['harsh'] + 0.005
for ds in DS:
    keep[f'{ds} rt_nd >= reactive - 0.01'] = c['rows'][ds]['rt_nd'] >= rr['rows'][ds]['rt_nd'] - 0.01
keep['seed2 headline delta >= +0.003'] = c2['headline']['rt_nd'] - r2['headline']['rt_nd'] >= 0.003
keep['mean us per decision < 2000'] = c['headline']['us_mean'] < 2000
keep_out = {'conditions': keep, 'kept': all(keep.values()),
            'deltas': {'headline_rt_nd': c['headline']['rt_nd'] - rr['headline']['rt_nd'],
                       'headline_harsh': c['headline']['harsh'] - rr['headline']['harsh'],
                       'seed2_headline_rt_nd': c2['headline']['rt_nd'] - r2['headline']['rt_nd'],
                       'rt_nd': {ds: c['rows'][ds]['rt_nd'] - rr['rows'][ds]['rt_nd'] for ds in DS},
                       'harsh': {ds: c['rows'][ds]['harsh'] - rr['rows'][ds]['harsh'] for ds in DS}}}

diag = {}
for spec, name in (('reactive:r2_mix_v2', 'attacks_reactive_r2_mix_v2_L0.1.json'),
                   ('rt:r0_predict_live_v1', 'attacks_rt_r0_predict_live_v1_L0.1.json')):
    d = load(os.path.join(RUN, 'diag', name))
    assert d['spec'] == spec and d['bench_sha256'] == SHA
    diag[spec] = {'file': 'runs_mini/v5rt_1101/diag/' + name, 'headline': d['headline'], 'rows': d['rows'],
                  'definitions': d['definitions'], 'W_beats': d['W_beats']}


def f(x):
    return '  n/a ' if x is None else f'{x:.4f}'


hdr = (f"{'run':38s} {'rt_nd':>6s} {'harsh':>6s} {'nd_1st':>6s} {'hr_1st':>6s} {'ontime':>6s} {'change':>6s} "
       f"{'late':>6s} | {'bach':>6s} {'folk':>6s} {'pop':>6s} {'pop909':>6s} | us mean")
lines = [hdr]
for lab, e in list(runs.items()) + list(sanity.items()):
    h = e['headline']
    lines.append(f"{lab:38s} {f(h['rt_nd'])} {f(h['harsh'])} {f(h['nd_first'])} {f(h['harsh_first'])} "
                 f"{f(h['on_time_right'])} {f(h['changes'])} {f(h['late_entry'])} | "
                 + ' '.join(f(e['rows'][ds]['rt_nd']) for ds in DS) + f" | {h['us_mean']:.0f}")
table = '\n'.join(lines)

out = {'run': 'v5rt_1101', 'phase': 1, 'written': time.strftime('%Y-%m-%d %H:%M:%S'),
       'split': 'val only (tuned on val; this bundle has no test split)',
       'bench': 'futures/v5_rt/bench_rt.py', 'bench_sha256': SHA,
       'metrics': 'see the bench_rt.py docstring (SPEC + how it is read); shares are of counted time (her singing '
                  'from beat 4 on labelled beats) or of her notes (onset >= 4, labelled onset beat); headline = mean '
                  'of the 4 datasets; us = microseconds per decision call (plan and hear), pooled',
       'runs': runs, 'sanity_L0': sanity, 'sanity_checks': checks, 'keep_rule_r0_predict_live_v1': keep_out,
       'diagnostic_attacks_L0.1': diag, 'table': table,
       'notes': ['oracle needs onset == plan point exactly: it answers 1070/1073 bach, 14114/17326 folk, 12374/14388 pop '
                 'but only 545/13523 pop909 notes early (pop909 onsets sit 0.0002-0.0003 beat after the grid), so it '
                 'is a weak ceiling on pop909',
                 'the harsh metric counts a held-over dissonance (reactive) the same as a freshly attacked wrong note '
                 '(look-ahead): see diagnostic_attacks_L0.1']}
path = os.path.join(RUN, 'baselines.json')
with open(path, 'x') as fh:
    json.dump(out, fh, indent=1)
print(table)
print('sanity:', {k: ('OK' if v['ok'] else 'MISMATCH') + f" ({v['max_abs_diff']:.1e})" for k, v in checks.items()})
print('keep r0:', 'KEPT' if keep_out['kept'] else 'NOT KEPT', [k for k, v in keep.items() if not v])
print('wrote', path)
