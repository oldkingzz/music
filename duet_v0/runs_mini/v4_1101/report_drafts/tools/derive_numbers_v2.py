"""Phase D report writer (v4_1101), part 2: per-idea numbers for report_drafts/*.md (read-only).

Reads only existing result files: nd/<name>/results/nd_<name>.json, the LAST val row per system in
futures/v3_nolatency/leaderboard_v3.jsonl, and diag/diag_rt_v1.json. Writes report_drafts/tools/derived_numbers_v2.json
(refuses if it exists). val only, tuned on val (this bundle has no test split).
"""
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v4_1101')
OUT = os.path.join(RUN, 'report_drafts', 'tools', 'derived_numbers_v2.json')
DS = ('bach', 'folk', 'pop', 'pop909')

if os.path.exists(OUT):
    sys.exit(f'refusing: {OUT} exists')


def nd(name):
    with open(os.path.join(RUN, 'nd', name, 'results', f'nd_{name}.json')) as fh:
        return json.load(fh)


last = {}
with open(os.path.join(ABS, 'futures', 'v3_nolatency', 'leaderboard_v3.jsonl')) as fh:
    for i, line in enumerate(fh, 1):
        row = json.loads(line)
        if row.get('split') == 'val':
            last[row['system']] = (i, row)

IDEAS = {
    'E1': ['r1_gru_chooser_v1', 'r1_gru_chooser_v2', 'r2_gru_chooser_v1', 'r2_gru_chooser_v2', 'r2_gru_chooser_v3'],
    'E2': ['r2_mix_v1', 'r2_mix_v2', 'r2_mix_v3', 'r2_mix_v4'],
    'E3': ['r2_style_hmm_v1', 'r2_style_hmm_v2', 'r2_style_hmm_v3', 'r2_style_hmm_v4'],
    'E2xE3': ['r3_style_mix_v1', 'r3_style_mix_v2', 'r3_style_mix_v3', 'r3_style_mix_v4'],
    'E4_R3': ['r3_cover_fit_v1', 'r3_cover_fit_v2', 'r3_cover_fit_v3', 'r3_cover_fit_v4'],
    'E5_R3': ['r3_true_cons_v1', 'r3_true_cons_v2', 'r3_true_cons_v3', 'r3_true_cons_v4'],
    'E4_R4': ['r4_cover_fit_v1'],
    'E5_R4': ['r4_true_cons_v1', 'r4_true_cons_v2'],
}
REF = {'E1': 'm1_filter_v5lo', 'E2': 'm1_filter_v5lo', 'E3': 'm1_filter_v5lo', 'E2xE3': 'r2_mix_v2',
       'E4_R3': 'r2_mix_v2', 'E5_R3': 'r2_mix_v2', 'E4_R4': 'r3_style_mix_v3', 'E5_R4': 'r3_style_mix_v3'}

out = {'what': 'per-idea headline music_nd, difference vs the champion of that round, folk big; val only, tuned on val',
       'src': 'runs_mini/v4_1101/nd/<name>/results/nd_<name>.json; big = last val row per system in futures/v3_nolatency/leaderboard_v3.jsonl; '
              'diag/diag_rt_v1.json',
       'ideas': {}}
count = 0
for idea, names in IDEAS.items():
    ref = nd(REF[idea])
    rows = {}
    for n in names:
        r = nd(n)
        line, b3 = last[n]
        rows[n] = {'headline_nd': round(r['headline_nd'], 6),
                   'delta_vs_ref': round(r['headline_nd'] - ref['headline_nd'], 4),
                   'delta_by_dataset': {d: round(r[d]['music_nd'] - ref[d]['music_nd'], 4) for d in DS},
                   'big_folk': round(b3['rows']['folk']['big'], 4), 'leaderboard_line': line}
        count += 1
    out['ideas'][idea] = {'reference': REF[idea], 'reference_headline_nd': round(ref['headline_nd'], 6), 'variants': rows}
out['scored_candidates_counted_here'] = count

m1_big_folk = last['m1_filter_v5lo'][1]['rows']['folk']['big']
out['keep_rule_folk_big_limit_vs_m1'] = round(m1_big_folk + 0.01, 4)
out['m1_big_folk'] = round(m1_big_folk, 4)
r2_big_folk = last['r2_mix_v2'][1]['rows']['folk']['big']
out['keep_rule_folk_big_limit_vs_r2_mix_v2'] = round(r2_big_folk + 0.01, 4)

with open(os.path.join(RUN, 'diag', 'diag_rt_v1.json')) as fh:
    H = json.load(fh)['headline_mean_of_4']['music_nd']
gap = H['told'] - H['rt']
out['gap_told_minus_rt'] = round(gap, 4)
for n in ('r2_mix_v2', 'r3_style_mix_v3'):
    out[f'share_of_gap_closed_{n}'] = round((nd(n)['headline_nd'] - H['rt']) / gap, 3)
out['remaining_gap_told_minus'] = {n: round(H['told'] - nd(n)['headline_nd'], 4) for n in ('r2_mix_v2', 'r3_style_mix_v3')}

with open(OUT, 'x') as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
print('wrote', OUT)
