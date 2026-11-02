"""Phase D report writer (v4_1101): read-only derivation of the numbers quoted in report_drafts/*.md.

Reads only existing result files (listed in SRC below), computes differences / means of 4 / shares, prints them and
writes report_drafts/tools/derived_numbers_v1.json (refuses if it exists). Edits nothing; no bundle data, no systems.
val only, tuned on val (this bundle has no test split).
"""
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v4_1101')
OUT = os.path.join(RUN, 'report_drafts', 'tools', 'derived_numbers_v1.json')
DS = ('bach', 'folk', 'pop', 'pop909')


def jl(*p):
    with open(os.path.join(*p)) as fh:
        return json.load(fh)


def nd(name):
    if name == 'm4_oracle':
        return jl(ABS, 'futures', 'v3_nolatency', 'results', 'nd_m4_oracle.json')
    return jl(RUN, 'nd', name, 'results', f'nd_{name}.json')


def r4(x):
    return round(x, 4)


if os.path.exists(OUT):
    sys.exit(f'refusing: {OUT} exists')

out = {'what': 'derived numbers for report_drafts (differences, means of 4, shares); val only, tuned on val',
       'src': {}}

# 1. music_nd per dataset and differences
names = ['m1_filter_v5lo', 'r2_mix_v2', 'r3_style_mix_v3', 'third_below', 'm3_seq_v5', 'm4_oracle']
N = {n: nd(n) for n in names}
out['music_nd'] = {n: {**{d: r4(N[n][d]['music_nd']) for d in DS}, 'headline': r4(N[n]['headline_nd'])} for n in names}
out['double'] = {n: {d: r4(N[n][d]['double']) for d in DS} for n in names}
out['double_max'] = {n: r4(max(N[n][d]['double'] for d in DS)) for n in names}


def diff(a, b):
    r = {d: r4(N[a][d]['music_nd'] - N[b][d]['music_nd']) for d in DS}
    r['headline'] = r4(N[a]['headline_nd'] - N[b]['headline_nd'])
    return r


out['diff'] = {'r2_mix_v2-m1': diff('r2_mix_v2', 'm1_filter_v5lo'),
               'r3_style_mix_v3-m1': diff('r3_style_mix_v3', 'm1_filter_v5lo'),
               'r3_style_mix_v3-r2_mix_v2': diff('r3_style_mix_v3', 'r2_mix_v2'),
               'm4_oracle-m1': diff('m4_oracle', 'm1_filter_v5lo'),
               'm4_oracle-r2_mix_v2': diff('m4_oracle', 'r2_mix_v2')}
out['src']['music_nd'] = 'runs_mini/v4_1101/nd/<name>/results/nd_<name>.json; m4_oracle: futures/v3_nolatency/results/nd_m4_oracle.json'

# 2. seed 2
S2 = {n: jl(RUN, 'seed2', f'{n}.json') for n in ('m1_filter_v5lo', 'r2_mix_v2', 'r3_style_mix_v3')}
out['seed2'] = {n: {**{d: r4(S2[n][d]['music_nd']) for d in DS}, 'headline': round(S2[n]['headline_nd'], 6)} for n in S2}
out['seed2_diff'] = {'r2_mix_v2-m1': r4(S2['r2_mix_v2']['headline_nd'] - S2['m1_filter_v5lo']['headline_nd']),
                     'r3_style_mix_v3-r2_mix_v2': r4(S2['r3_style_mix_v3']['headline_nd'] - S2['r2_mix_v2']['headline_nd']),
                     'r3_style_mix_v3-m1': r4(S2['r3_style_mix_v3']['headline_nd'] - S2['m1_filter_v5lo']['headline_nd'])}
out['src']['seed2'] = 'runs_mini/v4_1101/seed2/<name>.json'

# 3. bench3: LAST val row per system in the leaderboard
last = {}
with open(os.path.join(ABS, 'futures', 'v3_nolatency', 'leaderboard_v3.jsonl')) as fh:
    for i, line in enumerate(fh, 1):
        row = json.loads(line)
        if row.get('split') == 'val':
            last[row['system']] = (i, row)
b3 = {}
for n in ('m1_filter_v5lo', 'r2_mix_v2', 'r3_style_mix_v3'):
    i, row = last[n]
    b3[n] = {'line': i, 'headline_music_raw': r4(row['headline']), 'flags': row['flags'],
             'big': {d: r4(row['rows'][d]['big']) for d in DS}, 'par': {d: r4(row['rows'][d]['par']) for d in DS},
             'us_per_note_max': round(max(row['rows'][d]['us_per_note'] for d in DS), 1)}
out['bench3_last_val_row'] = b3
out['src']['bench3_last_val_row'] = 'futures/v3_nolatency/leaderboard_v3.jsonl (last val row per system; line numbers given)'

# 4. diag_rt_v1: same chooser, different chord knowledge; miss buckets; chord accuracy; lock-on curve
D = jl(RUN, 'diag', 'diag_rt_v1.json')
H = D['headline_mean_of_4']
out['same_chooser_music_nd'] = {k: r4(v) for k, v in H['music_nd'].items()}
out['m1_miss_share_mean4'] = {k: r4(v) for k, v in H['miss_share']['rt'].items()}
out['m1_miss_share_sum_plus_nd'] = r4(sum(H['miss_share']['rt'].values()) + H['music_nd']['rt'])
out['told_onset_miss_share_mean4'] = {k: r4(v) for k, v in H['miss_share']['told'].items()}
out['no_good_below_but_above_ok_mean4'] = r4(H['no_good_below_but_above_ok_share'])
acc = H['chord_acc']
out['top1_acc_mean4'] = {g: r4(acc[g]['all']['acc']) for g in ('rt', 'gru', 'smooth', 'past')}
out['top1_acc_by_dataset'] = {g: {d: r4(D['datasets'][d]['chord_acc'][g]['all']['acc']) for d in DS} for g in ('rt', 'gru', 'smooth', 'past')}
out['lockon_mean4'] = {g: [r4(acc[g][f'note_{j}_in_chord']['acc']) for j in (1, 2, 3, 4)] for g in ('rt', 'gru', 'smooth', 'past')}
out['acc_changed_vs_same_mean4'] = {g: {'chord_changed': r4(acc[g]['chord_changed']['acc']), 'chord_same': r4(acc[g]['chord_same']['acc'])}
                                    for g in ('rt', 'gru', 'past')}
out['acc_misheard_vs_heard_right_mean4'] = {g: {'heard_right': r4(acc[g]['heard_right']['acc']), 'misheard': r4(acc[g]['misheard']['acc'])}
                                            for g in ('rt', 'gru')}
out['src']['diag_rt'] = 'runs_mini/v4_1101/diag/diag_rt_v1.json (headline = mean of 4; told = told the onset chord, past = told the previous beat chord, smooth = offline HMM)'

# 5. champions' chord accuracy, lock-on, flatness (diag_champs_v1 table_4dp)
C = jl(RUN, 'diag', 'diag_champs_v1.json')['table_4dp']
out['champs_table_4dp'] = {n: {k: C[n]['headline'][k] for k in ('music_nd', 'acc', 'n1', 'n2', 'n3', 'n4+', 'p_top', 'wrong_chord_guess', 'double')}
                           for n in C}
out['champs_acc_by_dataset'] = {n: {d: C[n][d]['acc'] for d in DS} for n in C}
out['src']['champs'] = 'runs_mini/v4_1101/diag/diag_champs_v1.json table_4dp'

# 6. always I, clean melody (recheck extra)
X = jl(RUN, 'diag', 'recheck', 'recheck_rt_extra_result_20261101_145317.json')
out['always_I_acc_mean4'] = r4(X['mean']['share_I'])
out['always_I_by_dataset'] = {d: r4(X['per_ds'][d]['share_I']) for d in DS}
out['m1_clean_melody'] = {'music_nd_mean4': r4(X['mean']['music_nd_clean_melody']), 'top1_acc_mean4': r4(X['mean']['rt_acc_clean_melody'])}
R = jl(RUN, 'diag', 'recheck', 'recheck_rt_result_20261101_145137.json')
out['held_note_crosses_chord_change_time_share'] = {d: r4(R['per_ds'][d]['scored_time_onset_chord_differs']) for d in DS}
out['src']['recheck'] = 'runs_mini/v4_1101/diag/recheck/recheck_rt_extra_result_20261101_145317.json, recheck_rt_result_20261101_145137.json'

# 7. why_d: is there anything to lock on to?
W = jl(RUN, 'diag', 'why', 'why_d_v1.json')['datasets']
out['why_d'] = {'unique_best_all_spans': {d: r4(W[d]['all_spans']['unique_best']) for d in DS},
                'mean_triads_tied_all_spans': {d: round(W[d]['all_spans']['mean_tied'], 2) for d in DS},
                'true_among_best_all_spans': {d: r4(W[d]['all_spans']['among_best']) for d in DS},
                'perfect_lockon_ceiling_mean4_by_note_index': [r4(sum(W[d]['perfect_lockon_ceiling_acc_by_note_index'][j] for d in DS) / 4)
                                                               for j in range(4)],
                'm1_argmax_flip_rate_within_span': {d: r4(W[d]['argmax_flip_rate_within_span']['m1']) for d in DS}}
out['src']['why_d'] = 'runs_mini/v4_1101/diag/why/why_d_v1.json (perfect lock-on ceiling: mean over the 4 datasets computed here)'

# 8. why_b: m1 guesses I too often
B = jl(RUN, 'diag', 'why', 'why_b_v1.json')['confusions_m1_rt']
out['why_b_I_share'] = {d: {'guess_I': B[d]['guess_share'][0], 'true_I': B[d]['true_chord_share'][0]} for d in DS}
out['src']['why_b'] = 'runs_mini/v4_1101/diag/why/why_b_v1.json confusions_m1_rt (chord order I ii iii IV V vi vii)'

# 9. transition / prior tweaks on m1 (why_a, why_b, why_e, why_f), real-time headline music_nd
A = jl(RUN, 'diag', 'why', 'why_a_v1.json')['variants']
Bm = jl(RUN, 'diag', 'why', 'why_b_v1.json')['mixtures_rt']
E = jl(RUN, 'diag', 'why', 'why_e_v1.json')['variants']
F = jl(RUN, 'diag', 'why', 'why_f_v1.json')['variants']
tw = {f'why_a {k}': r4(A[k]['rt']['head']['music_nd']) for k in ('T_bal0', 'T_sticky70', 'T_sticky85', 'T_sticky95', 'T_dsA_oracle', 'E_lam1.0')}
tw.update({f'why_b {k}': r4(Bm[k]['head']['music_nd']) for k in ('MIX4_tables_lam1.5', 'MIX4_tables_lam1.0')})
tw.update({f'why_e {k}': r4(E[k]['rt']['head']['music_nd']) for k in ('DUR_pooled_lam1.5', 'DUR_pooled_lam1.0', 'DUR_ORACLE_ds_lam1.5')})
tw.update({f'why_f {k}': r4(F[k]['rt']['head']['music_nd']) for k in ('POP3_lam1.0_pitch0',)})
out['m1_tweaks_rt_music_nd'] = tw
out['src']['tweaks'] = 'runs_mini/v4_1101/diag/why/why_a_v1.json, why_b_v1.json, why_e_v1.json, why_f_v1.json (rt head music_nd)'

# 10. r3/C: belief flatness and ablation
RC = jl(RUN, 'r3', 'C', 'diag_20261101_153920.json')
out['r3C_mean4'] = {k: {'top1_acc': v['mean4']['top1_acc'], 'mean_top_prob': v['mean4']['mean_top_prob']} for k, v in RC.items() if isinstance(v, dict)}
AB = jl(RUN, 'r3', 'C', 'ablate_20261101_154122.json')['configs']
out['r3C_ablation_headline'] = {k: v['headline_nd'] for k, v in AB.items()}
out['src']['r3C'] = 'runs_mini/v4_1101/r3/C/diag_20261101_153920.json, ablate_20261101_154122.json'

# 11. r4/A: held-note boundary calibration (own pooled table = implied_change[0])
CA = jl(RUN, 'r4', 'A', 'calib_20261101_160528.json')['calibration_val']
out['r4A_calibration_val'] = {d: {'held_actual': r4(CA[d]['held']['actual_change']), 'held_implied_pooled': r4(CA[d]['held']['implied_change'][0]),
                                  'all_actual': r4(CA[d]['all']['actual_change']), 'all_implied_pooled': r4(CA[d]['all']['implied_change'][0]),
                                  'held_boundaries': CA[d]['held']['boundaries'], 'all_boundaries': CA[d]['all']['boundaries']} for d in DS}
out['src']['r4A'] = 'runs_mini/v4_1101/r4/A/calib_20261101_160528.json calibration_val (implied_change[0] = own_pooled table)'

# 12. r4/B: where the cost of mishearing goes (champion r3_style_mix_v3)
WB = jl(RUN, 'r4', 'B', 'check_why_20261101_160400.json')['rows']
out['r4B_mishearing'] = {d: {'cost_total': WB[d]['cost_total'], 'cost_belief': WB[d]['cost_belief'], 'cost_choice': WB[d]['cost_choice'],
                             'belief_share': round(WB[d]['cost_belief'] / WB[d]['cost_total'], 3),
                             'misheard_in_scale_share': round(WB[d]['by_note']['misheard|in_scale']['notes'] /
                                                              (WB[d]['by_note']['misheard|in_scale']['notes'] + WB[d]['by_note']['misheard|out_of_scale']['notes']), 3)}
                         for d in DS}
out['src']['r4B'] = 'runs_mini/v4_1101/r4/B/check_why_20261101_160400.json'
BL = jl(RUN, 'r4', 'B', 'check_belief_20261101_160733.json')['rows']
out['r4B_top1_true_notes_vs_heard'] = {'heard_mean4': r4(sum(BL['champion_copy'][d]['all']['top1'] for d in DS) / 4),
                                       'true_notes_mean4': r4(sum(BL['oracle_true_notes'][d]['all']['top1'] for d in DS) / 4)}
out['src']['r4B_belief'] = 'runs_mini/v4_1101/r4/B/check_belief_20261101_160733.json (mean of 4 computed here)'

# 13. r3/A: where her note is (time share of later beats where the chord changed under a held note)
WA = jl(RUN, 'r3', 'A', 'where_20261101_154310.json')['systems']['r2_mix_v2']
out['r3A_time_share_later_chord_changed'] = {d: r4(WA[d]['time_share']['later_chord_changed']) for d in DS}
out['src']['r3A'] = 'runs_mini/v4_1101/r3/A/where_20261101_154310.json'

# 14. audio clip context
CC = jl(RUN, 'phaseD', 'A', 'clip_context.json')['summary']
out['clip_context'] = {'song_level': CC['song_level'], 'clip_level': CC['clip_level']}
out['src']['clip_context'] = 'runs_mini/v4_1101/phaseD/A/clip_context.json (champion = r2_mix_v2, 17 Bach val songs)'

with open(OUT, 'x') as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
print('wrote', OUT)
