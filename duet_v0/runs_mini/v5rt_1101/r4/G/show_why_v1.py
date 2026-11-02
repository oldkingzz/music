"""show_why_v1 (run v5rt_1101, round 4, idea G): what changed in the live behaviour at the large-lag cells, read only from
r4/G/table_G_v1.json (engine decision counts of the scored runs) and dev_log_G_v1.jsonl (the systems' own decision stats
for the same configs; the dev runs equal the scored files). Val only, tuned on val (no test split).
"""
import json

RUN = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101'
T = json.load(open(RUN + '/r4/G/table_G_v1.json'))
for cell in ('sung@bt_cal', 'sung_low@bt_cal', 'sung@micbt_cal'):
    w = T['W']['decision_counts_bt_cal'][cell]
    print(cell, 'W', {k: v for k, v in w.items() if v})
    for n, r in T['rows'].items():
        d = r['cells'][cell]['decision_counts']
        print('   ', n, {k: v - w.get(k, 0) for k, v in d.items() if v - w.get(k, 0)})
KEYS = ('hear_switch', 'hear_same', 'hedge_moved', 'hear_kept_plan', 'guard_keep', 'trk_keep', 'trk_sil', 'trk_react',
        'trk_attack', 'doubt_noatt', 'plan_attack', 'g_large', 'g_late_pred', 'g_late_keep', 'g_late_keep_harsh',
        'g_late_attack', 'g_ehedge')
want = {'W': 'W', 'ln2_add0_mu0': 'r4_G_hedge_v1', 'e1_f25': 'r4_G_ehedge_v1', 'e1_c3_ln3': 'r4_G_ehedgesafe_v1'}
seen = set()
with open(RUN + '/r4/G/dev_log_G_v1.jsonl') as fh:
    for line in fh:
        r = json.loads(line)
        tag = r['spec'].get('tag')
        if tag not in want or tag in seen or 'sung@bt_cal' not in r['cells']:
            continue
        seen.add(tag)
        for cell, v in r['cells'].items():
            st = v['stats']
            print(f"{want[tag]:20s} {cell:16s} rt {v['m']['rt_nd_live']:.4f} " + ' '.join(f'{k}={st.get(k, 0)}' for k in KEYS))
