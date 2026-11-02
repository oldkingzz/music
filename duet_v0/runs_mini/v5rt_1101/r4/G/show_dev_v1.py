"""show_dev_v1 (run v5rt_1101, round 4, idea G): print the decision stats of dev_log_G_v1.jsonl lines (read only).
Val only, tuned on val (no test split).

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r4/G/show_dev_v1.py [tag ...]
"""
import json
import sys

LOG = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/r4/G/dev_log_G_v1.jsonl'
KEYS = ('hear_switch', 'hear_enter', 'hear_same', 'hedge_moved', 'hear_kept_plan', 'guard_keep', 'guard_keep_sil',
        'trk_keep', 'trk_sil', 'trk_react', 'trk_attack', 'doubt_noatt', 'plan_attack', 'hop_blocked', 'g_large',
        'g_late_pred', 'g_late_keep', 'g_late_keep_harsh', 'g_late_sil', 'g_late_attack', 'g_follow_moved')

want = set(sys.argv[1:])
with open(LOG) as fh:
    for line in fh:
        r = json.loads(line)
        tag = r['spec'].get('tag', '')
        if want and tag not in want:
            continue
        for cell, v in r['cells'].items():
            st = v['stats']
            m = v['m']
            print(f"{r['time']} {tag:14s} {cell:16s} rt {m['rt_nd_live']:.4f} h {m['harsh']:.4f} fr {m['fresh_per100']:.2f} "
                  f"st {m['stumbles_per100']:.2f} otr {m['on_time_right']:.3f} wa {m['wrong_att_per100']:.1f} | "
                  + ' '.join(f"{k}={st.get(k, 0)}" for k in KEYS if st.get(k, 0)))
        if 'brt' in r:
            h = r['brt']['head']
            print(f"{r['time']} {tag:14s} brt L{r['brt']['L']} rt {h['rt_nd']:.4f} h {h['harsh']:.4f} fr {h['fresh']:.2f}")
