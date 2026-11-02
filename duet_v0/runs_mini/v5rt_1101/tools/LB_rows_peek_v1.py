"""Print the baseline rows of a live-bench selection JSON compactly as JSON lines (read-only).

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/tools/LB_rows_peek_v1.py runs_mini/v5rt_1101/live_bench/L_select_R3p1_v1.json
"""
import json
import sys

r = json.load(open(sys.argv[1]))
keys = ('engine', 'voice', 'lat', 'file', 'rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100', 'on_time_right',
        'wrong_att_per100', 'fresh_heard', 'harsh_heard')
for row in r['rows']:
    out = {}
    for k in keys:
        if k in row:
            v = row[k]
            out[k] = round(v, 4) if isinstance(v, float) else v
    print(json.dumps(out))
print(json.dumps(r['reactive']))
print(json.dumps({s: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x['deltas_vs_reactive'].items()}
                  for s, x in r['selection'].items()}))
