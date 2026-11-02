"""One-off reproduction check for r2_mix_v4 (v4 round 2, implementer B), on train songs only: its own chooser copy with w_big
set back to 0.4 must sing exactly r2_mix_v3's notes, and with a = 0 as well exactly m1_filter_v5lo's notes."""
import json
import sys
import time
from pathlib import Path

ABS = Path('/Users/san/claude_sandbox/music/duet_v0')
sys.path.insert(0, str(ABS / 'futures' / 'v3_nolatency'))
import bench3  # noqa: E402
from systems import m1_filter_v5lo, r2_mix_v3, r2_mix_v4  # noqa: E402

OUT = ABS / 'runs_mini' / 'v4_1101' / 'r2' / 'B' / 'check_v4.json'
N = 100   # the first N train songs of every dataset

if OUT.exists():
    sys.exit(f'refusing to overwrite {OUT}')
t0 = time.time()
train = bench3.train_sets()
m1 = m1_filter_v5lo.build(train)
v3 = r2_mix_v3.build(train)
v4 = r2_mix_v4.build(train)
res = {'w_big_file': r2_mix_v4.W_BIG, 'alpha_v3': r2_mix_v3.ALPHA, 'alpha_v4': r2_mix_v4.ALPHA, 'songs_per_dataset': N}
for ds, songs in train.items():
    r = dict(songs=0, notes=0, wbig04_same_as_v3=0, wbig04_a0_same_as_m1=0, file_v4_diff_vs_v3=0)
    for s in songs[:N]:
        o3, om1 = bench3.run_song(v3, s), bench3.run_song(m1, s)
        v4.w_big = 0.4
        o4 = bench3.run_song(v4, s)
        v4.a = 0.0
        o4a0 = bench3.run_song(v4, s)
        v4.a, v4.w_big = r2_mix_v4.ALPHA, r2_mix_v4.W_BIG
        o4f = bench3.run_song(v4, s)
        r['songs'] += 1
        r['notes'] += len(o3)
        r['wbig04_same_as_v3'] += sum(a == b for a, b in zip(o4, o3))
        r['wbig04_a0_same_as_m1'] += sum(a == b for a, b in zip(o4a0, om1))
        r['file_v4_diff_vs_v3'] += sum(a != b for a, b in zip(o4f, o3))
    res[ds] = r
res['all_ok'] = all(v['wbig04_same_as_v3'] == v['notes'] and v['wbig04_a0_same_as_m1'] == v['notes']
                    for v in res.values() if isinstance(v, dict))
res['secs'] = round(time.time() - t0, 1)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
