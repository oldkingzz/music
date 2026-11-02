"""One-off faithfulness check for r2_mix (v4 round 2, implementer B), on train songs only: the mix at a = 0 must sing exactly
m1_filter_v5lo's notes, at a = 1 exactly r1_gru_chooser_v1's, and the 3-GRU ensemble must advance exactly once per note."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ABS = Path('/Users/san/claude_sandbox/music/duet_v0')
sys.path.insert(0, str(ABS / 'futures' / 'v3_nolatency'))
import bench3  # noqa: E402
from systems import m1_filter_v5lo, r1_gru_chooser_v1, r2_mix_v1  # noqa: E402

OUT = ABS / 'runs_mini' / 'v4_1101' / 'r2' / 'B' / 'check_mix_v1.json'
N = 25   # the first N train songs of every dataset


def recorder(obj, method):
    """Shadow obj.belief on the instance so every belief the chooser uses is recorded (the class is untouched)."""
    log = []

    def rec(k):
        b = np.asarray(method(obj, k), dtype=np.float64)
        log.append(b / b.sum())
        return method(obj, k)
    obj.belief = rec
    return log


if OUT.exists():
    sys.exit(f'refusing to overwrite {OUT}')
t0 = time.time()
train = bench3.train_sets()
m1 = m1_filter_v5lo.build(train)
r1 = r1_gru_chooser_v1.build(train)
mix = r2_mix_v1.build(train)
calls = [0]
_orig = mix.ens.chord_probs


def counted(onset, midi, prev_dur):
    calls[0] += 1
    return _orig(onset, midi, prev_dur)


mix.ens.chord_probs = counted
L_m1, L_r1, L_mix = recorder(m1, m1_filter_v5lo.Filter.belief), recorder(r1, r1_gru_chooser_v1.G.belief), \
    recorder(mix, r2_mix_v1.Mix.belief)
res = {'alpha_file': r2_mix_v1.ALPHA, 'songs_per_dataset': N}
for ds, songs in train.items():
    r = dict(songs=0, notes=0, a0_same_notes_as_m1=0, a1_same_notes_as_r1=0, file_alpha_diff_vs_m1=0,
             file_alpha_diff_vs_r1=0, gru_calls=0, beliefs_a0=0, beliefs_a1=0, max_belief_diff_a0_vs_m1=0.0,
             max_belief_diff_a1_vs_r1=0.0)
    for s in songs[:N]:
        del L_m1[:], L_r1[:], L_mix[:]
        o_m1, o_r1 = bench3.run_song(m1, s), bench3.run_song(r1, s)
        mix.a = 0.0
        o_a0 = bench3.run_song(mix, s)
        b_a0 = list(L_mix)
        del L_mix[:]
        mix.a = 1.0
        o_a1 = bench3.run_song(mix, s)
        b_a1 = list(L_mix)
        mix.a = r2_mix_v1.ALPHA
        c0 = calls[0]
        o_mx = bench3.run_song(mix, s)
        n = len(o_m1)
        r['songs'] += 1
        r['notes'] += n
        r['gru_calls'] += calls[0] - c0
        r['a0_same_notes_as_m1'] += sum(a == b for a, b in zip(o_a0, o_m1))
        r['a1_same_notes_as_r1'] += sum(a == b for a, b in zip(o_a1, o_r1))
        r['file_alpha_diff_vs_m1'] += sum(a != b for a, b in zip(o_mx, o_m1))
        r['file_alpha_diff_vs_r1'] += sum(a != b for a, b in zip(o_mx, o_r1))
        r['beliefs_a0'] += len(b_a0)
        r['beliefs_a1'] += len(b_a1)
        if b_a0:
            r['max_belief_diff_a0_vs_m1'] = max(r['max_belief_diff_a0_vs_m1'],
                                                float(np.abs(np.array(b_a0) - np.array(L_m1)).max()))
        if b_a1:
            r['max_belief_diff_a1_vs_r1'] = max(r['max_belief_diff_a1_vs_r1'],
                                                float(np.abs(np.array(b_a1) - np.array(L_r1)).max()))
    res[ds] = r
res['all_ok'] = all(v['a0_same_notes_as_m1'] == v['notes'] and v['a1_same_notes_as_r1'] == v['notes']
                    and v['gru_calls'] == v['notes'] and v['beliefs_a0'] == v['notes']
                    for k, v in res.items() if isinstance(v, dict))
res['secs'] = round(time.time() - t0, 1)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
