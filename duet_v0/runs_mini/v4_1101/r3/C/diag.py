"""One-off diagnostic for v4 round 3, implementer C (not a system, not scored): how good is each real-time chord belief?

At every heard note from beat 4 on whose beat has a chord label, the belief the chooser actually used is recorded for:
m1_filter_v5lo (pooled HMM), r2_style_hmm_v1 (style HMM), r1_gru_chooser_v1 (3-GRU), r2_mix_v2 (champion, pooled^0.3 GRU^0.7),
r3_style_mix_v3 (style^0.5 GRU^0.5), r3_style_mix_v1 (style^0.3 GRU^0.7). Per dataset: top-1 accuracy against the true
chord of that beat, mean log P(true chord), mean top probability (sharpness). Val only, tuned on val.
Writes runs_mini/v4_1101/r3/C/diag_<time>.json (mode 'x').
"""
import json
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402
from systems import m1_filter_v5lo, r1_gru_chooser_v1, r2_mix_v2, r2_style_hmm_v1  # noqa: E402
from systems import r3_style_mix_v1, r3_style_mix_v3  # noqa: E402

OUTDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'r3', 'C')
SYSTEMS = (('pooled_hmm m1_filter_v5lo', m1_filter_v5lo), ('style_hmm r2_style_hmm_v1', r2_style_hmm_v1),
           ('gru3 r1_gru_chooser_v1', r1_gru_chooser_v1), ('champion r2_mix_v2 (pooled, a 0.7)', r2_mix_v2),
           ('r3_style_mix_v3 (style, a 0.5)', r3_style_mix_v3), ('r3_style_mix_v1 (style, a 0.7)', r3_style_mix_v1))


def recorder(obj):
    method = type(obj).belief
    log = []

    def rec(k):
        b = method(obj, k)
        x = np.asarray(b, dtype=np.float64)
        log.append((k, x / x.sum()))
        return b
    obj.belief = rec
    return log


def main():
    t0 = time.time()
    train = bench3.train_sets()
    every = bench3.eval_sets('val')
    res = {'what': 'real-time chord belief quality at each heard note (beat >= 4, labelled beats), val, tuned on val'}
    for label, mod in SYSTEMS:
        system = mod.build(train)
        log = recorder(system)
        r = {}
        for ds, songs in every.items():
            n = hit = 0
            lp = mp = 0.0
            for s in songs:
                del log[:]
                bench3.run_song(system, s)
                z = [b['z'] for b in s['beats']]
                for k, b in log:
                    if k < bench3.J or k >= len(z) or z[k] < 0:
                        continue
                    n += 1
                    hit += int(np.argmax(b) == z[k])
                    lp += float(np.log(max(b[z[k]], 1e-300)))
                    mp += float(b.max())
            r[ds] = {'notes': n, 'top1_acc': round(hit / n, 4), 'mean_logp_true': round(lp / n, 4),
                     'mean_top_prob': round(mp / n, 4)}
        r['mean4'] = {k: round(sum(r[d][k] for d in every) / 4, 4) for k in ('top1_acc', 'mean_logp_true', 'mean_top_prob')}
        res[label] = r
        print(label, json.dumps(r), flush=True)
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"diag_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('secs', res['secs'], 'wrote', path)


if __name__ == '__main__':
    main()
