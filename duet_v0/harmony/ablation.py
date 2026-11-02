"""Feature-set ablation for the alto CRF, scored on the VALIDATION split only.

    .venv/bin/python harmony/ablation.py      -> harmony/params/ablation_alto.json

The winning feature set / sigma is then retrained by train.py --features <name>.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import load_split                                     # noqa: E402
from model import VOICE_RANGE, HarmonyModel, Templates             # noqa: E402
from train import sequences, soprano_median, train_crf, val_accuracy  # noqa: E402

U, P = Templates.UNARY, Templates.PAIR
FEATURE_SETS = {
    'hmm-like (2 templates)': ['iv|mode,deg', 'dh|ds'],
    'no lookahead': [f for f in Templates.DEFAULT if f != 'ivpc|mode,deg,nextdeg'],
    'no phrase flags': [f for f in Templates.DEFAULT if f not in ('ivpc|mode,end,deg', 'ivpc|mode,start,deg')],
    'unary only': list(Templates.DEFAULT[:6]),
    'default (11)': list(Templates.DEFAULT),
    'default + prevdeg': list(Templates.DEFAULT) + ['ivpc|mode,prevdeg,deg'],
    'all (14)': list(U) + list(P),
}


def main(voice='alto', sigmas=(0.1, 0.3)):   # sigma=1.0 never beat 0.3 on the 3 sets tried (results/ablation_partial_sigma_grid.log)
    S = load_split()
    lo, hi = VOICE_RANGE[voice]
    med = soprano_median(S['train'])
    out = []
    for name, feats in FEATURE_SETS.items():
        tpl = Templates(lo, hi, feats)
        tr = sequences(S['train'], tpl, voice, med)
        va = sequences(S['val'], tpl, voice, med)
        for sigma in sigmas:
            t0 = time.time()
            w, res = train_crf(tr, tpl, sigma, maxiter=300)
            cfg = {'voice': voice, 'iv_range': [lo, hi], 'sop_median': med, 'features': feats,
                   'duet': {'lam': 0.0, 'no_parallel': False}}
            acc = val_accuracy(HarmonyModel(w, cfg), va)
            row = {'set': name, 'features': feats, 'sigma': sigma, 'W': tpl.W, 'val_acc': acc,
                   'train_nll_per_pos': float(res.fun) / sum(len(s['y']) for s in tr),
                   'iters': int(res.nit), 'seconds': time.time() - t0}
            print(f"{name:28s} sigma={sigma:<4} W={tpl.W:5d} val acc {acc:.3f}  ({row['seconds']:.0f}s)", flush=True)
            out.append(row)
    with open(os.path.join(HERE, 'params', f'ablation_{voice}.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    best = max(out, key=lambda r: r['val_acc'])
    print('best:', best['set'], best['sigma'], f"{best['val_acc']:.3f}")


if __name__ == '__main__':
    main()
