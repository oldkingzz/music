"""Train on the training chorales, tune on validation, score on the held-out test chorales.

Writes duet_beat/out/results.json. Run:  .venv/bin/python duet_beat/run_eval.py
Tuning picks, on validation only, the q_min (and gamma, and lambda for the 12-vector mode) that
maximise U = (in-chord notes - out-of-chord notes) per beat; the test set is scored once.
"""
import itertools
import json
import os
import sys
from dataclasses import asdict, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load  # noqa: E402
from metrics import bootstrap, paired_diff, song_counts, summarize  # noqa: E402
from model import ChordHMM  # noqa: E402
from planner import Config, run  # noqa: E402
from theory import CHORDS  # noqa: E402

OUT = os.path.join(HERE, 'out', 'results.json')
QS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def evaluate(songs, model, cfg, system):
    return [song_counts(s, run(s, model, cfg, system), cfg.J) for s in songs]


def tune(songs, model, base, grid):
    best = None
    rows = []
    for vals in itertools.product(*grid.values()):
        cfg = replace(base, **dict(zip(grid, vals)))
        sm = summarize(evaluate(songs, model, cfg, 'hmm'))
        rows.append({**dict(zip(grid, vals)), **{k: sm[k] for k in ('m1', 'm2', 'm5', 'u')}})
        if best is None or sm['u'] > best[0]:
            best = (sm['u'], cfg)
    return best[1], rows


def main():
    d = load()['split']
    model = ChordHMM().fit(d['train'])
    val, test = d['val'], d['test']

    # ---- tuning on validation
    main_cfg, grid_main = tune(val, model, Config(mode='long', N=4), {'q_min': QS, 'gamma': (0.0, 0.03, 0.1, 0.3)})
    vec_cfg, grid_vec = tune(val, model, Config(mode='vec', N=4), {'q_min': QS, 'gamma': (main_cfg.gamma,), 'lam': (0.5, 1.0, 2.0)})
    n1_cfg = replace(main_cfg, N=1)
    curve = [{'q_min': q, **{k: v for k, v in summarize(evaluate(val, model, replace(main_cfg, q_min=q), 'hmm')).items()
                              if k in ('m1', 'm2', 'm5', 'u')}} for q in QS]

    # ---- the test set, scored once
    systems = {
        'hmm_long_N4': (main_cfg, 'hmm', 'HMM, longest note, plan 4 beats (main)'),
        'hmm_vec_N4': (vec_cfg, 'hmm', 'HMM, 12-vector, plan 4 beats'),
        'hmm_long_N1': (n1_cfg, 'hmm', 'HMM, longest note, plan 1 beat'),
        'B0_tonic': (replace(main_cfg, q_min=0.0), 'tonic', 'B0 always chord I'),
        'B1_prior': (main_cfg, 'prior', 'B1 transitions only, never listens'),
        'B2_rule': (main_cfg, 'rule', "B2 repo rule: 3rd/6th below her last note"),
        'B3_peek': (main_cfg, 'peek', 'B3 sees the current beat (not causal, upper bound)'),
        'B4_oracle': (replace(main_cfg, q_min=0.0), 'oracle', "B4 told Bach's chord (upper bound)"),
    }
    res, counts = {}, {}
    for key, (cfg, system, label) in systems.items():
        cs = evaluate(test, model, cfg, system)
        counts[key] = cs
        res[key] = {'label': label, 'system': system, 'cfg': asdict(cfg), **summarize(cs), 'ci': bootstrap(cs)}
        print(f"{key:12s} M1 {res[key]['m1']:.3f}  M2 {res[key]['m2']:.3f}  M2b {res[key]['m2b']:.3f}  "
              f"M3 {res[key]['m3']:.3f}  M4 {res[key]['m4']:.2f}  M5 {res[key]['m5']:.3f}  U {res[key]['u']:.3f}", flush=True)

    diffs = {}
    for a, b, k in (('hmm_long_N4', 'B1_prior', 'm1'), ('hmm_long_N4', 'B0_tonic', 'm1'), ('hmm_long_N4', 'B0_tonic', 'm2'),
                    ('hmm_long_N4', 'B0_tonic', 'u'), ('hmm_long_N4', 'B2_rule', 'm2'),
                    ('hmm_long_N4', 'B2_rule', 'u'), ('hmm_long_N4', 'hmm_long_N1', 'm2'), ('hmm_long_N4', 'hmm_long_N1', 'u'),
                    ('hmm_vec_N4', 'hmm_long_N4', 'm1'), ('hmm_vec_N4', 'hmm_long_N4', 'u'),
                    ('B3_peek', 'hmm_long_N4', 'm1'), ('hmm_long_N4', 'B3_peek', 'u')):
        diffs[f'{a}-{b}:{k}'] = paired_diff(counts[a], counts[b], k)

    # ---- what the model learned, for the page
    learned = {
        'chords': list(CHORDS),
        'init': model.init.round(4).tolist(),
        'A_downbeat': model.A2[0].round(4).tolist(),
        'A_other': model.A2[1].round(4).tolist(),
        'A4': model.A4.round(4).tolist(),
        'B': model.B.round(4).tolist(),
        'label_share_train': (np.bincount([b['z'] for r in d['train'] for b in r['beats'] if b['z'] >= 0], minlength=7)
                              / sum(1 for r in d['train'] for b in r['beats'] if b['z'] >= 0)).round(4).tolist(),
        'label_fit_test': float(np.mean([b['fit'] for r in test for b in r['beats'] if b['z'] >= 0])),
    }
    data_info = {p: {'chorales': len(d[p]), 'beats': sum(r['K'] for r in d[p])} for p in ('train', 'val', 'test')}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump({'results': res, 'diffs': diffs, 'tuning': {'main': grid_main, 'vec': grid_vec, 'q_curve_val': curve,
                   'chosen': {'main': asdict(main_cfg), 'vec': asdict(vec_cfg)}}, 'learned': learned, 'data': data_info,
                   'per_song_main': {s['id']: summarize([c]) for s, c in zip(test, counts['hmm_long_N4'])}},
                  fh, indent=1, default=float)
    print('chosen main', asdict(main_cfg))
    print('chosen vec', asdict(vec_cfg))
    for k, v in diffs.items():
        print(f'{k:36s} {v[0]:+.3f}  [{v[1][0]:+.3f}, {v[1][1]:+.3f}]')


if __name__ == '__main__':
    main()
