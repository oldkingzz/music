"""End-to-end check: does the learned harmony survive the real rehearsal front end?

Each TEST chorale soprano is sung by the synthetic singer (singer.render: glides, vibrato,
intonation error sigma = 0.1 semitone, breaths at fermatas), analysed with the causal YIN tracker
(follow.analyse) and segmented by follow.offline_notes - exactly the path a rehearsal recording
takes. Both planners then run on the *estimated* notes. Each plan entry is matched back to the
score note whose onset is nearest (<= 80 ms) to score it against Bach.

    .venv/bin/python harmony/pipeline_eval.py      -> harmony/results/pipeline_results.json

Limits: the singer is synthetic (same model used to tune offline_notes), so this measures
robustness to segmentation/pitch-tracking artefacts, not to real voices.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import singer                                                  # noqa: E402
from chorales import to_notes                                  # noqa: E402
from dataset import load_split                                 # noqa: E402
from follow import analyse, offline_notes, plan_harmony        # noqa: E402
from metrics import line_metrics, pool, rates                   # noqa: E402
from model import plan_harmony_learned                         # noqa: E402


def main(tol=0.08):
    S = load_split()
    t0 = time.time()
    out = {'n_chorales': len(S['test']), 'per_planner': {}, 'segmentation': {}}
    counts = {'rule': [], 'learned': []}
    seg = {'score_notes': 0, 'est_notes': 0, 'matched': 0, 'pitch_ok': 0}   # matched = score notes found
    for i, rec in enumerate(S['test']):
        score = to_notes(rec)
        x, _ = singer.render([(n['on'] + 0.5, n['off'] + 0.5, n['midi']) for n in score], seed=100 + i)
        est = offline_notes(analyse(x))
        on_s = np.array([n['on'] + 0.5 for n in score])
        seg['score_notes'] += len(score)
        seg['est_notes'] += len(est)
        hit = {}
        for n in est:
            j = int(np.argmin(np.abs(on_s - n['on'])))
            if abs(on_s[j] - n['on']) <= tol and j not in hit:
                hit[j] = n['midi'] == score[j]['midi']
        seg['matched'] += len(hit)
        seg['pitch_ok'] += int(sum(hit.values()))
        for name, fn in (('rule', plan_harmony), ('learned', plan_harmony_learned)):
            plan, _ = fn(est)
            s, h, a, ch = [], [], [], []
            used = set()
            for p in plan:
                j = int(np.argmin(np.abs(on_s - p['r'])))
                if abs(on_s[j] - p['r']) > tol or j in used:
                    continue                                   # spurious extra segment: not scored
                used.add(j)
                row = rec['notes'][j]
                s.append(row[2])                               # the TRUE melody note
                h.append(p['h'])                               # what the machine would sing
                a.append(row[4])
                ch.append(row[7])
            counts[name].append(line_metrics(s, h, a, ch))
        if (i + 1) % 10 == 0:
            print(f'{i + 1}/{len(S["test"])} {time.time() - t0:.0f}s', flush=True)
    for name in counts:
        out['per_planner'][name] = rates(pool(counts[name]))
    out['segmentation'] = {**seg, 'onset_recall_%': 100 * seg['matched'] / seg['score_notes'],
                           'pitch_correct_of_matched_%': 100 * seg['pitch_ok'] / max(seg['matched'], 1)}
    out['seconds'] = time.time() - t0
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'pipeline_results.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
