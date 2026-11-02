"""Plug-in check: run the existing live simulation (run_v0.run_scenario) with the learned plan.

    .venv/bin/python harmony/integration_check.py

Same synthetic singer, same calibration and scenario as run_v0.py; only the plan differs
(follow.plan_harmony vs model.plan_harmony_learned). Timing should be unchanged (the
Accompanist uses plan onsets, not pitches); the harmony pitches differ. Writes wavs and
metrics under harmony/results/integration/<planner>/<scenario>/ (about 5 MB per scenario).
"""
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_v0                                                        # noqa: E402
import singer                                                        # noqa: E402
from follow import analyse, offline_notes, plan_harmony              # noqa: E402
from model import plan_harmony_learned                               # noqa: E402

SCEN = ('natural', 'pause')


def main():
    s_notes = singer.score_notes()
    xs, true_s = singer.render(s_notes)
    F = analyse(xs)
    det = offline_notes(F)
    gaps = run_v0.find_gaps(F)
    ref_bias, live_bias, _ = run_v0.calibrate(xs, true_s, F, det, [(a + b) / 2 for a, b in gaps])
    true_on = [n['on'] for n in true_s]
    truth_on = [true_on[k] if k is not None else None for k in run_v0.match([n['on'] for n in det], true_on)]
    for nt in det:
        nt['on'] -= ref_bias.get(nt['kind'], 0.0)
    dur = len(xs) / 24000
    join_r = next((nt['on'] for nt in det if nt['on'] >= 0.25 * dur and nt['kind'] == 'attack'),
                  next(nt['on'] for nt in det if nt['on'] >= 0.25 * dur))
    out = {}
    for label, fn in (('rule', plan_harmony), ('learned', plan_harmony_learned)):
        plan, key = fn(det)
        out[label] = {'key': key}
        for name in SCEN:
            m = run_v0.run_scenario(name, dict(run_v0.SCENARIOS[name]), xs, F, det, plan, key, join_r, gaps,
                                    truth_on, Path(HERE) / 'results' / 'integration' / label / name, live_bias)
            p = m['pred']
            out[label][name] = {'median_abs_ms': p['median_abs_ms'], 'p90_abs_ms': p['p90_abs_ms'],
                                'played': p['played'], 'held': p['held'], 'missed': p['missed'],
                                'wrong': p['wrong'], 'harmony_consonant': m['harmony_consonant']}
            print(label, name, out[label][name], flush=True)
    with open(os.path.join(HERE, 'results', 'integration_results.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
