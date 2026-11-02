"""Evaluate harmony generators on held-out Bach chorales (tune-disjoint test split).

    .venv/bin/python harmony/evaluate.py            # writes harmony/results/*.json + figures

Steps
  1. duet-penalty sweep (lambda) on the VALIDATION split -> chooses the shipped default
  2. key-detection accuracy (follow.detect_key vs model.detect_key_melody) on val / test
  3. all generators on the TEST split: pooled metrics, bootstrap CIs over chorales,
     paired bootstrap of learned vs rule
  4. runtime per song
The markdown report harmony/eval.md is written by report.py from these JSON files.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from chorales import to_notes                                          # noqa: E402
from dataset import load_split                                         # noqa: E402
from follow import detect_key, plan_harmony                            # noqa: E402
from metrics import bootstrap_ci, line_metrics, paired_bootstrap, pool, rates  # noqa: E402
from model import HarmonyModel, detect_key_melody                      # noqa: E402

RES = os.path.join(HERE, 'results')
LAMS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def load_models():
    p = lambda n: os.path.join(HERE, 'params', n)                       # noqa: E731
    return {'crf': HarmonyModel.load('alto'),
            'unigram': HarmonyModel.load(path=p('unigram_alto.npz')),
            'count_hmm': HarmonyModel.load(path=p('count_hmm_alto.npz'))}


def generators(M, lam_duet, beta_duet):
    crf = M['crf']
    return {
        'Rule (follow.plan_harmony)': lambda rec, n: plan_harmony(n)[0],
        'Unigram counts': lambda rec, n: M['unigram'].harmonize(n, lam=0.0, beta=0.0, no_parallel=False)[0],
        'Count HMM + Viterbi': lambda rec, n: M['count_hmm'].harmonize(n, lam=0.0, beta=0.0, no_parallel=False)[0],
        'CRF (Bach style)': lambda rec, n: crf.harmonize(n, lam=0.0, beta=0.0, no_parallel=False)[0],
        'CRF + duet constraints': lambda rec, n: crf.harmonize(n, lam=lam_duet, beta=beta_duet,
                                                               no_parallel=True)[0],
        'CRF + duet, oracle key': lambda rec, n: crf.harmonize(n, key=(rec['tonic'], rec['mode']),
                                                               lam=lam_duet, beta=beta_duet,
                                                               no_parallel=True)[0],
        'Bach (reference)': lambda rec, n: [{'note': k, 'h': r[4]} for k, r in enumerate(rec['notes'])
                                            if r[4] is not None],
    }


def piece_counts(rec, plan):
    rows = rec['notes']
    s = [rows[p['note']][2] for p in plan]
    h = [p['h'] for p in plan]
    a = [rows[p['note']][4] for p in plan]
    ch = [rows[p['note']][7] for p in plan]
    return line_metrics(s, h, a, ch)


def run(gen, recs):
    per = []
    for rec in recs:
        per.append(piece_counts(rec, gen(rec, to_notes(rec))))
    return per


def duet_sweep(crf, recs):
    """Grid over the two decoding penalties (lambda: dissonance, beta: repeated note) on val."""
    out = []
    for lam in LAMS:
        for beta in BETAS:
            per = run(lambda rec, n: crf.harmonize(n, lam=lam, beta=beta, no_parallel=True)[0], recs)
            r = rates(pool(per))
            out.append({'lam': lam, 'beta': beta, **r})
            print(f"  lam={lam:<4} beta={beta:<4} exact {r['alto_exact_%']:.1f}  cons2v "
                  f"{r['consonance_2v_%']:.1f}  chord {r['chord_tone_%']:.1f}  "
                  f"repeat {r['repeated_note_%']:.1f}", flush=True)
    return out


def choose_duet(sweep, bach_repeat, min_cons=95.0):
    """Product requirements for a two-voice duet, checked on validation:
         two-voice consonance >= 95 % (at most ~1 dissonant note in 20)
         repeated-note rate   <= Bach's own alto (no drones)
       Among settings that meet both, the one closest to Bach (exact alto accuracy) wins."""
    ok = [r for r in sweep if r['consonance_2v_%'] >= min_cons and r['repeated_note_%'] <= bach_repeat]
    if not ok:
        ok = sweep
    best = max(ok, key=lambda r: (round(r['alto_exact_%'], 1), r['chord_tone_%']))
    return best['lam'], best['beta']


def key_accuracy(recs):
    kk = ae = 0
    for rec in recs:
        n = to_notes(rec)
        t, m, _ = detect_key(n)
        kk += (t == rec['tonic'] and m == rec['mode'])
        t, m = detect_key_melody(n)
        ae += (t == rec['tonic'] and m == rec['mode'])
    return {'n': len(recs), 'follow.detect_key_%': 100 * kk / len(recs),
            'detect_key_melody_%': 100 * ae / len(recs)}


def timing(crf, recs):
    import singer
    g = [{'on': a, 'off': b, 'midi': m} for a, b, m in singer.score_notes()]
    longest = max(recs, key=lambda r: len(r['notes']))
    out = {}
    for name, notes in (('greensleeves', g), (f"longest test chorale ({len(longest['notes'])} notes)",
                                              to_notes(longest))):
        crf.harmonize(notes)
        t0 = time.perf_counter()
        for _ in range(20):
            crf.harmonize(notes)
        out[name] = 1000 * (time.perf_counter() - t0) / 20
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    S = load_split()
    M = load_models()
    crf = M['crf']
    print('duet decoding sweep on validation')
    bach_val = rates(pool(run(generators(M, 0, 0)['Bach (reference)'], S['val'])))
    rule_val = rates(pool(run(generators(M, 0, 0)['Rule (follow.plan_harmony)'], S['val'])))
    sweep = duet_sweep(crf, S['val'])
    lam, beta = choose_duet(sweep, bach_val['repeated_note_%'])
    print('chosen lambda, beta', lam, beta)
    crf.cfg['duet'] = {'lam': lam, 'beta': beta, 'no_parallel': True, 'max_iv': None}
    crf.save(os.path.join(HERE, 'params', 'crf_alto.npz'))           # ship the chosen default
    keys = {'val': key_accuracy(S['val']), 'test': key_accuracy(S['test'])}
    print('key accuracy', keys)
    res = {'split_sizes': {k: len(v) for k, v in S.items()}, 'lam': lam, 'beta': beta,
           'bach_val': bach_val, 'rule_val': rule_val, 'duet_sweep_val': sweep,
           'key_accuracy': keys, 'test': {}, 'test_ci': {}, 'paired': {}}
    per_all = {}
    for name, gen in generators(M, lam, beta).items():
        per = run(gen, S['test'])
        per_all[name] = per
        res['test'][name] = rates(pool(per))
        res['test_ci'][name] = {
            'alto_exact_%': bootstrap_ci(per, 'exact', 'n_lab'),
            'chord_tone_%': bootstrap_ci(per, 'chordtone', 'n'),
            'consonance_2v_%': bootstrap_ci(per, 'cons2', 'n')}
        r = res['test'][name]
        print(f"{name:28s} exact {r['alto_exact_%']:5.1f}  chord {r['chord_tone_%']:5.1f}  "
              f"cons2 {r['consonance_2v_%']:5.1f}  par5 {r['parallel_5ths_per_100']:.2f}  "
              f"par8 {r['parallel_8ves_per_100']:.2f}  leap>P5 {r['leaps_gt_P5_%']:.1f}", flush=True)
    rule = per_all['Rule (follow.plan_harmony)']
    for name in ('CRF (Bach style)', 'CRF + duet constraints'):
        res['paired'][name] = {
            'alto_exact_%': paired_bootstrap(per_all[name], rule, 'exact', 'n_lab'),
            'chord_tone_%': paired_bootstrap(per_all[name], rule, 'chordtone', 'n'),
            'consonance_2v_%': paired_bootstrap(per_all[name], rule, 'cons2', 'n')}
    res['interval_hist'] = {}
    G = generators(M, lam, beta)
    for name in ('Rule (follow.plan_harmony)', 'CRF (Bach style)', 'CRF + duet constraints', 'Bach (reference)'):
        hist = np.zeros(13)
        mot = np.zeros(25)
        for rec in S['test']:
            plan = G[name](rec, to_notes(rec))
            s = np.array([rec['notes'][p['note']][2] for p in plan])
            h = np.array([p['h'] for p in plan])
            np.add.at(hist, np.clip(s - h, 0, 12), 1)
            np.add.at(mot, np.clip(np.diff(h), -12, 12) + 12, 1)
        res['interval_hist'][name] = {'interval': hist.tolist(), 'motion': mot.tolist()}
    res['runtime_ms'] = timing(crf, S['test'])
    print('runtime ms', res['runtime_ms'])
    with open(os.path.join(RES, 'eval_results.json'), 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
