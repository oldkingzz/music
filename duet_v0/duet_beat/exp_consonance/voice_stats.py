"""Voice-leading sanity stats + causality power check for the consonance experiment (test split, read-only).
Run from duet_beat/:  ../.venv/bin/python exp_consonance/voice_stats.py"""
import copy, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from data import load
from metrics import song_counts, summarize, bootstrap
from model import ChordHMM
from planner import Config, run
from run_consonance import P, run_exp, learn_step_prior, her_notes

d = load()['split']; model = ChordHMM().fit(d['train']); test = d['test']
R = json.load(open(os.path.join(HERE, 'results.json')))
cfg = Config(**{k: (tuple(v) if k == 'alto' else v) for k, v in R['cfg'].items()})
Ps = np.array(R['step_prior'])

def stats(runs):
    leaps, cross, unis, tot = [], 0.0, 0.0, 0.0
    for s, bs in zip(test, runs):
        prev = None
        for b in bs:
            if b.k < cfg.J or b.h < 0:
                prev = None; continue
            if prev is not None: leaps.append(abs(b.h - prev))
            prev = b.h
            for m, ov in her_notes(s['sop'], b.k):
                tot += ov; cross += ov * (m < b.h); unis += ov * (m == b.h)
    L = np.array(leaps)
    return dict(mean_leap=L.mean(), leap_gt7=(L > 7).mean(), repeat=(L == 0).mean(), her_below_duet=cross / tot, unison=unis / tot)

systems = {'hmm_long_N4': [run(s, model, cfg, 'hmm') for s in test],
           'hmm_long_N4_qmin0 (sings every beat)': [run(s, model, Config(**{**cfg.__dict__, 'q_min': 0.0}), 'hmm') for s in test],
           'B0_tonic': [run(s, model, Config(**{**cfg.__dict__, 'q_min': 0.0}), 'tonic') for s in test]}
for name in ('EC_chord_x_step', 'EC_chord_x_step_+M2', 'EC_chord_only'):
    systems[name] = [run_exp(s, model, cfg, Ps, P(**R['tuning'][name]['chosen'])) for s in test]
out = {}
for name, runs in systems.items():
    cs = [song_counts(s, r, cfg.J) for s, r in zip(test, runs)]
    sm = summarize(cs); st = stats(runs)
    out[name] = {**st, 'm2': sm['m2'], 'm3': sm['m3'], 'm3_ci': bootstrap(cs)['m3'], 'm4': sm['m4'], 'm5': sm['m5']}
    print(f"{name:38s} M2 {sm['m2']:.3f} M3 {sm['m3']:.3f} M4 {sm['m4']:.2f} M5 {sm['m5']:.2f} | " + ' '.join(f'{k} {v:.3f}' for k, v in st.items()))

# causality power: perturbing future notes must change LATER decisions (so the check can fail)
p = P(**R['tuning']['EC_chord_x_step_+M2']['chosen']); changed = 0
for s in test[:10]:
    k0 = s['K'] // 2; a = copy.deepcopy(s)
    for nt in a['sop']:
        if nt[0] >= k0 - 0.432: nt[2] += 5
    r0, r1 = run_exp(s, model, cfg, Ps, p), run_exp(a, model, cfg, Ps, p)
    changed += sum(x.h != y.h for x, y in zip(r0[k0 + 1:], r1[k0 + 1:]))
print('power: later decisions changed', changed)
out['_power_changed_later_decisions'] = changed
json.dump(out, open(os.path.join(HERE, 'voice_stats.json'), 'w'), indent=1, default=float)
