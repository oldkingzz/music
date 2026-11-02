"""Three follow-up questions (10/29 chat).

A. Four-part output: if Duet sang the whole guessed chord under her (not one note), how often is
   her note one of that chord's notes?  'support' = time-weighted share of her sounding melody, over
   the beats Duet sings, whose pitch class is in CT[z_hat]. Shown next to the two-part m3.
B. More data: refit the HMM on 25 % / 50 % / 100 % of the training chorales (3 random subsets).
C. A bigger predictor for her next note: n-gram over her beat notes (last 1-3 notes + bar position,
   backoff), against the HMM's own prediction and two trivial baselines. Previous beat fully heard.
Writes duet_beat/out/more.json.  Run:  .venv/bin/python duet_beat/exp_more.py
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load  # noqa: E402
from exp_ceiling import Repeat  # noqa: E402
from metrics import song_counts, summarize  # noqa: E402
from model import ChordHMM, soprano_obs  # noqa: E402
from planner import Config, run  # noqa: E402
from songs import twinkle  # noqa: E402
from static import run_static  # noqa: E402
from theory import CT  # noqa: E402


def support(song, beats, J):
    n = d = 0.0
    t = song['tonic']
    for b in beats:
        if b.k < J or b.h < 0 or b.z_hat < 0:
            continue
        for on, off, m in song['sop']:
            ov = min(off, b.k + 1) - max(on, b.k)
            if ov > 1e-9:
                d += ov
                n += ov * ((m - t) % 12 in CT[b.z_hat])
    return n, d


def systems(song, model, cfg, name):
    D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
    if name == 'static':
        return run_static(song, model, cfg)
    if name == 'oracle':
        return run(song, model, replace(cfg, q_min=0.0), 'oracle')
    hint = {'hmm': None, 'repeat_2': Repeat(song, 2, D),
            'next_true': lambda k, j, f: soprano_obs(song['sop'], k)}[name]
    return run(song, model, cfg, 'hmm', hint=hint)


def score(songs, model, cfg, name):
    cs, sn, sd = [], 0.0, 0.0
    for s in songs:
        c = replace(cfg, bpm=s['bpm']) if 'bpm' in s else cfg
        beats = systems(s, model, c, name)
        cs.append(song_counts(s, beats, cfg.J))
        n, d = support(s, beats, cfg.J)
        sn, sd = sn + n, sd + d
    sm = summarize(cs)
    return {'two_part': sm['m3'], 'four_part_support': sn / sd, 'chord': sm['m1']}


def notes(song):
    return [None if m is None else (m - song['tonic']) % 12
            for m in (soprano_obs(song['sop'], k)[1] for k in range(song['K']))]


class NGram:
    def __init__(self, n):
        self.n = n
        self.c = defaultdict(Counter)

    def fit(self, songs):
        for s in songs:
            o = notes(s)
            for k in range(1, s['K']):
                if o[k] is None:
                    continue
                for m in range(0, self.n + 1):
                    ctx = tuple(o[max(0, k - m):k])
                    self.c[(m, s['beats'][k]['pos'], ctx)][o[k]] += 1
        return self

    def predict(self, o, k, pos):
        for m in range(self.n, -1, -1):                  # longest context seen in training
            c = self.c.get((m, pos, tuple(o[max(0, k - m):k])))
            if c:
                return c.most_common(1)[0][0]


def next_note(songs, model, train, J):
    preds = {'most common note': None, 'same as last beat': None, 'HMM': None}
    preds.update({f'{n}-gram': NGram(n).fit(train) for n in (1, 2, 3)})
    top = Counter(x for s in train for x in notes(s) if x is not None).most_common(1)[0][0]
    hit, tot = Counter(), 0
    for s in songs:
        o, pos, M = notes(s), [b['pos'] for b in s['beats']], s['M']
        alpha = None
        for k in range(s['K']):
            p = model.init if k == 0 else model.predict(alpha, pos[k], M)
            if k >= J and o[k] is not None:
                tot += 1
                hit['most common note'] += o[k] == top
                hit['same as last beat'] += o[k] == o[k - 1]
                hit['HMM'] += o[k] == int(np.argmax(p @ model.B))
                for n in (1, 2, 3):
                    hit[f'{n}-gram'] += o[k] == preds[f'{n}-gram'].predict(o, k, pos[k])
            alpha = model.update(p, model.loglik(o[k], 'long'))
    return {k: hit[k] / tot for k in preds}


def main():
    d = load()['split']
    res = json.load(open(os.path.join(HERE, 'out', 'results.json')))
    base = dict(res['tuning']['chosen']['main'])
    base['alto'] = tuple(base['alto'])
    cfg = Config(**base)
    model = ChordHMM().fit(d['train'])
    sets = {'bach_test': d['test'], 'twinkle': [twinkle()]}
    out = {'A': {}, 'B': {}, 'C': {}}

    print('A. two-part consonance vs four-part support (her note in Duet\'s chord)')
    for name, songs in sets.items():
        out['A'][name] = {}
        for sysn in ('hmm', 'repeat_2', 'next_true', 'static', 'oracle'):
            r = score(songs, model, cfg, sysn)
            out['A'][name][sysn] = r
            print(f"  {name:9s} {sysn:10s} two-part {r['two_part']:.3f}  four-part {r['four_part_support']:.3f}"
                  f"  chord {r['chord']:.3f}")

    print('B. learning curve on the test chorales (hmm)')
    n_all = len(d['train'])
    for frac in (0.25, 0.5, 1.0):
        rs = []
        for seed in range(3 if frac < 1 else 1):
            sub = random.Random(seed).sample(d['train'], int(round(frac * n_all)))
            rs.append(score(d['test'], ChordHMM().fit(sub), cfg, 'hmm'))
        avg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        out['B'][frac] = avg
        print(f"  {frac:4.0%} of train ({int(round(frac * n_all))} chorales): two-part {avg['two_part']:.3f}"
              f"  four-part {avg['four_part_support']:.3f}  chord {avg['chord']:.3f}")

    print('C. guessing her next beat\'s note (previous beat fully heard)')
    for name, songs in sets.items():
        r = next_note(songs, model, d['train'], cfg.J)
        out['C'][name] = r
        print(f'  {name:9s} ' + '  '.join(f'{k} {v:.3f}' for k, v in r.items()))
    with open(os.path.join(HERE, 'out', 'more.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
