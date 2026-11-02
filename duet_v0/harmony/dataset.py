"""Train / validation / test split of the Bach chorales, grouped by TUNE.

Bach harmonised many hymn tunes more than once (e.g. several settings of the same melody
under different titles). Splitting by chorale file alone would leak: the test soprano would
have been seen in training with a (possibly identical) alto. We therefore group chorales whose
soprano lines share most of their pitch-interval 4-grams (Jaccard >= 0.5, transposition
invariant) into one tune group with union-find, and split whole groups 70 / 15 / 15.

Run:  .venv/bin/python harmony/dataset.py   -> writes harmony/data/split.json
"""
import json
import os

import numpy as np

from chorales import load

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(HERE, 'data', 'split.json')


def _grams(rec, n=4):
    s = [r[2] for r in rec['notes']]
    d = [b - a for a, b in zip(s[:-1], s[1:]) if b != a]      # ignore repeated notes
    return {tuple(d[i:i + n]) for i in range(len(d) - n + 1)}


def tune_groups(chorales, thr=0.5):
    G = [_grams(r) for r in chorales]
    parent = list(range(len(chorales)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            if not G[i] or not G[j]:
                continue
            jac = len(G[i] & G[j]) / len(G[i] | G[j])
            if jac >= thr:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(G))]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots]


def make_split(seed=5160, frac=(0.70, 0.15, 0.15)):
    ch = load()['chorales']
    g = np.array(tune_groups(ch))
    ng = g.max() + 1
    rng = np.random.default_rng(seed)
    order = rng.permutation(ng)
    sizes = np.bincount(g, minlength=ng)
    target = np.cumsum(frac) * len(ch)
    assign = {}
    acc = 0
    for gi in order:
        acc += sizes[gi]
        assign[int(gi)] = 'train' if acc <= target[0] else ('val' if acc <= target[1] else 'test')
    split = {'train': [], 'val': [], 'test': []}
    for rec, gi in zip(ch, g):
        split[assign[int(gi)]].append(rec['id'])
    info = {'seed': seed, 'n_chorales': len(ch), 'n_tune_groups': int(ng),
            'largest_group': int(sizes.max()), 'groups_with_2plus': int((sizes >= 2).sum()),
            'group_of': {rec['id']: int(gi) for rec, gi in zip(ch, g)}}
    return split, info


def load_split():
    with open(SPLIT) as fh:
        js = json.load(fh)
    ch = {r['id']: r for r in load()['chorales']}
    return {k: [ch[i] for i in js[k]] for k in ('train', 'val', 'test')}


if __name__ == '__main__':
    split, info = make_split()
    with open(SPLIT, 'w') as fh:
        json.dump({**split, 'info': info}, fh, indent=1)
    print({k: len(v) for k, v in split.items()},
          {k: v for k, v in info.items() if k != 'group_of'})
