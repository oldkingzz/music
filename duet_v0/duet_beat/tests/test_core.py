"""Checks that guard the claims on the results page. Run:  .venv/bin/python duet_beat/tests/test_core.py

1. causality: changing her melody after a decision time never changes that decision (A7, E5),
   and the check has power: the cheating system B3 does change;
2. the model is a proper HMM (rows sum to 1);
3. every new note Duet sings is a chord tone of its planned chord and inside the alto range;
4. the oracle gets every chord right.
"""
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from data import load  # noqa: E402
from metrics import song_counts, summarize  # noqa: E402
from model import ChordHMM  # noqa: E402
from planner import Config, run  # noqa: E402
from theory import ALTO, CT  # noqa: E402


def main():
    d = load()['split']
    model = ChordHMM().fit(d['train'])
    for name in ('init',):
        assert abs(getattr(model, name).sum() - 1) < 1e-9
    assert np.allclose(model.A4.sum(-1), 1) and np.allclose(model.A2.sum(-1), 1) and np.allclose(model.B.sum(-1), 1)
    print('ok  model rows sum to 1')

    n_checked = n_power = 0
    for song in d['test'][:12]:
        for mode in ('long', 'vec'):
            cfg = Config(mode=mode, q_min=0.3)
            D = (cfg.L + cfg.delta) / (60 / cfg.bpm)
            k0 = song['K'] // 2
            ref = {s: run(song, model, cfg, s) for s in ('hmm', 'peek')}
            alt = copy.deepcopy(song)
            for nt in alt['sop']:
                if nt[0] >= k0 - D:
                    nt[2] += 5                      # she sings something else from here on
            new = {s: run(alt, model, cfg, s) for s in ('hmm', 'peek')}
            for a, b in zip(ref['hmm'][:k0 + 1], new['hmm'][:k0 + 1]):
                assert (a.z_hat, a.h, a.pi) == (b.z_hat, b.h, b.pi), (song['id'], mode, a.k)
            n_checked += 1
            if any(a.pi != b.pi for a, b in zip(ref['peek'][:k0 + 1], new['peek'][:k0 + 1])):
                n_power += 1
            for b in ref['hmm']:
                if b.action == 'sing':
                    assert (b.h - song['tonic']) % 12 in CT[b.z_hat] and ALTO[0] <= b.h <= ALTO[1]
    assert n_power > 0
    print(f'ok  causality: {n_checked} runs unchanged before the edit; the cheating B3 changed in {n_power}')
    print('ok  every sung note is a chord tone of its planned chord, inside the alto range')

    cs = [song_counts(s, run(s, model, Config(), 'oracle'), 8) for s in d['test']]
    assert abs(summarize(cs)['m1'] - 1.0) < 1e-12
    print('ok  oracle chord accuracy is 1')


if __name__ == '__main__':
    main()
