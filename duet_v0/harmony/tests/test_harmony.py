"""Tests for the learned harmony generator.

    cd project/duet_v0 && .venv/bin/python -m unittest discover -s harmony/tests -v
"""
import itertools
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARM = os.path.dirname(HERE)
ROOT = os.path.dirname(HARM)
sys.path.insert(0, ROOT)
sys.path.insert(0, HARM)

import singer                                                    # noqa: E402
from follow import Accompanist, plan_harmony                     # noqa: E402
from metrics import line_metrics                                  # noqa: E402
from model import (DISSONANT_2V, HarmonyModel, Templates, detect_key_melody,  # noqa: E402
                   duet_masks, frame_shift, plan_harmony_learned, viterbi)
from train import _fb                                             # noqa: E402


def greensleeves():
    return [{'on': a, 'off': b, 'midi': m, 'kind': 'attack'} for a, b, m in singer.score_notes()]


def random_melody(rng, n=30, tonic=0):
    scale = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16]
    t, out = 0.3, []
    for _ in range(n):
        d = float(rng.choice([0.3, 0.6, 0.6, 1.2]))
        out.append({'on': t, 'off': t + d - 0.02, 'midi': int(60 + tonic + rng.choice(scale)), 'kind': 'attack'})
        t += d
    return out


class TestPlanContract(unittest.TestCase):
    """plan_harmony_learned must be a drop-in for follow.plan_harmony."""

    def test_same_keys_and_notes_as_rule(self):
        rng = np.random.default_rng(0)
        for notes in [greensleeves()] + [random_melody(rng) for _ in range(5)]:
            notes = list(notes)
            notes[3] = dict(notes[3], midi=0)                    # an unpitched segment
            notes[5] = dict(notes[5], off=notes[5]['on'] + 0.05)  # a too-short segment
            rule, _ = plan_harmony(notes)
            learned, key = plan_harmony_learned(notes)
            self.assertEqual([p['note'] for p in rule], [p['note'] for p in learned])
            for a, b in zip(rule, learned):
                self.assertEqual(set(a), set(b))
                for k in ('note', 'r', 'dur', 'midi'):
                    self.assertEqual(a[k], b[k])
                self.assertIsInstance(b['h'], int)
            self.assertRegex(key, r'^[A-G][#b]? (major|minor)$')

    def test_harmony_below_melody_within_range(self):
        rng = np.random.default_rng(1)
        for notes in [greensleeves()] + [random_melody(rng, tonic=t) for t in range(-5, 7)]:
            plan, _ = plan_harmony_learned(notes)
            iv = np.array([p['midi'] - p['h'] for p in plan])
            self.assertTrue((iv >= 0).all() and (iv <= 16).all(), iv)

    def test_default_decoding_has_no_parallel_perfects(self):
        rng = np.random.default_rng(2)
        for notes in [greensleeves()] + [random_melody(rng, n=60) for _ in range(10)]:
            plan, _ = plan_harmony_learned(notes)
            c = line_metrics([p['midi'] for p in plan], [p['h'] for p in plan])
            self.assertEqual(c['par5'] + c['par8'], 0)

    def test_edge_cases(self):
        self.assertEqual(plan_harmony_learned([])[0], [])
        one = [{'on': 0.0, 'off': 1.0, 'midi': 67, 'kind': 'attack'}]
        plan, _ = plan_harmony_learned(one)
        self.assertEqual(len(plan), 1)
        two = one + [{'on': 1.0, 'off': 2.0, 'midi': 0, 'kind': 'attack'}]
        self.assertEqual(len(plan_harmony_learned(two)[0]), 1)

    def test_register_follows_singer_octave(self):
        """A man singing the tune an octave lower gets the same line an octave lower."""
        g = greensleeves()
        low = [dict(n, midi=n['midi'] - 12) for n in g]
        h1 = [p['h'] for p in plan_harmony_learned(g)[0]]
        h2 = [p['h'] for p in plan_harmony_learned(low)[0]]
        self.assertEqual([a - 12 for a in h1], h2)

    def test_transposition_equivariance(self):
        """Same melody in another key -> same intervals (key detection and frame are relative)."""
        g = greensleeves()
        base = [p['midi'] - p['h'] for p in plan_harmony_learned(g)[0]]
        for k in (-2, -1, 1, 2):
            moved = [dict(n, midi=n['midi'] + k) for n in g]
            iv = [p['midi'] - p['h'] for p in plan_harmony_learned(moved)[0]]
            self.assertEqual(base, iv, f'transposed by {k}')

    def test_accompanist_accepts_plan(self):
        g = greensleeves()
        plan, _ = plan_harmony_learned(g)
        acc = Accompanist(g, plan, join_r=g[1]['on'])
        self.assertEqual(len(acc.plan), len(plan) - 1)


class TestInference(unittest.TestCase):
    def test_viterbi_matches_brute_force(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            T, K = 4, 5
            u = rng.normal(size=(T, K))
            p = rng.normal(size=(T - 1, K, K))
            best = max(itertools.product(range(K), repeat=T),
                       key=lambda y: sum(u[t, y[t]] for t in range(T)) + sum(p[t, y[t], y[t + 1]] for t in range(T - 1)))
            self.assertEqual(list(viterbi(u, p)), list(best))

    def test_forward_backward_partition_and_marginals(self):
        rng = np.random.default_rng(4)
        T, K = 4, 4
        u = rng.normal(size=(T, K))
        p = rng.normal(size=(T - 1, K, K))
        paths = list(itertools.product(range(K), repeat=T))
        sc = np.array([sum(u[t, y[t]] for t in range(T)) + sum(p[t, y[t], y[t + 1]] for t in range(T - 1))
                       for y in paths])
        logZ = np.log(np.exp(sc - sc.max()).sum()) + sc.max()
        lz, q, xi = _fb(u, p)
        self.assertAlmostEqual(lz, logZ, places=9)
        prob = np.exp(sc - logZ)
        q_bf = np.zeros((T, K))
        for y, pr in zip(paths, prob):
            for t in range(T):
                q_bf[t, y[t]] += pr
        np.testing.assert_allclose(q, q_bf, atol=1e-10)
        self.assertTrue(np.allclose(xi.sum((1, 2)), 1.0))

    def test_duet_masks(self):
        tpl = Templates(0, 16)
        c = {'s': np.array([72, 74, 74]), 'ds': np.array([0, 2, 0]), 'dur': np.array([1, 1, 2]),
             'end': np.array([0, 0, 1])}
        u, p = duet_masks(c, tpl, lam=1.0, no_parallel=True, beta=0.5)
        iv = np.arange(17)
        self.assertTrue(np.all(u[0][DISSONANT_2V[iv % 12]] == -1.0))
        self.assertTrue(np.all(u[2][DISSONANT_2V[iv % 12]] == -2.5))       # long + phrase end
        self.assertEqual(p[0, 7, 7], -np.inf)                    # fifth -> fifth while melody moves
        self.assertEqual(p[0, 12, 0], -np.inf)                   # octave -> unison while melody moves
        self.assertEqual(p[1, 7, 7], -0.5)                       # melody repeats: held note, penalised
        self.assertEqual(p[0, 3, 5], -0.5)                       # 72-3=69 -> 74-5=69: held note


class TestKeyAndFrame(unittest.TestCase):
    def test_frame_shift(self):
        for tonic in range(12):
            sh = frame_shift(tonic)
            self.assertEqual((tonic + sh) % 12, 0)
            self.assertTrue(-6 <= sh <= 5)

    def test_key_greensleeves(self):
        self.assertEqual(detect_key_melody(greensleeves()), (9, 'minor'))


class TestDataHygiene(unittest.TestCase):
    def test_split_is_tune_disjoint(self):
        with open(os.path.join(HARM, 'data', 'split.json')) as fh:
            js = json.load(fh)
        g = js['info']['group_of']
        sets = {k: {g[i] for i in js[k]} for k in ('train', 'val', 'test')}
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['train'] & sets['val'])
        self.assertFalse(sets['val'] & sets['test'])
        ids = js['train'] + js['val'] + js['test']
        self.assertEqual(len(ids), len(set(ids)))

    def test_params_file_loads(self):
        m = HarmonyModel.load('alto')
        self.assertEqual(m.w.shape, (m.tpl.W,))
        self.assertTrue(np.isfinite(m.w).all())
        self.assertIn('duet', m.cfg)


if __name__ == '__main__':
    unittest.main()
