"""How much better could Duet get if it predicted her next beat better? (score input, no audio)

Every system is the main HMM + planner; they differ only in a guess of beat k's melody that the
belief is updated with at decision time (planner.run's `hint`):
  hmm          no guess (today's system)
  repeat_C     causal: find the most recent earlier place in THIS song whose last C beats of melody
               (cut at the same decision point) and bar position match, and guess she sings what
               she sang next there. Uses only beats fully heard before the decision.
  repeat_ideal the same matches, but the guess is replaced by the truth (is the match worth more?)
  next_true    ceiling: told her exact melody for beat k on every beat
  oracle       ceiling: told the true chord
Writes duet_beat/out/ceiling.json.  Run:  .venv/bin/python duet_beat/exp_ceiling.py
"""
import json
import math
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load  # noqa: E402
from metrics import bootstrap, song_counts, summarize  # noqa: E402
from model import ChordHMM, soprano_obs  # noqa: E402
from planner import Config, run  # noqa: E402
from songs import arpeggios, twinkle  # noqa: E402

GRID = 12                                         # melody sampled 12 times per beat for matching


def sampled(sop, a, b):
    """Her pitch at GRID points per beat in [a, b) (0 = rest), as a tuple."""
    out = []
    n = int(round((b - a) * GRID))
    for i in range(n):
        x = a + (i + 0.5) / GRID
        out.append(next((m for on, off, m in sop if on <= x < off), 0))
    return tuple(out)


class Repeat:
    def __init__(self, song, C, D, ideal=False):
        self.sop, self.C, self.D, self.ideal = song['sop'], C, D, ideal
        self.pos = [b['pos'] for b in song['beats']]
        self.fired = self.correct = 0

    def __call__(self, k, jstar, frac):
        if jstar < self.C:
            return None
        cut = k - self.D                                   # the decision time, in beats
        ctx = sampled(self.sop, cut - self.C, cut)
        if not any(ctx):
            return None
        for j in range(jstar - 1, self.C - 1, -1):         # beat j must be fully heard: j + 1 <= jstar
            if self.pos[j] != self.pos[k]:
                continue
            if sampled(self.sop, j - self.D - self.C, j - self.D) == ctx:
                self.fired += 1
                guess = soprano_obs(self.sop, j)
                truth = soprano_obs(self.sop, k)
                self.correct += int(sampled(self.sop, j, j + 1) == sampled(self.sop, k, k + 1))
                return truth if self.ideal else guess
        return None


def score(songs, model, cfg, system):
    cs, fired, correct, beats = [], 0, 0, 0
    D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
    for s in songs:
        hint, sysname = None, 'hmm'
        if system == 'oracle':
            sysname = 'oracle'
        elif system == 'next_true':
            hint = lambda k, j, f, s=s: soprano_obs(s['sop'], k)  # noqa: E731
        elif system.startswith('repeat'):
            C = int(system.split('_')[1])
            hint = Repeat(s, C, D, ideal=system.endswith('ideal'))
        c = replace(cfg, q_min=0.0) if system == 'oracle' else cfg
        cs.append(song_counts(s, run(s, model, c, sysname, hint=hint), cfg.J))
        if isinstance(hint, Repeat):
            fired, correct = fired + hint.fired, correct + hint.correct
        beats += s['K']
    out = {**summarize(cs), 'ci': bootstrap(cs, n=500)}
    if system.startswith('repeat'):
        out['fired'] = fired / beats
        out['guess_right'] = correct / fired if fired else float('nan')
    return out


def main():
    d = load()['split']
    model = ChordHMM().fit(d['train'])
    res = json.load(open(os.path.join(HERE, 'out', 'results.json')))
    base = dict(res['tuning']['chosen']['main'])
    base['alto'] = tuple(base['alto'])
    cfg = Config(**base)
    sets = {'bach_test': d['test'], 'twinkle': [twinkle()], 'arpeggios': [arpeggios()]}
    systems = ('hmm', 'repeat_2', 'repeat_4', 'repeat_4_ideal', 'next_true', 'oracle')
    out = {}
    for name, songs in sets.items():
        out[name] = {}
        print(f'\n{name}  ({len(songs)} songs, {sum(s["K"] for s in songs)} beats)')
        print(f'  {"system":15s} chord  in-chord  consonant   U      repeat fired / guess right')
        for sysn in systems:
            bpm = songs[0].get('bpm', cfg.bpm) if name != 'bach_test' else cfg.bpm
            r = score(songs, model, replace(cfg, bpm=bpm), sysn)
            out[name][sysn] = r
            extra = f"   {r['fired']:.2f} / {r['guess_right']:.2f}" if 'fired' in r else ''
            print(f"  {sysn:15s} {r['m1']:.3f}  {r['m2']:.3f}     {r['m3']:.3f}     {r['u']:+.3f}{extra}")
    with open(os.path.join(HERE, 'out', 'ceiling.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
