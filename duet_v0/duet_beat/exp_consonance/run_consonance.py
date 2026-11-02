"""Experiment: choose Duet's note by expected consonance with HER NEXT NOTE, not by committing to a chord.

At the decision time for beat k (k - D beats, same causal cut as planner.run) we have
  * the HMM chord belief for beat k (identical computation to planner.run, system 'hmm', mode 'long');
  * her latest known pitch `last` (identical to planner.run).
Predictive distribution over her MIDI note m during beat k (time-weighted, like M3):
    P(m) ∝ Pc(pc(m))^a * Ps(m - last)^b
    Pc(pc) = sum_z prior[z] * B[z, pc]            (HMM emission = time-weighted pc share)
    Ps(d)  = time-weighted interval histogram, learned on the TRAINING split at the same decision time
Duet's note h (alto range, diatonic, strictly below `last`) maximises
    E_P[ |m-h| % 12 in CONSONANT ] + eta * P(pc(h) in chord)  - c_move*|h-h_prev| - c_mid*|h-ALTO_MID|
Weights are tuned on the validation split only; the test split is scored once per variant.
Writes exp_consonance/results.json. Run:  ../.venv/bin/python exp_consonance/run_consonance.py  (from duet_beat/)
"""
import copy
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BEAT = os.path.dirname(HERE)
sys.path.insert(0, BEAT)
from data import load  # noqa: E402
from metrics import bootstrap, paired_diff, song_counts, summarize  # noqa: E402
from model import ChordHMM  # noqa: E402
from planner import Config, ScoreSource, run, to_obs  # noqa: E402
from theory import ALTO, ALTO_MID, CONSONANT, CT, MAJOR, N_CHORDS  # noqa: E402

OUT = os.path.join(HERE, 'results.json')
DMAX = 14
CONS = np.array([int(i % 12 in CONSONANT) for i in range(200)])
CHORD_MEMBER = np.array([[int(pc in CT[z]) for pc in range(12)] for z in range(N_CHORDS)])  # (7, 12)


@dataclass
class Beat:                      # the fields metrics.song_counts reads
    k: int
    z_hat: int = -1
    h: int = -1
    extra: dict = field(default_factory=dict)


def decision_states(song, model, cfg):
    """Yield (k, prior over chords for beat k, her latest known midi) exactly as planner.run does."""
    source = ScoreSource(song)
    t, M, K = song['tonic'], song['M'], song['K']
    pos = [b['pos'] for b in song['beats']]
    D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)

    def ll(vec, mid):
        return model.loglik(to_obs(vec, mid, t, cfg.mode), cfg.mode, cfg.lam)

    pi_full, alpha_full = [], []
    for j in range(K):
        p = model.init if j == 0 else model.predict(alpha_full[j - 1], pos[j], M)
        pi_full.append(p)
        alpha_full.append(model.update(p, ll(*source.full(j))))
    her_last = None
    for k in range(K):
        when = k - D
        jstar = math.floor(when)
        frac = when - jstar
        if jstar >= 0:
            vec, mid = source.partial(jstar, frac)
            if mid is None and frac < 1:
                for jj in range(jstar - 1, -1, -1):
                    _, mid = source.full(jj)
                    if mid is not None:
                        break
            her_last = mid if mid is not None else her_last
        if jstar < 0:
            prior = pi_full[k]
        else:
            post = model.update(pi_full[jstar], ll(*source.partial(jstar, frac)))
            for i in range(jstar + 1, k + 1):
                post = model.predict(post, pos[i], M)
            prior = post
        yield k, prior, her_last


def her_notes(sop, k):
    """[(midi, overlap)] of her notes inside beat k (what M3 weighs)."""
    out = []
    for on, off, m in sop:
        if on >= k + 1:
            break
        ov = min(off, k + 1) - max(on, k)
        if ov > 1e-9:
            out.append((m, ov))
    return out


def learn_step_prior(train, model, cfg, kappa=0.5):
    """Ps(d): time-weighted histogram of (her note in beat k) - (her latest known note at decision time)."""
    h = np.zeros(2 * DMAX + 1)
    for s in train:
        for k, _, last in decision_states(s, model, cfg):
            if last is None:
                continue
            for m, ov in her_notes(s['sop'], k):
                d = int(np.clip(m - last, -DMAX, DMAX))
                h[d + DMAX] += ov
    return (h + kappa) / (h + kappa).sum()


@dataclass
class P:
    a: float = 1.0          # exponent of the chord-belief pitch-class term
    b: float = 1.0          # exponent of the melodic-step term
    eta: float = 0.0        # bonus for expected chord-tone membership (M2-aware)
    c_move: float = 0.01    # per semitone of Duet's leap
    c_mid: float = 0.002    # per semitone from the middle of the alto range
    below: bool = True      # strictly below her latest note (no unison / crossing)
    J: int = 8


def choose(prior, last, h_prev, tonic, model, Ps, p):
    """Return (h, expected consonance of h, argmax chord)."""
    Pc = prior @ model.B                                      # (12,) relative pitch class of her note
    ms = np.arange(last - DMAX, last + DMAX + 1)
    w = (Pc[(ms - tonic) % 12] ** p.a) * (Ps ** p.b)
    w = w / w.sum()
    lo, hi = ALTO
    hs = np.array([h for h in range(lo, hi + 1) if (h - tonic) % 12 in MAJOR])
    if p.below:
        sel = hs <= last - 1
        if sel.any():
            hs = hs[sel]
    ec = (w[None, :] * CONS[np.abs(ms[None, :] - hs[:, None])]).sum(1)
    member = prior @ CHORD_MEMBER                             # P(pc in chord) for each relative pc
    score = ec + p.eta * member[(hs - tonic) % 12] - p.c_mid * np.abs(hs - ALTO_MID)
    if h_prev is not None:
        score = score - p.c_move * np.abs(hs - h_prev)
    i = int(np.argmax(score))
    return int(hs[i]), float(ec[i]), int(np.argmax(prior))


def run_exp(song, model, cfg, Ps, p):
    out, h_prev = [], None
    for k, prior, last in decision_states(song, model, cfg):
        bt = Beat(k=k, z_hat=int(np.argmax(prior)))
        if k >= p.J and last is not None:
            h, ec, _ = choose(prior, last, h_prev, song['tonic'], model, Ps, p)
            bt.h, bt.extra = h, {'ec': ec}
            h_prev = h
        out.append(bt)
    return out


def evaluate(songs, model, cfg, Ps, p):
    return [song_counts(s, run_exp(s, model, cfg, Ps, p), p.J) for s in songs]


def tune(val, model, cfg, Ps, grid, base):
    rows, best = [], None
    for vals in itertools.product(*grid.values()):
        p = P(**{**base, **dict(zip(grid, vals))})
        sm = summarize(evaluate(val, model, cfg, Ps, p))
        rows.append({**dict(zip(grid, vals)), **{k: sm[k] for k in ('m1', 'm2', 'm3', 'm4')}})
        if best is None or sm['m3'] > best[0]:
            best = (sm['m3'], p)
    return best[1], rows


def causality_check(songs, model, cfg, Ps, p):
    """Changing her melody after a decision time must not change that decision."""
    D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
    n = 0
    for s in songs[:10]:
        k0 = s['K'] // 2
        ref = run_exp(s, model, cfg, Ps, p)
        alt = copy.deepcopy(s)
        cut = k0 - D
        for nt in alt['sop']:
            if nt[0] >= cut:
                nt[2] += 5
            elif nt[1] > cut:                       # split the note straddling the cut: keep the heard part
                pass
        alt['sop'] = [nt for nt in alt['sop']]
        # only notes starting after the cut changed; a note straddling it keeps its pitch (heard already)
        new = run_exp(alt, model, cfg, Ps, p)
        for a, b in zip(ref[:k0 + 1], new[:k0 + 1]):
            assert (a.h, a.z_hat) == (b.h, b.z_hat), (s['id'], a.k)
            n += 1
    return n


def main():
    d = load()['split']
    model = ChordHMM().fit(d['train'])
    train, val, test = d['train'], d['val'], d['test']
    with open(os.path.join(BEAT, 'out', 'results.json')) as fh:
        old = json.load(fh)
    cfg = Config(**{k: (tuple(v) if k == 'alto' else v) for k, v in old['results']['hmm_long_N4']['cfg'].items()})
    Ps = learn_step_prior(train, model, cfg)
    print('step prior (train) P(d=0) %.3f, P(|d|<=2) %.3f' % (Ps[DMAX], Ps[DMAX - 2:DMAX + 3].sum()))

    variants = {
        # main idea: chord belief x melodic step, pure expected consonance
        'EC_chord_x_step': ({'a': (0.5, 1.0, 2.0, 3.0), 'c_move': (0.0, 0.005, 0.01, 0.02, 0.04)}, {'b': 1.0, 'eta': 0.0}),
        # chord belief only (no melodic model)
        'EC_chord_only': ({'a': (1.0,), 'c_move': (0.0, 0.005, 0.01, 0.02, 0.04)}, {'b': 0.0, 'eta': 0.0}),
        # melodic step only (no harmony model at all)
        'EC_step_only': ({'a': (0.0,), 'c_move': (0.0, 0.005, 0.01, 0.02, 0.04)}, {'b': 1.0, 'eta': 0.0}),
        # main idea + bonus for being a likely chord tone (keeps M2 up)
        'EC_chord_x_step_+M2': ({'a': (1.0, 2.0), 'c_move': (0.005, 0.01, 0.02), 'eta': (0.1, 0.2, 0.4)}, {'b': 1.0}),
    }
    res, counts, tuning = {}, {}, {}
    ref = {k: run(s, model, cfg, 'hmm') for k, s in []}  # placeholder (unused)
    counts['hmm_long_N4'] = [song_counts(s, run(s, model, cfg, 'hmm'), cfg.J) for s in test]
    counts['B0_tonic'] = [song_counts(s, run(s, model, Config(**{**cfg.__dict__, 'q_min': 0.0}), 'tonic'), cfg.J) for s in test]
    counts['hmm_long_N1'] = [song_counts(s, run(s, model, Config(**{**cfg.__dict__, 'N': 1}), 'hmm'), cfg.J) for s in test]
    for key in ('hmm_long_N4', 'hmm_long_N1', 'B0_tonic'):
        res[key] = {**summarize(counts[key]), 'ci': bootstrap(counts[key]), 'label': 'existing system (re-run)'}
    for name, (grid, base) in variants.items():
        p, rows = tune(val, model, cfg, Ps, grid, base)
        tuning[name] = {'chosen': p.__dict__, 'grid': rows}
        cs = evaluate(test, model, cfg, Ps, p)
        counts[name] = cs
        res[name] = {**summarize(cs), 'ci': bootstrap(cs), 'params': p.__dict__,
                     'val_m3': max(r['m3'] for r in rows)}
    diffs = {}
    for name in list(variants) + ['hmm_long_N1']:
        for b in ('hmm_long_N4', 'B0_tonic'):
            for k in ('m3', 'm2'):
                diffs[f'{name}-{b}:{k}'] = paired_diff(counts[name], counts[b], k)
    best_name = max(variants, key=lambda n: tuning[n] and res[n]['val_m3'])
    nchk = causality_check(test, model, cfg, Ps, P(**tuning[best_name]['chosen']))
    print(f'causality ok on {nchk} decisions ({best_name})')
    for k, v in res.items():
        print(f"{k:22s} M1 {v['m1']:.3f}  M2 {v['m2']:.3f}  M3 {v['m3']:.3f} [{v['ci']['m3'][0]:.3f}, {v['ci']['m3'][1]:.3f}]"
              f"  M4 {v['m4']:.2f}  M5 {v['m5']:.3f}  {v.get('params', '')}")
    for k, v in diffs.items():
        print(f'{k:40s} {v[0]:+.3f}  [{v[1][0]:+.3f}, {v[1][1]:+.3f}]')
    with open(OUT, 'w') as fh:
        json.dump({'results': res, 'diffs': diffs, 'tuning': tuning, 'step_prior': Ps.round(5).tolist(),
                   'cfg': cfg.__dict__, 'causality_checked_decisions': nchk}, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
