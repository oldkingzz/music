"""Train the harmony CRF (and the simpler count-based baselines) on the Bach chorales.

    .venv/bin/python harmony/train.py [--voice alto]

Objective: conditional log-likelihood of Bach's line given the soprano, with an L2 prior,
maximised by L-BFGS (scipy). Positions where the true voice is resting or lies outside the
state range are left unlabeled (the numerator marginalises over them), so no chorale is
dropped. The prior strength sigma is chosen on the validation split (tune-disjoint from
train and test); the test split is not touched here.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from chorales import to_notes                                   # noqa: E402
from dataset import load_split                                  # noqa: E402
from model import (VOICE_COL, VOICE_RANGE, HarmonyModel, Templates, context,  # noqa: E402
                   detect_key_melody, frame_shift)


# ---------------------------------------------------------------- data
def soprano_median(recs):
    v = [r[2] + frame_shift(rec['tonic']) for rec in recs for r in rec['notes']]
    return float(np.median(v))


def sequences(recs, tpl, voice, sop_median, key_source='detected'):
    """One dict per chorale: index tables, gold labels (-1 = unlabeled), context."""
    col = VOICE_COL[voice]
    out = []
    for rec in recs:
        notes = to_notes(rec)
        if key_source == 'detected':
            tonic, mode = detect_key_melody(notes)
        else:
            tonic, mode = rec['tonic'], rec['mode']
        keep, c, shift = context(notes, tonic, mode, sop_median)
        gold = []
        for k in keep:
            v = rec['notes'][k][col]
            iv = None if v is None else rec['notes'][k][2] - v
            gold.append(iv - tpl.lo if iv is not None and tpl.lo <= iv <= tpl.hi else -1)
        U, P = tpl.index(c)
        out.append({'U': U, 'P': P, 'y': np.array(gold), 'c': c, 'keep': keep, 'rec': rec,
                    'notes': notes, 'key': (tonic, mode), 'shift': shift})
    return out


# ---------------------------------------------------------------- CRF
def _fb(u, p, mask=None):
    """Forward-backward in scaled probability space. Returns logZ, unary and pair marginals."""
    T, K = u.shape
    if mask is not None:
        u = np.where(mask, u, -np.inf)
    um = u.max(1, keepdims=True)
    eu = np.exp(u - um)
    if T > 1:
        pm = p.max((1, 2), keepdims=True)
        ep = np.exp(p - pm)
    a = np.zeros((T, K))
    b = np.ones((T, K))
    z = np.zeros(T)
    a[0] = eu[0]
    z[0] = a[0].sum()
    a[0] /= z[0]
    for t in range(1, T):
        a[t] = (a[t - 1] @ ep[t - 1]) * eu[t]
        z[t] = a[t].sum()
        a[t] /= z[t]
    for t in range(T - 2, -1, -1):
        b[t] = ep[t] @ (eu[t + 1] * b[t + 1])
        b[t] /= z[t + 1]
    logZ = np.log(z).sum() + um.sum() + (pm.sum() if T > 1 else 0.0)
    q = a * b
    xi = None
    if T > 1:
        xi = a[:-1, :, None] * ep * (eu[1:] * b[1:])[:, None, :] / z[1:, None, None]
    return logZ, q, xi


def objective(w, seqs, W, sigma):
    f = -0.5 * np.dot(w, w) / sigma ** 2
    g = -w / sigma ** 2
    for sq in seqs:
        U, P, y = sq['U'], sq['P'], sq['y']
        u = w[U].sum(0)
        p = w[P].sum(0) if P.shape[1] else None
        mask = np.ones(u.shape, bool)
        lab = y >= 0
        mask[lab] = False
        mask[np.nonzero(lab)[0], y[lab]] = True
        lz_c, q_c, xi_c = _fb(u, p, mask)
        lz, q, xi = _fb(u, p)
        f += lz_c - lz
        dq = q_c - q
        g += np.bincount(U.reshape(len(U), -1).ravel(), np.tile(dq.ravel(), len(U)), W)
        if xi is not None and len(P):
            dxi = xi_c - xi
            g += np.bincount(P.reshape(len(P), -1).ravel(), np.tile(dxi.ravel(), len(P)), W)
    return -f, -g


def train_crf(seqs, tpl, sigma, maxiter=400, w0=None, verbose=False):
    w0 = np.zeros(tpl.W) if w0 is None else w0
    t0 = time.time()
    res = minimize(objective, w0, args=(seqs, tpl.W, sigma), jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'gtol': 1e-4})
    if verbose:
        print(f'  sigma={sigma}: nll={res.fun:.1f} iters={res.nit} {time.time() - t0:.0f}s {res.message}')
    return res.x, res


# ---------------------------------------------------------------- count baselines
def count_weights(seqs, tpl, alpha=1.0):
    """'HMM-like' product of two smoothed conditionals expressed in the CRF's weight space:
    log P(interval | mode, degree) + log P(harmony motion | melody motion). Everything else 0."""
    K = tpl.K
    sl = tpl.slices()
    w = np.zeros(tpl.W)
    cu = np.zeros(sl['iv|mode,deg'].stop - sl['iv|mode,deg'].start)
    cp = np.zeros(25 * 15)
    for sq in seqs:
        y, U, P = sq['y'], sq['U'], sq['P']
        for t in np.nonzero(y >= 0)[0]:
            cu[U[0, t, y[t]] - sl['iv|mode,deg'].start] += 1
        for t in range(len(y) - 1):
            if y[t] >= 0 and y[t + 1] >= 0:
                cp[P[0, t, y[t], y[t + 1]] - sl['dh|ds'].start] += 1
    cu = cu.reshape(-1, K) + alpha
    cp = cp.reshape(15, 25) + alpha
    w[sl['iv|mode,deg']] = np.log(cu / cu.sum(1, keepdims=True)).ravel()
    w[sl['dh|ds']] = np.log(cp / cp.sum(1, keepdims=True)).ravel()
    return w


def unigram_weights(seqs, tpl, alpha=1.0):
    w = count_weights(seqs, tpl, alpha)
    w[tpl.slices()['dh|ds']] = 0.0
    return w


def val_accuracy(model, seqs, **kw):
    ok = n = 0
    for sq in seqs:
        iv = model.decode(sq['c'], **kw) - model.tpl.lo
        lab = sq['y'] >= 0
        ok += int((iv[lab] == sq['y'][lab]).sum())
        n += int(lab.sum())
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voice', default='alto', choices=list(VOICE_RANGE))
    ap.add_argument('--sigmas', default='0.3,1,3')
    ap.add_argument('--key', default='detected', choices=['detected', 'music21'])
    ap.add_argument('--feature-set', default='default (11)',
                    help='a name from ablation.FEATURE_SETS (chosen on validation by ablation.py)')
    args = ap.parse_args()
    from ablation import FEATURE_SETS
    feats = FEATURE_SETS[args.feature_set]
    S = load_split()
    tpl = Templates(*VOICE_RANGE[args.voice], feats)
    med = soprano_median(S['train'])
    t0 = time.time()
    tr = sequences(S['train'], tpl, args.voice, med, args.key)
    va = sequences(S['val'], tpl, args.voice, med, 'detected')
    lab = np.concatenate([s['y'] for s in tr])
    print(f'{args.voice}: {len(tr)} train / {len(va)} val chorales, {len(lab)} train positions '
          f'({(lab < 0).mean() * 100:.2f} % unlabeled), W={tpl.W}, prep {time.time() - t0:.1f}s')
    base_cfg = {'voice': args.voice, 'iv_range': list(VOICE_RANGE[args.voice]), 'sop_median': med,
                'key_source': args.key, 'features': feats, 'feature_set': args.feature_set}
    log = {'voice': args.voice, 'n_train': len(tr), 'n_val': len(va), 'W': tpl.W, 'runs': []}
    tpl2 = Templates(*VOICE_RANGE[args.voice], FEATURE_SETS['hmm-like (2 templates)'])
    tr2 = sequences(S['train'], tpl2, args.voice, med, args.key)
    va2 = sequences(S['val'], tpl2, args.voice, med, 'detected')
    cfg2 = {**base_cfg, 'features': FEATURE_SETS['hmm-like (2 templates)'], 'feature_set': 'hmm-like (2 templates)'}
    for name, wfun in (('unigram', unigram_weights), ('count_hmm', count_weights)):
        m = HarmonyModel(wfun(tr2, tpl2), {**cfg2, 'duet': {'lam': 0.0, 'no_parallel': False}})
        acc = val_accuracy(m, va2)
        print(f'  {name}: val acc {acc:.3f}')
        log['runs'].append({'model': name, 'val_acc': acc})
        m.save(os.path.join(HERE, 'params', f'{name}_{args.voice}.npz'))
    best = None
    w = None
    for sigma in [float(x) for x in args.sigmas.split(',')]:
        w, res = train_crf(tr, tpl, sigma, w0=None, verbose=True)
        m = HarmonyModel(w, {**base_cfg, 'duet': {'lam': 0.0, 'no_parallel': False}})
        acc = val_accuracy(m, va)
        acc_c = val_accuracy(m, va, no_parallel=True)
        print(f'    val acc {acc:.3f} (with no-parallel constraint {acc_c:.3f})')
        log['runs'].append({'model': 'crf', 'sigma': sigma, 'val_acc': acc, 'val_acc_noparallel': acc_c,
                            'nll': float(res.fun), 'iters': int(res.nit)})
        if best is None or acc > best[0]:
            best = (acc, sigma, w)
    acc, sigma, w = best
    cfg = {**base_cfg, 'sigma': sigma, 'val_acc': acc,
           'duet': {'lam': 0.0, 'no_parallel': True, 'max_iv': None}}
    HarmonyModel(w, cfg).save(os.path.join(HERE, 'params', f'crf_{args.voice}.npz'))
    log['chosen_sigma'] = sigma
    with open(os.path.join(HERE, 'params', f'train_log_{args.voice}.json'), 'w') as fh:
        json.dump(log, fh, indent=1)
    print(f'saved crf_{args.voice}.npz (sigma={sigma}, val acc {acc:.3f}) total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
