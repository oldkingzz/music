"""D4 / A4.3 chord-prediction leaderboard: the whole (clean) melody of a song -> the chord z of every beat.

Systems (all scored on every beat with a label z >= 0, clean melody, offline):
  hmm_causal   baseline 1: the PDR ChordHMM (Bach train, mode 'long') through duet_beat/planner.run
               exactly as the realtime system uses it (causal, decided L+delta before each beat).
               Also reported under the bench protocol (10 % misheard notes, from beat J=8) for reference.
  hmm_fb       baseline 2: the same ChordHMM, offline forward-backward (posterior argmax).
  hmm_vit      baseline 2b: the same ChordHMM, offline Viterbi.
  style_hmm    ours: one HMM per style (trained on that style's train split), duration-weighted
               emissions (+ note-onset weight), separate emissions on downbeats, positional transitions;
               lam / w_on / kappa / decoder tuned on val.
  bigru        ours: 2-layer bidirectional GRU on the union of train splits with a style id (trained on
               MPS by gru_train.py), inference in numpy.
  bigru_vit    ours: BiGRU log-posteriors + the style HMM's transitions, Viterbi (beta tuned on val).
  ens          ours: BiGRU log-posteriors + w x style-HMM emissions + transitions, Viterbi (w, beta on val).
  The system reported is the one with the best mean VAL accuracy; test is run once with it.

  .venv/bin/python futures/d4_styles/chord_board.py            # val only (tuning)
  .venv/bin/python futures/d4_styles/chord_board.py --test     # the ONE final test run (refuses twice)
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feats import STYLES, dur_onset, features, labels, load_all  # noqa: E402

import bench  # noqa: E402  (duet_beat; only config() and heard() are used: nothing is written there)
from model import ChordHMM  # noqa: E402
from planner import run as planner_run  # noqa: E402

S = 7
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ scoring helpers
def counts(pred, y):
    m = y >= 0
    return int((np.asarray(pred)[m] == y[m]).sum()), int(m.sum())


def pooled(c, n):
    return float(np.sum(c) / max(np.sum(n), 1))


def boot_gain(ca, cb, n, B=2000, seed=5160):
    """Replicates of pooled-accuracy gain a - b, resampling songs (paired)."""
    ca, cb, n = map(np.asarray, (ca, cb, n))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), (B, len(n)))
    nn = n[idx].sum(1)
    return (ca[idx].sum(1) - cb[idx].sum(1)) / np.maximum(nn, 1)


def ci(rep):
    return [float(np.percentile(rep, 2.5)), float(np.percentile(rep, 97.5))]


def _selftest_scoring():
    y = np.array([0, 1, -1, 4])
    assert counts([0, 1, 3, 4], y) == (3, 3)
    assert counts([1, 1, 0, 0], y) == (1, 3)
    ca, cb, n = [3, 2, 5], [3, 2, 5], [3, 4, 5]
    r = boot_gain(ca, cb, n)
    assert np.allclose(r, 0)
    r = boot_gain([3, 4, 5], [0, 0, 0], [3, 4, 5])
    assert np.allclose(r, 1)
    r = boot_gain([1, 0], [0, 0], [1, 1], B=4000)        # gain 0.5 on average, CI spans 0 .. 1
    assert abs(r.mean() - 0.5) < 0.05 and ci(r)[0] == 0 and ci(r)[1] == 1
    song = {'K': 3, 'tonic': 2, 'sop': [[0.0, 2.0, 62], [2.5, 3.0, 66]], 'M': 4,
            'beats': [{'pos': 0, 'z': 0}, {'pos': 1, 'z': 4}, {'pos': 2, 'z': -1}]}
    d, o = dur_onset(song)
    assert d[0, 0] == 1 and d[1, 0] == 1 and abs(d[2, 4] - 0.5) < 1e-9 and o[0, 0] == 1 and o[2, 4] == 1 and o[1].sum() == 0
    X = features(song, 'pop')
    assert X.shape == (3, 39) and X[0, 29] == 1 and X[1, 29] == 0 and X[0, 34 + 2] == 1


# ------------------------------------------------------------------ HMM decoders (log space)
def trans_mats(model_or_hmm, song):
    pos, M = [b['pos'] for b in song['beats']], song['M']
    return [model_or_hmm.trans(pos[i], M) for i in range(1, song['K'])]


def fb(logp0, LL, A):
    """Forward-backward posteriors. logp0 (S,), LL (K,S) log-likelihoods, A list of K-1 (S,S)."""
    K = len(LL)
    al = np.zeros((K, S))
    a = np.exp(logp0 + LL[0] - (logp0 + LL[0]).max())
    al[0] = a / a.sum()
    for k in range(1, K):
        e = np.exp(LL[k] - LL[k].max())
        a = (al[k - 1] @ A[k - 1]) * e
        al[k] = a / a.sum()
    be = np.ones(S)
    post = np.zeros((K, S))
    post[-1] = al[-1]
    for k in range(K - 2, -1, -1):
        e = np.exp(LL[k + 1] - LL[k + 1].max())
        be = A[k] @ (e * be)
        be /= be.sum()
        p = al[k] * be
        post[k] = p / p.sum()
    return post


def viterbi(logp0, LL, logA):
    K = len(LL)
    d = logp0 + LL[0]
    back = np.zeros((K, S), int)
    for k in range(1, K):
        t = d[:, None] + logA[k - 1]
        back[k] = t.argmax(0)
        d = t.max(0) + LL[k]
    z = np.zeros(K, int)
    z[-1] = int(d.argmax())
    for k in range(K - 1, 0, -1):
        z[k - 1] = back[k, z[k]]
    return z


def _selftest_hmm():
    A = [np.full((S, S), 1 / S)] * 2
    LL = np.log(np.eye(S)[[3, 3, 5]] * 0.9 + 0.1 / S)
    p = fb(np.log(np.full(S, 1 / S)), LL, A)
    assert np.allclose(p.sum(1), 1) and list(p.argmax(1)) == [3, 3, 5]
    assert list(viterbi(np.log(np.full(S, 1 / S)), LL, [np.log(a) for a in A])) == [3, 3, 5]
    sticky = np.full((S, S), 0.01)
    np.fill_diagonal(sticky, 1 - 0.06)
    LL2 = np.log(np.eye(S)[[3, 5, 3]] * 0.6 + 0.4 / S)   # a weak blip in the middle is smoothed away
    assert list(viterbi(np.zeros(S), LL2, [np.log(sticky)] * 2)) == [3, 3, 3]


# ------------------------------------------------------------------ baseline HMM (PDR ChordHMM)
def pdr_ll(model, song, mode='long', lam=1.0):
    from model import soprano_obs
    t = song['tonic']
    LL = np.zeros((song['K'], S))
    for k in range(song['K']):
        vec, mid = soprano_obs(song['sop'], k)
        if mode == 'long':
            obs = None if mid is None else (mid - t) % 12
        else:
            v = np.roll(vec, -t)
            obs = v if v.sum() > 0 else None
        LL[k] = model.loglik(obs, mode, lam)
    return LL


def clean_input(song):
    return dict(song, beats=[{'pos': b['pos'], 'z': -1, 'onset_pcs': []} for b in song['beats']])


# ------------------------------------------------------------------ style HMM (ours)
class StyleHMM:
    def __init__(self, kappa=1.0, w_on=0.5):
        self.kappa, self.w_on = kappa, w_on

    def obs(self, song):
        d, o = dur_onset(song)
        return np.minimum(d, 1.0) + self.w_on * o

    def fit(self, songs):
        k = self.kappa
        init, A4, A2, em = np.zeros(S), np.zeros((4, S, S)), np.zeros((2, S, S)), np.zeros((2, S, 12))
        for s in songs:
            zs, pos, M = labels(s), [b['pos'] for b in s['beats']], s['M']
            v = self.obs(s)
            f = next((z for z in zs if z >= 0), None)
            if f is not None:
                init[f] += 1
            for i in range(1, len(zs)):
                a, b = zs[i - 1], zs[i]
                if a < 0 or b < 0:
                    continue
                if M == 4:
                    A4[pos[i] % 4, a, b] += 1
                A2[0 if pos[i] == 0 else 1, a, b] += 1
            for i, z in enumerate(zs):
                if z >= 0:
                    em[0 if pos[i] == 0 else 1, z] += v[i]
        self.init = (init + k) / (init.sum() + S * k)
        self.A4 = (A4 + k) / (A4.sum(-1, keepdims=True) + S * k)
        self.A2 = (A2 + k) / (A2.sum(-1, keepdims=True) + S * k)
        self.logB = np.log((em + k) / (em.sum(-1, keepdims=True) + 12 * k))
        return self

    def trans(self, pos, M):
        return self.A4[pos % 4] if M == 4 else self.A2[0 if pos % M == 0 else 1]

    def ll(self, song, lam):
        v = self.obs(song)
        pos = np.array([b['pos'] for b in song['beats']])
        return lam * np.where((pos == 0)[:, None], v @ self.logB[0].T, v @ self.logB[1].T)

    def decode(self, song, lam, how='fb', extra_ll=None, beta=1.0):
        LL = self.ll(song, lam) if extra_ll is None else extra_ll
        A = trans_mats(self, song)
        if how == 'fb':
            return fb(np.log(self.init), LL, A).argmax(1)
        return viterbi(beta * np.log(self.init), LL, [beta * np.log(a) for a in A])


# ------------------------------------------------------------------ BiGRU in numpy
class NpBiGRU:
    def __init__(self, path):
        w = np.load(path)
        self.w = {k: w[k] for k in w.files}
        self.H = int(self.w['H'])
        self.layers = sum(1 for k in self.w if k.startswith('gru.weight_ih_l') and not k.endswith('reverse'))

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _run(self, X, sfx):
        w, H = self.w, self.H
        Wi, Wh, bi, bh = w['gru.weight_ih' + sfx], w['gru.weight_hh' + sfx], w['gru.bias_ih' + sfx], w['gru.bias_hh' + sfx]
        G = X @ Wi.T + bi
        h = np.zeros(H, np.float32)
        out = np.zeros((len(X), H), np.float32)
        for t in range(len(X)):
            gh = Wh @ h + bh
            r = self._sig(G[t, :H] + gh[:H])
            z = self._sig(G[t, H:2 * H] + gh[H:2 * H])
            n = np.tanh(G[t, 2 * H:] + r * gh[2 * H:])
            h = (1 - z) * n + z * h
            out[t] = h
        return out

    def logits(self, X):
        w = self.w
        h = np.tanh(X @ w['inp.weight'].T + w['inp.bias'])
        for l in range(self.layers):
            f = self._run(h, f'_l{l}')
            b = self._run(h[::-1], f'_l{l}_reverse')[::-1]
            h = np.concatenate([f, b], 1)
        return h @ w['out.weight'].T + w['out.bias']

    def logp(self, X):
        L = self.logits(X)
        L = L - L.max(1, keepdims=True)
        return L - np.log(np.exp(L).sum(1, keepdims=True))


def _selftest_gru(path):
    import torch
    from gru_train import Net
    g = NpBiGRU(path)
    net = Net(g.H)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in g.w.items() if k != 'H'})
    net.eval()
    X = np.random.default_rng(0).random((17, 39)).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.from_numpy(X[None]))[0].numpy()
    err = float(np.abs(ref - g.logits(X)).max())
    assert err < 1e-4, err
    return err


# ------------------------------------------------------------------ the board
def evaluate(split, data, gru_path, tuned=None, log=print):
    cfg = bench.config()
    pdr = ChordHMM().fit(data['bach']['train'])
    res = {}          # system -> ds -> (c list, n list)

    def add(sysname, ds, c, n):
        res.setdefault(sysname, {}).setdefault(ds, ([], []))
        res[sysname][ds][0].append(c)
        res[sysname][ds][1].append(n)

    t0 = time.time()
    for ds in STYLES:
        for s in data[ds][split]:
            y = labels(s)
            c = replace(cfg, bpm=s['bpm']) if 'bpm' in s else cfg
            bt = planner_run(clean_input(s), pdr, c, 'hmm')
            add('hmm_causal', ds, *counts([b.z_hat for b in bt], y))
            bt2 = planner_run(bench.heard(s), pdr, c, 'hmm')          # bench protocol: noise, from J
            yj = y.copy()
            yj[:cfg.J] = -1
            add('hmm_causal_benchproto', ds, *counts([b.z_hat for b in bt2], yj))
            LL = pdr_ll(pdr, s)
            A = trans_mats(pdr, s)
            add('hmm_fb', ds, *counts(fb(np.log(pdr.init), LL, A).argmax(1), y))
            add('hmm_vit', ds, *counts(viterbi(np.log(pdr.init), LL, [np.log(a) for a in A]), y))
    log(f'baselines {split} {time.time() - t0:.0f}s')

    # style HMM: tune per style on val (grid), then apply the chosen setting
    grid = [(kap, won, lam, how) for kap in (0.5, 2.0) for won in (0.0, 0.5) for lam in (0.5, 1.0, 2.0) for how in ('fb', 'vit')]
    if tuned is None:
        tuned = {'style_hmm': {}, 'bigru_vit_beta': None}
        for ds in STYLES:
            best, fitted = None, {}
            for kap, won, lam, how in grid:
                if (kap, won) not in fitted:
                    fitted[(kap, won)] = StyleHMM(kap, won).fit(data[ds]['train'])
                h = fitted[(kap, won)]
                c = n = 0
                for s in data[ds]['val']:
                    cc, nn = counts(h.decode(s, lam, how), labels(s))
                    c, n = c + cc, n + nn
                if best is None or c / n > best[0]:
                    best = (c / n, (kap, won, lam, how))
            tuned['style_hmm'][ds] = best[1]
            log(f'style_hmm {ds} val best {best[0]:.3f} with {best[1]}')
    hmms = {ds: StyleHMM(*tuned['style_hmm'][ds][:2]).fit(data[ds]['train']) for ds in STYLES}
    for ds in STYLES:
        kap, won, lam, how = tuned['style_hmm'][ds]
        for s in data[ds][split]:
            add('style_hmm', ds, *counts(hmms[ds].decode(s, lam, how), labels(s)))
    log(f'style_hmm {split} done {time.time() - t0:.0f}s')

    if gru_path and os.path.exists(gru_path):
        g = NpBiGRU(gru_path)
        lp = {ds: [g.logp(features(s, ds)) for s in data[ds][split]] for ds in STYLES}
        log(f'bigru inference {time.time() - t0:.0f}s')
        for ds in STYLES:
            for s, L in zip(data[ds][split], lp[ds]):
                add('bigru', ds, *counts(L.argmax(1), labels(s)))
        if tuned.get('bigru_vit_beta') is None:
            if split != 'val':
                raise RuntimeError('beta must be tuned on val first')
            best = None
            for beta in (0.0, 0.25, 0.5, 1.0):
                accs = []
                for ds in STYLES:
                    c = n = 0
                    for s, L in zip(data[ds][split], lp[ds]):
                        cc, nn = counts(hmms[ds].decode(s, 0, 'vit', extra_ll=L, beta=beta), labels(s))
                        c, n = c + cc, n + nn
                    accs.append(c / n)
                if best is None or np.mean(accs) > best[0]:
                    best = (float(np.mean(accs)), beta)
            tuned['bigru_vit_beta'] = best[1]
            log(f'bigru_vit beta {best[1]} val mean {best[0]:.3f}')
        for ds in STYLES:
            for s, L in zip(data[ds][split], lp[ds]):
                add('bigru_vit', ds, *counts(hmms[ds].decode(s, 0, 'vit', extra_ll=L, beta=tuned['bigru_vit_beta']), labels(s)))
        # ensemble: BiGRU log-posterior + w * style-HMM emission log-likelihood, Viterbi with the style transitions
        hl = {ds: [hmms[ds].ll(s, tuned['style_hmm'][ds][2]) for s in data[ds][split]] for ds in STYLES}
        if tuned.get('ens') is None:
            if split != 'val':
                raise RuntimeError('ensemble must be tuned on val first')
            best = None
            for w in (0.15, 0.3, 0.6):
                for beta in (0.25, 0.5, 1.0):
                    accs = []
                    for ds in STYLES:
                        c = n = 0
                        for s, L, H in zip(data[ds][split], lp[ds], hl[ds]):
                            cc, nn = counts(hmms[ds].decode(s, 0, 'vit', extra_ll=L + w * H, beta=beta), labels(s))
                            c, n = c + cc, n + nn
                        accs.append(c / n)
                    if best is None or np.mean(accs) > best[0]:
                        best = (float(np.mean(accs)), (w, beta))
            tuned['ens'] = best[1]
            log(f'ens (w, beta) {best[1]} val mean {best[0]:.3f}')
        w, beta = tuned['ens']
        for ds in STYLES:
            for s, L, H in zip(data[ds][split], lp[ds], hl[ds]):
                add('ens', ds, *counts(hmms[ds].decode(s, 0, 'vit', extra_ll=L + w * H, beta=beta), labels(s)))
    return res, tuned


def summarize(res, ours, base1='hmm_causal', base2=None):
    out = {'acc': {}, 'gain': {}}
    for sysname, per in res.items():
        out['acc'][sysname] = {ds: pooled(*per[ds]) for ds in per}
        out['acc'][sysname]['mean'] = float(np.mean([out['acc'][sysname][ds] for ds in STYLES]))
    for b in (base1, base2):
        g = {}
        reps = []
        for i, ds in enumerate(STYLES):
            ca, n = res[ours][ds]
            cb, nb = res[b][ds]
            assert nb == n
            r = boot_gain(ca, cb, n, seed=5160 + i)
            reps.append(r)
            g[ds] = {'gain': pooled(ca, n) - pooled(cb, n), 'ci95': ci(r)}
        m = np.mean(reps, 0)
        g['mean'] = {'gain': float(np.mean([g[ds]['gain'] for ds in STYLES])), 'ci95': ci(m)}
        out['gain'][f'{ours}_vs_{b}'] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--gru', default=os.path.join(HERE, 'models', 'bigru.npz'))
    a = ap.parse_args()
    _selftest_scoring()
    _selftest_hmm()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
    if os.path.exists(a.gru):
        log(f'numpy GRU vs torch max abs logit error {_selftest_gru(a.gru):.2e}')
    log('self-tests passed (scoring, bootstrap, features, HMM decoders, numpy GRU)')
    data = load_all()
    res, tuned = evaluate('val', data, a.gru, log=log)
    ours_all = [s for s in ('style_hmm', 'bigru', 'bigru_vit', 'ens') if s in res]
    val_acc = {s: float(np.mean([pooled(*res[s][ds]) for ds in STYLES])) for s in ours_all}
    best = max(val_acc, key=val_acc.get)
    base2 = max(('hmm_fb', 'hmm_vit'), key=lambda s: np.mean([pooled(*res[s][ds]) for ds in STYLES]))
    tuned.update({'chosen_on_val': best, 'base2_on_val': base2, 'val_mean_acc': val_acc})
    summ = summarize(res, best, base2=base2)
    summ['tuned'] = tuned
    summ['songs'] = {ds: len(data[ds]['val']) for ds in STYLES}
    summ['beats'] = {ds: int(np.sum(res[best][ds][1])) for ds in STYLES}
    json.dump(summ, open(os.path.join(OUT, 'board_val.json'), 'w'), indent=1, default=float)
    log(json.dumps(summ['acc'], default=lambda x: round(x, 4)))
    if a.test:
        tp = os.path.join(OUT, 'board_test.json')
        if os.path.exists(tp):
            raise SystemExit('test already run once: ' + tp)
        rt, _ = evaluate('test', data, a.gru, tuned=tuned, log=log)
        st = summarize(rt, best, base2=base2)
        st.update({'tuned': tuned, 'songs': {ds: len(data[ds]['test']) for ds in STYLES},
                   'beats': {ds: int(np.sum(rt[best][ds][1])) for ds in STYLES},
                   'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'note': 'single final test run; all choices made on val'})
        for o in ours_all:
            if o != best:
                st.setdefault('other_ours', {})[o] = summarize(rt, o, base2=base2)['gain']
        json.dump(st, open(tp, 'w'), indent=1, default=float)
        log('TEST ' + json.dumps(st['acc'], default=lambda x: round(x, 4)))
        log('TEST gains ' + json.dumps(st['gain'], default=lambda x: round(x, 4)))
    open(os.path.join(OUT, 'board_log.txt'), 'a').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
