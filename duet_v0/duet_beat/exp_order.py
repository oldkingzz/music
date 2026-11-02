"""Does the chord HMM get better if the next chord depends on the last 2 or 3 chords, not just 1?

An order-n model keeps its belief over the last n chords (a joint state; 8^n states with a
'start' symbol for the first beats of a song). Its next-chord table P(z_k | z_{k-n..k-1}, bar
position) is smoothed by backing off to the order n-1 table (Dirichlet backoff, weight beta), so
order 1 is exactly the main model. beta is chosen on 20 held-out training chorales, then the model
is refit on all 135. Duet's note is still planned by planner.Planner from the marginal chord odds.

Reports, on the 27 test chorales and Twinkle:
  next-chord log-loss   -log P(true next chord | true earlier chords), the pure 'grammar' test
  chord@decision (m1)   live, causal, as planner.run 'hmm'
  consonant (m3)        live, as planner.run 'hmm'
  us/beat               time for one predict + update step
Writes duet_beat/out/order.json.  Run:  .venv/bin/python duet_beat/exp_order.py
"""
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load  # noqa: E402
from metrics import song_counts, summarize  # noqa: E402
from model import ChordHMM  # noqa: E402
from planner import Beat, Config, Planner, ScoreSource, to_obs  # noqa: E402
from songs import twinkle  # noqa: E402
from theory import CT, N_CHORDS  # noqa: E402

S, START = N_CHORDS, N_CHORDS
A = S + 1                                            # history alphabet: 7 chords + start
KEYS = [('A4', p) for p in range(4)] + [('A2', 0), ('A2', 1)]


class OrderN:
    def __init__(self, base: ChordHMM, n, beta=5.0):
        self.base, self.n, self.beta = base, n, beta

    def fit(self, chorales):
        n = self.n
        self.cnt = {m: {} for m in range(2, n + 1)}     # (key, ctx tuple of length m) -> counts[7]
        for r in chorales:
            zs = [b['z'] for b in r['beats']]
            for i in range(1, len(zs)):
                b = zs[i]
                if b < 0:
                    continue
                hist = [START] * n + zs[:i]
                key = self.base.trans_key(r['beats'][i]['pos'], r['M'])
                keys = [key] + ([('A2', 0 if key[1] == 0 else 1)] if key[0] == 'A4' else [])
                for m in range(2, n + 1):
                    ctx = tuple(hist[-m:])
                    if any(c < 0 for c in ctx) or ctx[-1] == START:
                        continue
                    for kk in keys:
                        self.cnt[m].setdefault((kk, ctx), np.zeros(S))[b] += 1
        self._build()
        return self

    def prob(self, key, ctx):
        """P(next chord | ctx) with ctx the last len(ctx) chords, oldest first."""
        last = ctx[-1]
        if last == START:
            return self.base.init
        p = self.base.mat(key)[last]
        for m in range(2, len(ctx) + 1):
            sub = ctx[-m:]
            c = self.cnt[m].get((key, sub))
            if c is not None:
                p = (c + self.beta * p) / (c.sum() + self.beta)
        return p

    def _build(self):
        n = self.n
        self.NS = A ** n
        st = np.array(np.unravel_index(np.arange(self.NS), (A,) * n)).T   # oldest first
        self.last = st[:, -1]
        self.nxt = (np.arange(self.NS)[:, None] % (A ** (n - 1))) * A + np.arange(S)[None, :]
        self.T = {}
        for key in KEYS:
            self.T[key] = np.array([self.prob(key, tuple(s)) for s in st])
        self.init = np.zeros(self.NS)
        self.init[[A ** n - 1 - (A - 1) + z for z in range(S)]] = self.base.init  # (start.., z)
        self.live = self.last < S

    def predict(self, b, pos, M):
        T = self.T[self.base.trans_key(pos, M)]
        return np.bincount(self.nxt.ravel(), (b[:, None] * T).ravel(), minlength=self.NS)

    def update(self, b, ll):
        w = np.where(self.live, b * np.exp(ll - ll.max())[np.minimum(self.last, S - 1)], 0.0)
        return w / w.sum()

    def marg(self, b):
        return np.bincount(self.last[self.live], b[self.live], minlength=S)[:S]


def logloss(f, songs):
    tot, n = 0.0, 0
    for r in songs:
        zs = [b['z'] for b in r['beats']]
        for i in range(1, len(zs)):
            if zs[i] < 0:
                continue
            hist = [START] * f.n + [z if z >= 0 else START for z in zs[:i]]
            p = f.prob(f.base.trans_key(r['beats'][i]['pos'], r['M']), tuple(hist[-f.n:]))
            tot -= math.log(p[zs[i]])
            n += 1
    return tot / n


def run_order(song, f, cfg):
    """planner.run(system='hmm') with the belief kept by f (an OrderN)."""
    src = ScoreSource(song)
    model = f.base
    t, M, K = song['tonic'], song['M'], song['K']
    pos = [b['pos'] for b in song['beats']]
    D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
    planner = Planner(t, cfg, model)
    ll = lambda vec, mid: model.loglik(to_obs(vec, mid, t, cfg.mode), cfg.mode, cfg.lam)  # noqa: E731
    pi_full, alpha_full = [], []
    for j in range(K):
        p = f.init if j == 0 else f.predict(alpha_full[j - 1], pos[j], M)
        pi_full.append(p)
        alpha_full.append(f.update(p, ll(*src.full(j))))
    out, joined, q_prev, h_last, her_last = [], False, 0.0, None, None
    for k in range(K):
        when = k - D
        jstar = math.floor(when)
        frac = when - jstar
        if jstar >= 0:
            vec, mid = src.partial(jstar, frac)
            if mid is None and frac < 1:
                for jj in range(jstar - 1, -1, -1):
                    _, mid = src.full(jj)
                    if mid is not None:
                        break
            her_last = mid if mid is not None else her_last
        if jstar < 0:
            post = pi_full[k]
        else:
            post = f.update(pi_full[jstar], ll(*src.partial(jstar, frac)))
            for i in range(jstar + 1, k + 1):
                post = f.predict(post, pos[i], M)
        prior = f.marg(post)
        bt = Beat(k=k, pos=pos[k], z_true=song['beats'][k]['z'])
        horizon = [model.trans_key(pos[i], M) for i in range(k + 1, min(K, k + cfg.N))]
        z0, h0, path = planner.plan(prior, horizon, h_last, her_last)
        q = float(prior[z0])
        bt.z_hat, bt.q = z0, q
        if not joined and k >= cfg.J and q >= cfg.q_min and q_prev >= cfg.q_min:
            joined = True
        if joined:
            if q >= cfg.q_min:
                bt.action, bt.h = 'sing', h0
            elif h_last is not None and (h_last - t) % 12 in CT[z0]:
                bt.action, bt.h = 'hold', h_last
            else:
                bt.action = 'silent'
            if bt.h >= 0:
                h_last = bt.h
        q_prev = q
        out.append(bt)
    return out


def step_time(f, reps=2000):
    b = f.init.copy()
    ll = np.log(np.full(S, 1 / S))
    t0 = time.perf_counter()
    for i in range(reps):
        b = f.update(f.predict(b, i % 4, 4), ll)
    return (time.perf_counter() - t0) / reps * 1e6


def main():
    d = load()['split']
    res = json.load(open(os.path.join(HERE, 'out', 'results.json')))
    base = dict(res['tuning']['chosen']['main'])
    base['alto'] = tuple(base['alto'])
    cfg = Config(**base)
    tr, va = d['train'][:-20], d['train'][-20:]
    out = {}
    for n in (1, 2, 3):
        betas = [None] if n == 1 else [0.5, 2, 5, 20, 50]
        best = None
        for beta in betas:
            f = OrderN(ChordHMM().fit(tr), n, beta or 1.0).fit(tr)
            v = logloss(f, va)
            if best is None or v < best[1]:
                best = (beta, v)
        beta = best[0] or 1.0
        f = OrderN(ChordHMM().fit(d['train']), n, beta).fit(d['train'])
        row = {'beta': best[0], 'states': f.NS, 'logloss_test': logloss(f, d['test']),
               'us_per_beat': step_time(f)}
        for name, songs in (('bach_test', d['test']), ('twinkle', [twinkle()])):
            cs = []
            for s in songs:
                c = replace(cfg, bpm=s.get('bpm', cfg.bpm)) if name != 'bach_test' else cfg
                cs.append(song_counts(s, run_order(s, f, c), cfg.J))
            sm = summarize(cs)
            row[name] = {'m1': sm['m1'], 'm3': sm['m3']}
        out[n] = row
        print(f"order {n}: beta {best[0]}  states {f.NS:4d}  log-loss {row['logloss_test']:.3f}  "
              f"{row['us_per_beat']:7.1f} us/beat   Bach chord {row['bach_test']['m1']:.3f} "
              f"consonant {row['bach_test']['m3']:.3f}   Twinkle chord {row['twinkle']['m1']:.3f} "
              f"consonant {row['twinkle']['m3']:.3f}")
    with open(os.path.join(HERE, 'out', 'order.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
