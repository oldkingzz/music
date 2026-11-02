"""Tiny causal transformer that guesses her next beat, plus an expected-consonance chooser.

Per beat j a token is built from the HEARD melody in the window [j, j + 1 - D) only (the part of a
beat Duet has heard by the decision time of beat j + 1):
  longest pitch class (rel. tonic, 12 = none), pitch class sounding at the window end, onset bit,
  bar position.  For beat k the model reads the tokens of beats k-T .. k-1 (all of them are complete
  by the decision time k - D) plus the bar position of beat k and outputs a 12-way distribution over
  the pitch-class time share of her beat k.  Trained on train_sets() melodies (clean targets, inputs
  with 10 % random-pitch-class mishearing), ~1 min on the M4 GPU (MPS).

Chooser: Duet's note h (alto range, below her latest heard note) maximizes
    sum_pc q(pc) * val(interval(pc, h)) - vl |h - h_prev| - vc |h - alto mid|
with val = 1 for 3rds/5th/6ths, w_oct for unison/octave (kept low so Duet harmonizes rather than
doubles), 0 otherwise.  Optionally q is mixed with the per-song repeat cache (exp_ceiling.Repeat).
Causal by construction: every token for beat k ends at time <= k - D; the windows for all beats of a
song are evaluated in one batched forward call, which gives exactly what beat-by-beat calls give
(set DUET_TF='{"stream":1}' to run one forward call per beat and time it honestly).
"""
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from exp_ceiling import Repeat
from planner import Beat
from theory import ALTO_MID, CONSONANT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDIR = os.path.join(HERE, 'bench', 'models', 'transformer')
P = dict(T=32, d=64, layers=2, heads=4, epochs=30, lr=2e-3, noise=0.10, w_oct=0.35, w_rep=0.5,
         rep_C=2, vl=0.01, vc=0.004, stream=0, retrain=0, seed=0, w_fourth=0.0, mode='own', a_ng=0.0)
P.update(json.loads(os.environ.get('DUET_TF', '{}')))
NPOS = 8


def feats(sop, tonic, K, pos, D):
    """Per beat j: (longest pc, pc sounding at j + 1 - D, onset bit, pos) from window [j, j+1-D)."""
    lng = np.full(K, 12, np.int64)
    end = np.full(K, 12, np.int64)
    ons = np.zeros(K, np.int64)
    w = 1.0 - D
    i0 = 0
    n = len(sop)
    for j in range(K):
        a, b = j, j + w
        while i0 < n and sop[i0][1] <= a:
            i0 += 1
        best, bl = 12, 0.0
        i = i0
        while i < n and sop[i][0] < b:
            on, off, m = sop[i]
            ov = min(off, b) - max(on, a)
            if ov > 1e-9:
                if ov > bl + 1e-9:
                    best, bl = (m - tonic) % 12, ov
                if on >= a - 1e-9:
                    ons[j] = 1
                if on <= b - 1e-9 < off:
                    end[j] = (m - tonic) % 12
            i += 1
        lng[j] = best
    p = np.array([min(max(int(x), 0), NPOS - 1) for x in pos], np.int64)
    return np.stack([lng, end, ons, p], 1)


def target(sop, tonic, K):
    y = np.zeros((K, 12), np.float32)
    for on, off, m in sop:
        for k in range(max(0, int(on)), min(K, int(math.ceil(off)))):
            ov = min(off, k + 1) - max(on, k)
            if ov > 1e-9:
                y[k, (m - tonic) % 12] += ov
    return y


def noisy(sop, rng, p):
    out = []
    for on, off, m in sop:
        if rng.random() < p:
            pc = (m + rng.randint(1, 11)) % 12
            m = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        out.append([on, off, m])
    return out


class Net(nn.Module):
    def __init__(self, T, d, layers, heads):
        super().__init__()
        self.T = T
        self.e_lng = nn.Embedding(13, d)
        self.e_end = nn.Embedding(13, d)
        self.e_ons = nn.Embedding(2, d)
        self.e_pos = nn.Embedding(NPOS, d)
        self.e_tpos = nn.Embedding(NPOS, d)
        self.e_t = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 12)
        self.register_buffer('mask', torch.triu(torch.full((T, T), float('-inf')), 1))

    def forward(self, x, tpos, valid):
        # x [B, T, 4] tokens (left padded), tpos [B], valid [B, T] bool
        h = (self.e_lng(x[..., 0]) + self.e_end(x[..., 1]) + self.e_ons(x[..., 2]) + self.e_pos(x[..., 3])
             + self.e_t.weight[None])
        h = h * valid[..., None]
        h = self.enc(h, mask=self.mask, src_key_padding_mask=~valid)
        z = self.ln(h[:, -1]) + self.e_tpos(tpos)
        return self.out(z)


def windows(F, pos, T):
    """Row k = tokens of beats k-T..k-1 (left padded) for every beat k of a song."""
    K = len(F)
    pad = np.zeros((T, 4), np.int64)
    G = np.concatenate([pad, F], 0)
    idx = np.arange(K)[:, None] + np.arange(T)[None, :]          # rows k: G[k .. k+T-1] = beats k-T..k-1
    X = G[idx]
    V = idx >= T
    V[:, -1] |= False
    tp = np.array([min(max(int(p), 0), NPOS - 1) for p in pos], np.int64)
    return X, V, tp


def _D(song):
    bpm = song.get('bpm', 100.0)
    return (0.052 + 0.020) / (60.0 / bpm)


def build_data(train, rng, p):
    Xs, Vs, Ts, Ys = [], [], [], []
    for name, songs in train.items():
        for s in songs:
            if 'tonic' not in s or 'beats' not in s:
                continue
            K = s['K']
            pos = [b['pos'] for b in s['beats']]
            sop = noisy(s['sop'], rng, p)
            F = feats(sop, s['tonic'], K, pos, _D(s))
            X, V, tp = windows(F, pos, P['T'])
            Y = target(s['sop'], s['tonic'], K)
            ok = (Y.sum(1) > 0) & (V.any(1))
            Xs.append(X[ok]); Vs.append(V[ok]); Ts.append(tp[ok]); Ys.append(Y[ok] / Y[ok].sum(1, keepdims=True))
    return (np.concatenate(Xs), np.concatenate(Vs), np.concatenate(Ts), np.concatenate(Ys))


def train_net(train):
    torch.manual_seed(P['seed'])
    rng = random.Random(P['seed'])
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net = Net(P['T'], P['d'], P['layers'], P['heads']).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=P['lr'], weight_decay=0.01)
    t0 = time.time()
    data = build_data(train, rng, P['noise'])
    total = P['epochs'] * ((len(data[0]) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, P['lr'], total_steps=total + 10)
    step = 0
    for ep in range(P['epochs']):
        X, V, Tp, Y = data if ep == 0 else build_data(train, rng, P['noise'])   # fresh noise every epoch
        n = len(X)
        perm = np.random.default_rng(ep).permutation(n)
        net.train()
        tot = 0.0
        for b in range(0, n, 256):
            ii = perm[b:b + 256]
            x = torch.from_numpy(X[ii]).to(dev)
            v = torch.from_numpy(V[ii]).to(dev)
            tp = torch.from_numpy(Tp[ii]).to(dev)
            y = torch.from_numpy(Y[ii]).to(dev)
            logit = net(x, tp, v)
            loss = -(y * torch.log_softmax(logit, -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step < total:
                sched.step()
            step += 1
            tot += float(loss) * len(ii)
        print(f'  [transformer] epoch {ep} loss {tot / n:.4f} n {n} {time.time() - t0:.0f}s', flush=True)
        if time.time() - t0 > 200:
            break
    net = net.cpu().eval()
    return net


class _Shim:
    """Stands in for expected_support's n-gram: returns the transformer's q for beat k (optionally
    times the original n-gram ** a_ng). Q is computed causally for the current song only."""

    def __init__(self, ng):
        self.ng, self.Q = ng, None

    def predict(self, o, k, pos):
        q = self.Q[k].astype(float)
        if P['a_ng'] > 0:
            q = q * self.ng.predict(o, k, pos) ** P['a_ng']
        return q / q.sum()


class System:
    def __init__(self, train):
        self.es = None
        if P['mode'] == 'es':
            from bench_systems import expected_support as es
            self.es = es.System(train)
            self.shim = _Shim(self.es.ng)
            self.es.ng = self.shim
        os.makedirs(MDIR, exist_ok=True)
        tag = '_'.join(f'{k}{P[k]}' for k in ('T', 'd', 'layers', 'epochs', 'noise', 'seed'))
        tag += '_' + '-'.join(sorted(f'{k}{len(v)}' for k, v in train.items()))
        path = os.path.join(MDIR, f'net_{tag}.pt')
        self.net = Net(P['T'], P['d'], P['layers'], P['heads'])
        if os.path.exists(path) and not P['retrain']:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(train_net(train).state_dict())
            torch.save(self.net.state_dict(), path)
        self.net.eval()
        torch.set_num_threads(1)
        self.note_guess = {}
        self.val = np.array([(P['w_oct'] if i == 0 else 1.0) if i in CONSONANT else
                             (P['w_fourth'] if i == 5 else 0.0) for i in range(12)])

    @torch.inference_mode()
    def predict(self, X, V, tp):
        return torch.softmax(self.net(torch.from_numpy(X), torch.from_numpy(tp), torch.from_numpy(V)), -1).numpy()

    def run(self, song, cfg):
        t, K, sop = song['tonic'], song['K'], song['sop']
        pos = [b['pos'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        F = feats(sop, t, K, pos, D)       # token j uses only [j, j + 1 - D): complete before beat j+1's decision
        X, V, tp = windows(F, pos, P['T'])
        if P['stream']:
            Q = np.concatenate([self.predict(X[k:k + 1], V[k:k + 1], tp[k:k + 1]) for k in range(K)])
        else:
            Q = self.predict(X, V, tp)
        if self.es is not None:
            self.shim.Q = Q
            self.es.note_guess = self.note_guess
            return self.es.run(song, cfg)
        rep = Repeat(song, int(P['rep_C']), D) if P['w_rep'] > 0 else None
        lo, hi = cfg.alto
        H = np.arange(lo, hi + 1)
        hpc = (H - t) % 12
        valm = self.val[(np.arange(12)[None, :] - hpc[:, None]) % 12]    # [nH, 12]
        out, h_last, her_last, ni = [], None, None, 0
        for k in range(K):
            when = k - D
            jstar = math.floor(when)
            frac = when - jstar
            q = Q[k].astype(float)
            if rep is not None and jstar >= 0:
                g = rep(k, jstar, frac)
                if g is not None:
                    rv = np.roll(g[0], -t)
                    if rv.sum() > 0:
                        q = P['w_rep'] * rv / rv.sum() + (1 - P['w_rep']) * q
            self.note_guess[k] = int(np.argmax(q))
            # her latest heard note (absolute) at the decision time
            while ni < len(sop) and sop[ni][0] < when:
                her_last = sop[ni][2]
                ni += 1
            sc = valm @ q - P['vc'] * np.abs(H - ALTO_MID)
            if h_last is not None:
                sc = sc - P['vl'] * np.abs(H - h_last)
            if her_last is not None:
                ok = H <= her_last - 1
                if ok.any():
                    sc = np.where(ok, sc, -np.inf)
            bt = Beat(k=k, pos=pos[k], z_true=-1, z_hat=-1, q=float(sc.max()))
            if k >= cfg.J:
                h0 = int(H[int(np.argmax(sc))])
                bt.action, bt.h = 'sing', h0
                h_last = h0
            out.append(bt)
        return out


def build(train):
    return System(train)
