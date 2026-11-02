"""Learn the decision directly: a small net scores every Duet pitch class by how consonant it will be.

At the decision time k - D for beat k, a streaming SongState turns what has been heard so far into
a feature vector (tonic-relative): longest pitch class of the last 7 fully heard beats, the partial
beat being heard, the note sounding now and how long it has been held, the song's pitch-class
histogram so far, per-song memory (what followed the same bar position / last 1, 2, 3 beat notes
earlier in THIS song, as time-share vectors + counts), bar position and her register.
An MLP (trained on MPS on every train set, inputs with 10 % mishearing like the bench) has two heads:
  val[c]  P(consonant) for Duet on pitch class c below her = time-weighted share of her TRUE beat-k
          melody consonant with c (BCE on that soft target): the decision value, learned directly
  q[pc]   her beat-k pitch-class time shares (soft cross-entropy): used only for the doubling penalty
Duet then sings h in the alto range, below her latest heard note, maximizing
  val[h pc] - wd * q[h pc] - vl * |h - h_prev| - vc * |h - alto mid|.
Causal: notes are consumed in onset order; only notes with onset < k - D are ever touched, clipped
at k - D. No chord labels are used. Weights: bench/models/policy/ (retrained when train sets change).
"""
import hashlib
import json
import math
import os
import random

import numpy as np

from planner import Beat
from theory import ALTO_MID, CONSONANT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDIR = os.path.join(HERE, 'bench', 'models', 'policy')
P = dict(hid=256, epochs=12, lr=2e-3, aug=3, wd=0.1, vl=0.01, vc=0.003, use='val', ctx=7,
         cap=60000, tag='v2', ens=3, drop=0.1)
P.update(json.loads(os.environ.get('DUET_POL', '{}')))
CONS = np.array([1.0 if i in CONSONANT else 0.0 for i in range(12)])
# CM[pc, c] = 1 if her pc above Duet's pc c is consonant
CM = CONS[(np.arange(12)[:, None] - np.arange(12)[None, :]) % 12]
NOISE = 0.10


def noisy(sop, rng, p=NOISE):
    out = []
    for on, off, m in sop:
        if rng.random() < p:
            pc = (m + rng.randint(1, 11)) % 12
            m = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        out.append([on, off, m])
    return out


def beat_targets(sop, K, t):
    """Clean time-share vectors of her melody per beat (training targets only)."""
    V = np.zeros((K, 12))
    for on, off, m in sop:
        for k in range(int(on), min(K, int(math.ceil(off)))):
            ov = min(off, k + 1) - max(on, k)
            if ov > 1e-9:
                V[k, (m - t) % 12] += ov
    return V


def _oh(i, n=13):
    v = np.zeros(n)
    v[i if i is not None and i >= 0 else n - 1] = 1.0
    return v


class SongState:
    """Streaming feature builder. Call features(k) for k = 0, 1, 2, ... in order."""

    def __init__(self, sop, t, pos, M, D, ctx):
        self.sop = sorted(sop, key=lambda n: n[0])
        self.t, self.pos, self.M, self.D, self.ctx = t, pos, M, D, ctx
        self.ni = 0                      # notes with onset < decision time
        self.done = 0                    # beats fully heard and finalized
        self.V, self.o = [], []          # per fully heard beat: time-share vec, longest pc (-1)
        self.hist = np.zeros(12)
        self.mem = {}                    # (order, key) -> [vec sum, count]
        self.her_last = None

    def _vec(self, j, end):
        v = np.zeros(12)
        best, bl = -1, 0.0
        for i in range(self.ni - 1, max(-1, self.ni - 40), -1):
            on, off, m = self.sop[i]
            ov = min(off, end) - max(on, j)
            if ov > 1e-9:
                pc = (m - self.t) % 12
                v[pc] += ov
                if ov > bl + 1e-9:
                    best, bl = pc, ov
            elif off <= j and on < j - 8:
                break
        return v, best

    def _keys(self, j, o):
        """Memory keys for predicting beat j from the beat notes o[j-3..j-1]."""
        p = self.pos[j]
        a = o[j - 1] if j >= 1 else -2
        b = o[j - 2] if j >= 2 else -2
        c = o[j - 3] if j >= 3 else -2
        return [(0, p), (1, a), (2, b, a), (3, p, c, b, a)]

    def features(self, k):
        when = k - self.D
        while self.ni < len(self.sop) and self.sop[self.ni][0] < when:
            self.ni += 1
        jstar = math.floor(when)
        frac = when - jstar
        # finalize fully heard beats j < jstar (j + 1 <= when)
        while self.done < max(jstar, 0):
            j = self.done
            v, b = self._vec(j, j + 1)
            self.V.append(v)
            self.o.append(b)
            self.hist += v
            if v.sum() > 0:
                vn = v / v.sum()
                for key in self._keys(j, self.o):
                    e = self.mem.setdefault(key, [np.zeros(12), 0.0])
                    e[0] += vn
                    e[1] += 1
            self.done += 1
        if jstar >= 0:
            pv, pb = self._vec(jstar, when)
        else:
            pv, pb = np.zeros(12), -1
        o = list(self.o) + ([pb] if jstar >= 0 else [])
        o += [-1] * (k - len(o))
        # sounding note
        cur, held, reg = -1, 0.0, 0.0
        if self.ni > 0:
            on, off, m = self.sop[self.ni - 1]
            if off > when - 0.25:
                cur, held = (m - self.t) % 12, min(when - on, 4.0)
            self.her_last = m
            reg = (m - 67) / 12.0
        f = [_oh(o[k - i] if k - i >= 0 else -1) for i in range(1, self.ctx + 1)]
        f.append(pv / pv.sum() if pv.sum() > 0 else pv)
        f.append(_oh(cur))
        f.append(np.array([held, frac, reg, math.log1p(self.hist.sum()) / 5.0]))
        f.append(self.hist / self.hist.sum() if self.hist.sum() > 0 else self.hist)
        for key in self._keys(k, o):
            e = self.mem.get(key)
            if e is None:
                f.append(np.zeros(13))
            else:
                f.append(np.concatenate([e[0] / e[1], [math.log1p(e[1]) / 3.0]]))
        p = self.pos[k]
        posv = np.zeros(6)
        posv[min(p, 3)] = 1.0
        posv[4] = float(p == 0)
        posv[5] = float(self.M == 3)
        f.append(posv)
        return np.concatenate(f).astype(np.float32)


def _song_rows(s, D, ctx, rng, aug):
    t, K = s['tonic'], s['K']
    pos = [b['pos'] for b in s['beats']]
    V = beat_targets(s['sop'], K, t)
    X, Y = [], []
    for a in range(aug):
        st = SongState(noisy(s['sop'], rng), t, pos, s['M'], D, ctx)
        for k in range(K):
            x = st.features(k)
            if k >= 1 and V[k].sum() > 1e-6:
                X.append(x)
                Y.append(V[k])
    return X, Y


def _fingerprint(train):
    sig = {n: len(v) for n, v in sorted(train.items()) if n != 'essen'}
    sig['P'] = {k: P[k] for k in ('hid', 'epochs', 'lr', 'aug', 'ctx', 'cap', 'tag', 'ens', 'drop')}
    return hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12], sig


def make_rows(train, D):
    import torch
    rng = random.Random(5160)
    X, Y = [], []
    for name, songs in train.items():
        if name == 'essen':
            continue                      # melody-only set has no chords but has melodies: keep it small
        tot_b = sum(s.get('K', 0) for s in songs)
        aug = max(1, min(int(P['aug']), int(P['cap']) // max(tot_b, 1)))
        used = 0
        for s in songs:                   # per-dataset cap on rows keeps training under a minute
            if used >= P['cap']:
                break
            if s.get('K', 0) < 2 or not s.get('sop'):
                continue
            used += s['K'] * aug
            x, y = _song_rows(s, D, int(P['ctx']), rng, aug)
            X += x
            Y += y
    return torch.tensor(np.stack(X)), torch.tensor(np.stack(Y), dtype=torch.float32)


def train_model(X, Y, seed=0):
    import torch
    torch.manual_seed(seed)
    w = Y.sum(1)
    Yn = Y / w[:, None]
    T = Yn @ torch.tensor(CM, dtype=torch.float32)          # consonance target per candidate pc
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    h = int(P['hid'])
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], h), torch.nn.GELU(), torch.nn.Dropout(P['drop']),
                              torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Dropout(P['drop']),
                              torch.nn.Linear(h, 24)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=P['lr'], weight_decay=1e-4)
    n = len(X)
    E = int(P['epochs'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, P['lr'], total_steps=E * ((n + 1023) // 1024))
    Xd, Yd, Td, wd = X.to(dev), Yn.to(dev), T.to(dev), w.to(dev)
    for ep in range(E):
        net.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, 1024):
            b = perm[i:i + 1024]
            out = net(Xd[b])
            lq = -(Yd[b] * torch.log_softmax(out[:, :12], 1)).sum(1)
            lv = torch.nn.functional.binary_cross_entropy_with_logits(out[:, 12:], Td[b], reduction='none').mean(1)
            loss = ((lq + 2.0 * lv) * wd[b]).sum() / wd[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
    net.eval().cpu()
    Ws = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
          for m in net if isinstance(m, torch.nn.Linear)]
    return Ws, n, tot / n


class System:
    def __init__(self, train):
        D = (0.052 + 0.020) / 0.6                           # bench config: L + delta at 100 bpm
        fp, sig = _fingerprint(train)
        path = os.path.join(MDIR, f'policy_{fp}.npz')
        E = int(P['ens'])
        if os.path.exists(path):
            z = np.load(path)
            Ws = [[(z[f'W{e}_{i}'], z[f'b{e}_{i}']) for i in range(3)] for e in range(E)]
        else:
            Ws, info = [], []
            X, Y = make_rows(train, D)
            for e in range(E):
                W, n, loss = train_model(X, Y, seed=e)
                Ws.append(W)
                info.append(loss)
            os.makedirs(MDIR, exist_ok=True)
            np.savez(path, **{f'W{e}_{i}': w for e, W in enumerate(Ws) for i, (w, _) in enumerate(W)},
                     **{f'b{e}_{i}': b for e, W in enumerate(Ws) for i, (_, b) in enumerate(W)})
            json.dump({'sig': sig, 'rows': n, 'loss': info}, open(path[:-4] + '.json', 'w'))
        # stack the ensemble members into one wide net (block-diagonal layers) for one forward pass
        self.E = E
        W0 = np.concatenate([W[0][0] for W in Ws], 1)
        b0 = np.concatenate([W[0][1] for W in Ws])
        h = Ws[0][1][0].shape[0]
        W1 = np.zeros((h * E, h * E), np.float32)
        W2 = np.zeros((h * E, 24 * E), np.float32)
        for e, W in enumerate(Ws):
            W1[e * h:(e + 1) * h, e * h:(e + 1) * h] = W[1][0]
            W2[e * h:(e + 1) * h, e * 24:(e + 1) * 24] = W[2][0]
        b1 = np.concatenate([W[1][1] for W in Ws])
        b2 = np.concatenate([W[2][1] for W in Ws])
        self.W = [(W0, b0), (W1, b1), (W2, b2)] if E > 1 else Ws[0]
        self.Ws = Ws
        self.note_guess = {}

    def forward(self, x):
        (W0, b0), (W1, b1), (W2, b2) = self.W
        a = x @ W0 + b0
        a = 0.5 * a * (1 + np.tanh(0.7978845608 * (a + 0.044715 * a ** 3)))
        a = a @ W1 + b1
        a = 0.5 * a * (1 + np.tanh(0.7978845608 * (a + 0.044715 * a ** 3)))
        o = (a @ W2 + b2).reshape(-1, 24)
        q = np.exp(o[:, :12] - o[:, :12].max(1, keepdims=True))
        q = (q / q.sum(1, keepdims=True)).mean(0)
        return q, (1.0 / (1.0 + np.exp(-o[:, 12:]))).mean(0)

    def run(self, song, cfg):
        t, M, K = song['tonic'], song['M'], song['K']
        pos = [b['pos'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        st = SongState(song['sop'], t, pos, M, D, int(P['ctx']))
        lo, hi = cfg.alto
        H = np.arange(lo, hi + 1)
        hpc = (H - t) % 12
        out, h_last = [], None
        for k in range(K):
            x = st.features(k)
            q, val = self.forward(x)
            self.note_guess[k] = int(np.argmax(q))
            bt = Beat(k=k, pos=pos[k], z_true=song['beats'][k].get('z', -1))
            her = st.her_last
            if her is not None:
                v = val if P['use'] == 'val' else CM.T @ q
                sc = v[hpc] - P['wd'] * q[hpc] - P['vc'] * np.abs(H - ALTO_MID)
                if h_last is not None:
                    sc = sc - P['vl'] * np.abs(H - h_last)
                ok = H <= her - 1
                if ok.any():
                    sc = np.where(ok, sc, -np.inf)
                h0 = int(H[int(np.argmax(sc))])
                bt.action, bt.h, bt.q = 'sing', h0, float(v[(h0 - t) % 12])
                h_last = h0
            out.append(bt)
        return out


def build(train):
    return System(train)
