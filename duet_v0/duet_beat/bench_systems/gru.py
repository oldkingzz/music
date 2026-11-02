"""A small streaming GRU guesses her next beat; Duet picks the note with the best EXPECTED consonance.

Stream: before beat k, Duet has heard her melody up to the decision time b = k - D. The GRU takes ONE
step per beat, fed only with what was heard in the window [k - 1 - D, k - D) (the beat of audio that
arrived since the last decision), so its hidden state is a causal summary of everything heard:
  share[12]   time share of each pitch class in the window (relative to the tonic)
  cur[13]     pitch class sounding at the decision instant (or rest), elapsed time since its onset
  last[12]    pitch class of the last note started before b, its register, onsets in the window
  pos, M      bar position of beat k and the meter
Output: q = distribution over the pitch-class time share of her beat k. Trained (PyTorch, MPS) on the
melodies of every train set (train_sets(), clean targets), with inputs misheard 10 % of notes like
the benchmark. Inference is a numpy GRU step per beat (no torch at run time).
Choice: h in the alto range and below her latest heard note maximizing
    sum_pc q[pc] * CONS[(pc - h) % 12]  - w_dbl * q[h pc]  - vl |h - h_prev| - vc |h - mid|
optionally mixed with the per-song repeat memory (exp_ceiling.Repeat) as in expected_support.
"""
import hashlib
import json
import math
import os
import random
import time

import numpy as np

from exp_ceiling import Repeat
from planner import Beat
from theory import ALTO_MID, CONSONANT, CT, N_CHORDS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, 'bench', 'models', 'gru')
P = dict(H=128, layers=2, train_s=90.0, copies=3, lr=3e-3, w_rep=0.3, w_dbl=0.15, vl=0.01, vc=0.004,
         below=1, rep_C=2, D_lat=0.072, seed=0, chord=1)
P.update(json.loads(os.environ.get('DUET_GRU', '{}')))
CONS = np.array([1.0 if i in CONSONANT else 0.0 for i in range(12)])
CTM = np.zeros((N_CHORDS, 12))
for _z in range(N_CHORDS):
    CTM[_z, list(CT[_z])] = 1.0
F = 12 + 13 + 1 + 12 + 1 + 1 + 8 + 5
MS = {2: 0, 3: 1, 4: 2, 6: 3}


# ---------------------------------------------------------------------- features (causal by construction)
def step_feats(sop, ons, k, D, tonic, pos_k, M):
    """Features of the window [k-1-D, k-D): only notes with onset < k-D, clipped at k-D."""
    import bisect
    b = k - D
    a = b - 1.0
    x = np.zeros(F)
    i1 = bisect.bisect_left(ons, b)               # notes 0..i1-1 started before b
    cur, last, nons = None, None, 0
    for i in range(i1 - 1, -1, -1):
        on, off, m = sop[i]
        if off <= a and last is not None:
            break
        if last is None:
            last = (on, off, m)
        ov = min(off, b) - max(on, a)
        if ov > 1e-9:
            x[(m - tonic) % 12] += ov
        if on >= a:
            nons += 1
        if cur is None and on <= b - 1e-6 < off:
            cur = (on, off, m)
        if on < a - 8:
            break
    o = 12
    if cur is not None:
        x[o + (cur[2] - tonic) % 12] = 1.0
        x[o + 13] = min(b - cur[0], 4.0) / 4.0
    else:
        x[o + 12] = 1.0
    o = 26
    if last is not None:
        x[o + (last[2] - tonic) % 12] = 1.0
        x[o + 12] = (last[2] - 67) / 12.0
    x[o + 13] = nons / 4.0
    x[40 + min(int(pos_k), 7)] = 1.0
    x[48 + MS.get(int(M), 4)] = 1.0
    return x


def beat_target(sop, k, tonic):
    v = np.zeros(12)
    for on, off, m in sop:
        if on >= k + 1:
            break
        ov = min(off, k + 1) - max(on, k)
        if ov > 1e-9:
            v[(m - tonic) % 12] += ov
    return v


def mishear(sop, rng, p=0.10):
    out = []
    for on, off, m in sop:
        if rng.random() < p:
            pc = (m + rng.randint(1, 11)) % 12
            m = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        out.append([on, off, m])
    return out


def song_arrays(s, rng, noisy=True):
    sop = sorted(s['sop'], key=lambda n: n[0])
    t, K = s['tonic'], s['K']
    M = s.get('M', 4)
    beats = s.get('beats') or [{'pos': k % M} for k in range(K)]
    bpm = s.get('bpm', 100.0)
    D = P['D_lat'] * bpm / 60.0 * rng.uniform(0.8, 1.25)
    hs = mishear(sop, rng) if noisy else sop
    ons = [n[0] for n in hs]
    X = np.stack([step_feats(hs, ons, k, D, t, beats[k]['pos'], M) for k in range(K)])
    # targets from the CLEAN melody (sweep)
    Y = np.zeros((K, 12))
    for on, off, m in sop:
        for k in range(max(0, int(on)), min(K - 1, int(math.ceil(off)) - 1) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov > 1e-9:
                Y[k, (m - t) % 12] += ov
    return X.astype(np.float32), Y.astype(np.float32)


# ---------------------------------------------------------------------- training (torch, MPS)
def _key(train):
    sig = {n: len(v) for n, v in sorted(train.items())}
    sig.update({k: P[k] for k in ('H', 'layers', 'train_s', 'copies', 'lr', 'seed')})
    return hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:10], sig


def train_model(train, log=print):
    import torch
    import torch.nn as nn
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(P['seed'])
    rng = random.Random(P['seed'])
    t0 = time.time()
    seqs = []
    for name, songs in train.items():
        for s in songs:
            if not s.get('sop') or s.get('K', 0) < 4 or s.get('tonic') is None:
                continue
            for c in range(int(P['copies']) if name == 'bach' else 1):
                seqs.append(song_arrays(s, rng))
    log(f'gru: {len(seqs)} sequences, {sum(len(x) for x, _ in seqs)} steps, feats {time.time() - t0:.1f}s')
    # chunk to length <= 128 (state reset at chunk start is a mild approximation for long songs)
    T = 128
    chunks = []
    for X, Y in seqs:
        for i in range(0, len(X), T):
            chunks.append((X[i:i + T], Y[i:i + T]))
    rng.shuffle(chunks)
    nval = max(1, len(chunks) // 20)
    val, tr = chunks[:nval], chunks[nval:]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(F, P['H'])
            self.gru = nn.GRU(P['H'], P['H'], num_layers=int(P['layers']), batch_first=True)
            self.out = nn.Linear(P['H'], 12)

        def forward(self, x):
            h, _ = self.gru(torch.tanh(self.inp(x)))
            return self.out(h)

    def batch(cs):
        L = max(len(x) for x, _ in cs)
        Xb = np.zeros((len(cs), L, F), np.float32)
        Yb = np.zeros((len(cs), L, 12), np.float32)
        for i, (x, y) in enumerate(cs):
            Xb[i, :len(x)] = x
            Yb[i, :len(y)] = y
        return torch.from_numpy(Xb).to(dev), torch.from_numpy(Yb).to(dev)

    def loss_fn(net, Xb, Yb):
        lp = torch.log_softmax(net(Xb), -1)
        w = Yb.sum(-1)
        return -(Yb * lp).sum() / w.sum().clamp(min=1e-6), w

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=P['lr'])
    B = 128
    vb = [batch(val[i:i + 256]) for i in range(0, len(val), 256)]
    step, ep, best, best_state = 0, 0, 1e9, None
    tstart = time.time()
    while time.time() - tstart < P['train_s']:
        rng.shuffle(tr)
        net.train()
        for i in range(0, len(tr), B):
            Xb, Yb = batch(tr[i:i + B])
            loss, _ = loss_fn(net, Xb, Yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            if time.time() - tstart > P['train_s']:
                break
        ep += 1
        net.eval()
        with torch.no_grad():
            vl = sum(float(loss_fn(net, X, Y)[0]) for X, Y in vb) / len(vb)
        if vl < best:
            best = vl
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
        log(f'gru: epoch {ep} steps {step} val CE {vl:.4f} ({time.time() - tstart:.0f}s on {dev})')
    return best_state, {'val_ce': best, 'epochs': ep, 'steps': step, 'device': str(dev),
                        'train_s': time.time() - tstart, 'n_seq': len(seqs)}


# ---------------------------------------------------------------------- numpy GRU step
class NpGRU:
    def __init__(self, st):
        self.Wi, self.bi = st['inp.weight'], st['inp.bias']
        self.L = int(P['layers'])
        self.W = [(st[f'gru.weight_ih_l{l}'], st[f'gru.weight_hh_l{l}'], st[f'gru.bias_ih_l{l}'],
                   st[f'gru.bias_hh_l{l}']) for l in range(self.L)]
        self.Wo, self.bo = st['out.weight'], st['out.bias']
        self.H = self.Wi.shape[0]

    def init(self):
        return [np.zeros(self.H, np.float32) for _ in range(self.L)]

    def step(self, x, hs):
        z = np.tanh(self.Wi @ x + self.bi)
        new = []
        H = self.H
        for (wih, whh, bih, bhh), h in zip(self.W, hs):
            gi = wih @ z + bih
            gh = whh @ h + bhh
            r = 1 / (1 + np.exp(-(gi[:H] + gh[:H])))
            u = 1 / (1 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h = (1 - u) * n + u * h
            new.append(h)
            z = h
        lo = self.Wo @ z + self.bo
        e = np.exp(lo - lo.max())
        return e / e.sum(), new


class FastRepeat:
    """exp_ceiling.Repeat with a hash table: same match rule (her pitch at 12 points per beat over the
    C beats before the decision time, same bar position, earlier beat j fully heard), O(1) lookup.
    Beat j's context is stored only once beat j is fully heard (j + 1 <= decision time)."""

    def __init__(self, sop, ons, pos, C, D):
        self.sop, self.ons, self.pos, self.C, self.D = sop, ons, pos, C, D
        self.ctx, self.table, self.added = {}, {}, 0

    def _pitch(self, x):
        import bisect
        i = bisect.bisect_right(self.ons, x) - 1
        if i >= 0 and self.sop[i][0] <= x < self.sop[i][1]:
            return self.sop[i][2]
        return 0

    def _sampled(self, a, b):
        n = int(round((b - a) * 12))
        return tuple(self._pitch(a + (i + 0.5) / 12) for i in range(n))

    def __call__(self, k):
        cut = k - self.D
        if math.floor(cut) < self.C:
            self.ctx[k] = None
            return None
        c = self._sampled(cut - self.C, cut)
        self.ctx[k] = c if any(c) else None
        while self.added + 1 <= cut:                     # beat `added` is now fully heard
            j = self.added
            cj = self.ctx.get(j)
            if cj is not None:
                v = np.zeros(12)
                for on, off, m in self.sop:
                    if on >= j + 1:
                        break
                    ov = min(off, j + 1) - max(on, j)
                    if ov > 1e-9:
                        v[m % 12] += ov
                self.table[(self.pos[j], cj)] = v
            self.added += 1
        if self.ctx[k] is None:
            return None
        return self.table.get((self.pos[k], self.ctx[k]))


class System:
    def __init__(self, train):
        key, sig = _key(train)
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, f'gru_{key}.npz')
        if os.path.exists(path):
            st = dict(np.load(path))
        else:
            st, info = train_model(train)
            np.savez(path, **st)
            json.dump({'sig': sig, **info}, open(path.replace('.npz', '.json'), 'w'), indent=1)
        self.net = NpGRU({k: v.astype(np.float32) for k, v in st.items()})
        self.note_guess = {}

    def run(self, song, cfg):
        t, K, M = song['tonic'], song['K'], song['M']
        sop = sorted(song['sop'], key=lambda n: n[0])
        ons = [n[0] for n in sop]
        pos = [b['pos'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        rep = FastRepeat(sop, ons, pos, int(P['rep_C']), D) if P['w_rep'] > 0 else None
        lo, hi = cfg.alto
        H = np.arange(lo, hi + 1)
        Hf = H.astype(float)
        hpc = (H - t) % 12
        cons = CONS[(np.arange(12)[None, :] - hpc[:, None]) % 12]          # [cand, her pc]
        dbl = (np.arange(12)[None, :] == hpc[:, None]).astype(float)
        hs = self.net.init()
        out, h_last, her_last = [], None, None
        import bisect
        for k in range(K):
            x = step_feats(sop, ons, k, D, t, pos[k], M)
            q, hs = self.net.step(x.astype(np.float32), hs)
            q = q.astype(float)
            b = k - D
            i1 = bisect.bisect_left(ons, b)
            if i1 > 0:
                her_last = sop[i1 - 1][2]
            if rep is not None:
                g = rep(k)
                if g is not None:
                    rv = np.roll(g, -t)
                    if rv.sum() > 0:
                        q = P['w_rep'] * rv / rv.sum() + (1 - P['w_rep']) * q
            self.note_guess[k] = int(np.argmax(q))
            sc = cons @ q - P['w_dbl'] * (dbl @ q) - P['vc'] * np.abs(Hf - ALTO_MID)
            if h_last is not None:
                sc = sc - P['vl'] * np.abs(Hf - h_last)
            if her_last is not None:
                ok = H <= her_last - P['below']
                if ok.any():
                    sc = np.where(ok, sc, -np.inf)
            hh = int(H[int(np.argmax(sc))])
            z = int(np.argmax(CTM @ q))
            bt = Beat(k=k, pos=pos[k], z_true=-1, z_hat=z, q=float((CTM @ q)[z]))
            if k >= cfg.J:
                bt.action, bt.h = 'sing', hh
                h_last = hh
            out.append(bt)
        return out


def build(train):
    return System(train)
