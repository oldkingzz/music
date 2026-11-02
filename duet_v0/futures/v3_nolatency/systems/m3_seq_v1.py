"""m3_seq_v1: learned sequence model (GRU) over the heard melody -> chord distribution -> voice-led note.

History use: a GRU reads her notes one by one (tonic-relative pitch class, octave, previous note's
duration, bar position, beat phase). Its hidden state is the memory of the past; at each note it outputs
P(chord) for the current beat. Duet then sings the note maximising P(consonant chord tone) minus a
voice-leading cost (small steps, no parallel 5ths/8ves, no doubling her).
DUET_HIST = H beats: the GRU is re-run from an empty state over only the notes that started in the last
H beats (the current note always included). Unset / -1 = full history (incremental hidden state).
"""
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from theory import ALTO, ALTO_MID, CONSONANT, CT

HERE = os.path.dirname(os.path.abspath(__file__))
NF = 12 + 1 + 1 + 4 + 2 + 1 + 3           # pc, octave, log prev dur, pos one-hot(4), phase sin/cos, onbeat, meter
CTM = np.zeros((7, 12), dtype=np.float32)
for _z in range(7):
    for _p in CT[_z]:
        CTM[_z, _p] = 1.0


def feat(pc, midi, prev_dur, pos, phase, M):
    f = np.zeros(NF, dtype=np.float32)
    f[pc] = 1.0
    f[12] = (midi - 66) / 12.0
    f[13] = 0.0 if prev_dur is None else math.log2(max(prev_dur, 1 / 16)) / 3.0
    f[14 + min(pos, 3)] = 1.0
    f[18] = math.sin(2 * math.pi * phase)
    f[19] = math.cos(2 * math.pi * phase)
    f[20] = 1.0 if phase < 1e-6 else 0.0
    f[21 + (0 if M == 2 else 1 if M == 3 else 2)] = 1.0
    return f


class Net(nn.Module):
    def __init__(self, hid=96):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(NF, hid), nn.Tanh())
        self.gru = nn.GRU(hid, hid, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hid + NF, hid), nn.ReLU(), nn.Linear(hid, 7))

    def forward(self, x, h=None):
        y, h = self.gru(self.inp(x), h)
        return self.out(torch.cat([y, x], -1)), h


def song_xy(song, rng, noise):
    t, M = song['tonic'], song['M']
    notes = sorted(song['sop'], key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    pos = [b['pos'] for b in song['beats']]
    X, Y, W = [], [], []
    prev = None
    for on, off, m in notes:
        if rng.random() < noise:
            pc = (m + rng.randint(1, 11)) % 12
            m = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        k = int(on)
        X.append(feat((m - t) % 12, m, None if prev is None else prev, pos[k] if k < len(pos) else 0, on - k, M))
        tgt = np.zeros(7, dtype=np.float32)
        for kk in range(k, int(off) + 1):          # soft target: chords of the beats this note covers
            ov = min(off, kk + 1) - max(on, kk)
            if ov > 1e-9 and kk < len(z) and z[kk] >= 0:
                tgt[z[kk]] += ov
        w = tgt.sum()
        Y.append(tgt / w if w > 0 else tgt)
        W.append(1.0 if w > 0 else 0.0)
        prev = off - on
    return np.array(X), np.array(Y), np.array(W, dtype=np.float32)


def train_net(train, secs=150, hid=96, seed=0, crop=None):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    sets = {k: v for k, v in train.items() if v}
    net = Net(hid)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t0, it = time.time(), 0
    names = list(sets)
    while time.time() - t0 < secs:
        batch = []
        for _ in range(64):
            s = rng.choice(sets[rng.choice(names)])
            X, Y, W = song_xy(s, rng, 0.10)
            if len(X) > 200:
                a = rng.randint(0, len(X) - 200)
                X, Y, W = X[a:a + 200], Y[a:a + 200], W[a:a + 200]
            if crop and len(X) > 4:
                a = rng.randint(0, len(X) - 1)
                X, Y, W = X[a:a + crop], Y[a:a + crop], W[a:a + crop]
            batch.append((X, Y, W))
        L = max(len(b[0]) for b in batch)
        xb = np.zeros((len(batch), L, NF), np.float32)
        yb = np.zeros((len(batch), L, 7), np.float32)
        wb = np.zeros((len(batch), L), np.float32)
        for i, (X, Y, W) in enumerate(batch):
            xb[i, :len(X)], yb[i, :len(X)], wb[i, :len(X)] = X, Y, W
        logits, _ = net(torch.from_numpy(xb))
        lp = torch.log_softmax(logits, -1)
        loss = -((lp * torch.from_numpy(yb)).sum(-1) * torch.from_numpy(wb)).sum() / max(wb.sum(), 1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        it += 1
        if it % 200 == 0:
            print(f'  it {it} loss {loss.item():.3f} {time.time() - t0:.0f}s', flush=True)
    net.eval()
    return net


class Duet:
    def __init__(self, net, hist, cfg):
        self.net, self.hist, self.cfg = net, hist, cfg

    def start(self, meta):
        self.t, self.M, self.pos = meta['tonic'], meta['M'], meta['pos']
        self.h, self.prev, self.prevmel, self.buf = None, None, None, []

    def chord_probs(self, onset, midi, prev_dur):
        k = int(onset)
        f = feat((midi - self.t) % 12, midi, prev_dur, self.pos[k] if k < len(self.pos) else 0, onset - k, self.M)
        with torch.no_grad():
            if self.hist < 0:
                lg, self.h = self.net(torch.from_numpy(f)[None, None], self.h)
            else:
                self.buf.append((onset, f))
                self.buf = [(o, g) for o, g in self.buf if o >= onset - self.hist]
                if self.hist == 0:
                    self.buf = self.buf[-1:]
                x = torch.from_numpy(np.stack([g for _, g in self.buf]))[None]
                lg, _ = self.net(x)
            p = torch.softmax(lg[0, -1], -1).numpy()
        return p

    def step(self, i, onset, midi, prev_dur):
        p = self.chord_probs(onset, midi, prev_dur)
        pfit = p @ CTM                               # P(pitch class is a tone of the current chord)
        c = self.cfg
        ref = self.prev if self.prev is not None else ALTO_MID
        best, bs = -1, -1e9
        for h in range(ALTO[0], ALTO[1] + 1):
            iv = abs(midi - h) % 12
            if iv not in CONSONANT:
                continue
            s = pfit[(h - self.t) % 12]
            if iv == 0:
                s -= c['dbl']
            if h >= midi:
                s -= c['above']
            if self.prev is not None:
                d = abs(h - self.prev)
                s -= c['step'] * d + (c['big'] if d > 7 else 0.0)
                if self.prevmel is not None:
                    dm, dh = midi - self.prevmel, h - self.prev
                    i0 = abs(self.prevmel - self.prev) % 12
                    if dm != 0 and dh != 0 and (dm > 0) == (dh > 0) and iv in (0, 7) and i0 == iv:
                        s -= c['par']
            else:
                s -= 0.01 * abs(h - ref)
            if s > bs:
                best, bs = h, s
        self.prev, self.prevmel = best, midi
        return best


CFG = dict(dbl=0.35, above=0.05, step=0.02, big=0.3, par=0.4)


def get_net(train, tag='full', crop=None, secs=150):
    path = os.path.join(HERE, f'm3_seq_{tag}.pt')
    if os.path.exists(path):
        net = Net(int(os.environ.get('M3_HID', '96')))
        net.load_state_dict(torch.load(path))
        net.eval()
        return net
    net = train_net(train, secs=float(os.environ.get('M3_SECS', secs)), crop=crop,
                    hid=int(os.environ.get('M3_HID', '96')), seed=int(os.environ.get('M3_SEED', '0')))
    torch.save(net.state_dict(), path)
    return net


def build(train):
    hist = float(os.environ.get('DUET_HIST', '-1'))
    cfg = dict(CFG)
    for k in cfg:
        if os.environ.get('M3_' + k.upper()):
            cfg[k] = float(os.environ['M3_' + k.upper()])
    net = get_net(train, os.environ.get('M3_TAG', 'full'))
    torch.set_num_threads(1)
    return Duet(net, hist, cfg)
