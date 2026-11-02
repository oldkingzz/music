"""Train a small bidirectional GRU (melody beats -> chord) on MPS, union of train splits + style id.
Tunes the epoch on VAL only; exports the best weights to numpy (models/bigru.npz) for CPU inference.

  .venv/bin/python futures/d4_styles/gru_train.py --minutes 6
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feats import NF, STYLES, features, labels, load_all  # noqa: E402

OVERSAMPLE = {'bach': 12, 'folk': 3, 'pop': 1, 'pop909': 2}


class Net(nn.Module):
    def __init__(self, H=96, layers=2, drop=0.2):
        super().__init__()
        self.inp = nn.Linear(NF, H)
        self.gru = nn.GRU(H, H, num_layers=layers, batch_first=True, bidirectional=True, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(2 * H, 7)

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        y, _ = self.gru(h)
        return self.out(self.drop(y))


def batches(items, bs, rng):
    idx = rng.permutation(len(items))
    for i in range(0, len(idx), bs):
        sel = [items[j] for j in idx[i:i + bs]]
        T = max(len(x) for x, _ in sel)
        X = np.zeros((len(sel), T, NF), np.float32)
        Y = np.full((len(sel), T), -1, np.int64)
        for b, (x, y) in enumerate(sel):
            X[b, :len(x)] = x
            Y[b, :len(y)] = y
        yield X, Y


def predict(net, items, dev):
    net.eval()
    out = []
    with torch.no_grad():
        for x, _ in items:
            out.append(net(torch.from_numpy(x[None]).to(dev))[0].argmax(-1).cpu().numpy())
    net.train()
    return out


def acc(preds, items):
    c = n = 0
    for p, (_, y) in zip(preds, items):
        m = y >= 0
        c += int((p[m] == y[m]).sum())
        n += int(m.sum())
    return c / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=6.0)
    ap.add_argument('--H', type=int, default=96)
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'bigru.npz'))
    ap.add_argument('--init', default='')
    ap.add_argument('--lr', type=float, default=2e-3)
    a = ap.parse_args()
    torch.manual_seed(5160)
    rng = np.random.default_rng(5160)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    data = load_all()
    train, val = [], {}
    for st in STYLES:
        tr = [(features(s, st), labels(s)) for s in data[st]['train']]
        train += tr * OVERSAMPLE[st]
        val[st] = [(features(s, st), labels(s)) for s in data[st]['val']]
    print('train seqs', len(train), 'device', dev, flush=True)
    net = Net(a.H)
    if a.init:
        w = np.load(a.init)
        net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != 'H'})
        print('warm start from', a.init, flush=True)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    t0, best, ep, log = time.time(), -1, 0, []
    while time.time() - t0 < a.minutes * 60:
        ep += 1
        for X, Y in batches(train, 32, rng):
            X, Y = torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)
            loss = lossf(net(X).reshape(-1, 7), Y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if time.time() - t0 > a.minutes * 60:
                break
        va = {st: acc(predict(net, val[st], dev), val[st]) for st in STYLES}
        mean = float(np.mean(list(va.values())))
        log.append({'epoch': ep, 'loss': float(loss.detach()), 'val': va, 'val_mean': mean, 't': time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if mean > best:
            best = mean
            sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
            np.savez(a.out, H=a.H, **sd)
    json.dump({'log': log, 'best_val_mean': best}, open(a.out.replace('.npz', '_log.json'), 'w'), indent=1)
    print('best val mean', best)


if __name__ == '__main__':
    main()
