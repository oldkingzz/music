"""v5rt_1101 r1 idea C: smoke test of train_fnet_C_v1 (no files written). Preps a few songs, times training steps,
runs val_metrics on a small val subset, and checks that NpNet (numpy) reproduces the torch forward pass and that
note_feat / query_feat reproduce prep()'s arrays."""
import os
import random
import sys
import time

import numpy as np
import torch

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'C'))
import train_fnet_C_v1 as T  # noqa: E402
from rt_systems.r1_C_fnet_v1 import NpNet, note_feat, query_feat  # noqa: E402

torch.set_num_threads(3)
t0 = time.time()
trs = T.bench3.train_sets()
tr = {ds: [p for p in (T.prep(s) for s in songs[:60]) if p is not None] for ds, songs in trs.items()}
print(f'prep of 4 x 60 songs {time.time() - t0:.1f}s', flush=True)
net = T.torch_net()
opt = torch.optim.Adam(net.parameters(), lr=T.LR)
rng, prng = np.random.default_rng(0), random.Random(0)
for it in range(6):
    c0 = time.time()
    items = []
    for _ in range(T.BATCH):
        S = prng.choice(tr[prng.choice(list(tr))])
        st = 0 if S['N'] <= T.CROP else prng.randint(0, S['N'] - T.CROP)
        items.append((S, st, min(S['N'], st + T.CROP), None))
    bt = T.batch(items, rng, T.NOISE)
    c1 = time.time()
    la, lb, lc, ld = T.losses(T.forward(net, bt), bt)
    loss = la + lb + lc + ld
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f'step {it}: batch {c1 - c0:.3f}s fwd+bwd {time.time() - c1:.3f}s plan points {len(bt["q"])} '
          f'losses {la.item():.3f} {lb.item():.3f} {lc.item():.3f} {ld.item():.3f}', flush=True)
# val subset
vs = T.bench3.eval_sets('val')
vsets = {}
for ds, songs in vs.items():
    items = []
    for s in songs[:8]:
        p = T.prep(s)
        hm = np.array([int(n[2]) for n in sorted(T.bench3.B2.heard(s)['sop'], key=lambda n: n[0])], dtype=np.int64)
        items.append((p, 0, p['N'], hm))
    vsets[ds] = items
c0 = time.time()
T.show(T.val_metrics(net, vsets), 'smoke')
print(f'val subset {time.time() - c0:.1f}s', flush=True)
# numpy forward == torch forward, and the run-time features == prep's arrays, on one val song
net.eval()
npn = NpNet(net.state_dict())
s = vs['folk'][3]
p = T.prep(s)
hs = sorted(T.bench3.B2.heard(s)['sop'], key=lambda n: n[0])
hm = np.array([int(n[2]) for n in hs], dtype=np.int64)
bt = T.batch([(p, 0, p['N'], hm)], None, 0.0)
with torch.no_grad():
    la, lb, lc, ld = T.forward(net, bt)
t, M, pos = s['tonic'], s['M'], [b['pos'] for b in s['beats']]
states, xs, h = [], [], npn.zero()
for i, (on, off, m) in enumerate(hs):
    k = int(on)
    prev = None if i == 0 else float(hs[i - 1][1] - hs[i - 1][0])
    x = note_feat((m - t) % 12, m, prev, pos[k] if k < len(pos) else 0, on - k, M)
    assert np.allclose(x, bt['X'][0, i].numpy(), atol=1e-6), (i, x, bt['X'][0, i])
    h = npn.step(h, x.astype(np.float64))
    states.append(h)
    xs.append(x)
err = {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0, 'qf': 0.0}
for r, q in enumerate(p['Q']):
    c = p['ctx'][r]
    if c >= 0:
        on, off, m = hs[c]
        last = (on, off if off + T.L_TRAIN <= q else None, m)
    else:
        last = None
    kq = int(q)
    qf = query_feat(q, last, M, pos[kq] if kq < len(pos) else 0)
    err['qf'] = max(err['qf'], float(np.abs(qf - p['QF'][r]).max()))
    hh = states[c] if c >= 0 else npn.zero()
    xx = xs[c].astype(np.float64) if c >= 0 else np.zeros(T.NF)
    pa, pb, pc, pd = npn.query(hh, xx, qf.astype(np.float64))
    err['a'] = max(err['a'], abs(pa - torch.sigmoid(la[r]).item()))
    err['b'] = max(err['b'], float(np.abs(pb - torch.softmax(lb[r], -1).numpy()).max()))
    err['c'] = max(err['c'], float(np.abs(pc - torch.softmax(lc[r], -1).numpy()).max()))
    err['d'] = max(err['d'], float(np.abs(pd.ravel() - torch.softmax(ld[r], -1).numpy()).max()))
print('numpy vs torch max abs diff', err, flush=True)
c0 = time.time()
for _ in range(1000):
    h = npn.step(h, xs[0].astype(np.float64))
c1 = time.time()
for _ in range(1000):
    npn.query(h, xs[0].astype(np.float64), qf.astype(np.float64))
print(f'numpy step {1e3 * (c1 - c0):.0f} us, query {1e3 * (time.time() - c1):.0f} us', flush=True)
