"""v5rt_1101 round 1, idea C: train the learned joint forecaster (futures/v5_rt/rt_systems/r1_C_fnet_v1.py) on
bench3.train_sets() ONLY (bach, folk, pop, pop909 train splits; inputs with simulated 10 % mishearing, targets TRUE
pitches and chords). Datasets are sampled equally (as m3_seq_v1). Val (bench3's own heard melody, seed1) is only
watched: losses and forecaster accuracy are printed and logged; the final weights are saved (no val-based selection).

  python train_fnet_C_v1.py --secs 420 --out <new .pt under runs_mini/v5rt_1101/models/> --log <new .json> [--init <.pt>]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench3  # noqa: E402
from rt_systems.r1_C_fnet_v1 import L_TRAIN, NF, NQ, TOL, meter_index, torch_net  # noqa: E402

MODELS = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'models')
HERE = os.path.dirname(os.path.abspath(__file__))
CROP = 192             # notes per training crop
BATCH = 48             # crops per step
P_FROM0 = 0.3          # share of crops that start at the song's first note
NOISE = 0.10           # simulated mishearing (bench's model: another pc, nearest octave)
LR = 2e-3


def prep(song, L=L_TRAIN):
    """Everything about one song except the heard pitch features (filled per batch, after mishearing)."""
    notes = sorted(song['sop'], key=lambda n: n[0])
    if not notes:
        return None
    t, M, K = song['tonic'], song['M'], song['K']
    on = np.array([float(n[0]) for n in notes])
    off = np.array([float(n[1]) for n in notes])
    mid = np.array([int(n[2]) for n in notes], dtype=np.int64)
    N = len(on)
    pos = [b['pos'] for b in song['beats']]
    z = [b['z'] for b in song['beats']]
    mi = meter_index(M)
    F = np.zeros((N, NF), np.float32)
    F[1:, 13] = np.log2(np.maximum(off[:-1] - on[:-1], 1 / 16)) / 3.0
    k = np.floor(on).astype(int)
    posk = np.array([min(pos[kk], 3) if 0 <= kk < len(pos) else 0 for kk in k], dtype=np.int64)
    F[np.arange(N), 14 + posk] = 1.0
    ph = on - k
    F[:, 18] = np.sin(2 * np.pi * ph)
    F[:, 19] = np.cos(2 * np.pi * ph)
    F[:, 20] = ph < 1e-6
    F[:, 21 + mi] = 1.0
    # chord soft target per note (m3_seq_v1.song_xy)
    Ys = np.zeros((N, 7), np.float32)
    for n in range(N):
        for kk in range(int(on[n]), int(off[n]) + 1):
            ov = min(off[n], kk + 1) - max(on[n], kk)
            if ov > 1e-9 and kk < len(z) and z[kk] >= 0:
                Ys[n, z[kk]] += ov
    w = Ys.sum(1)
    Ys[w > 0] /= w[w > 0, None]
    # plan points
    Q = np.array([0.5 * j for j in range(int(math.ceil(K / 0.5)) + 1) if 0.5 * j < K])
    nQ = len(Q)
    ctx = np.searchsorted(on + L, Q, side='right') - 1
    kq = np.floor(Q).astype(int)
    posq = np.array([min(pos[kk], 3) if kk < len(pos) else 0 for kk in kq], dtype=np.int64)
    QF = np.zeros((nQ, NQ), np.float32)
    QF[np.arange(nQ), posq] = 1.0
    QF[:, 4] = (Q - kq) > 0.25
    QF[:, 5 + mi] = 1.0
    hc = ctx >= 0
    c = np.maximum(ctx, 0)
    QF[~hc, 8] = 1.0
    dt = Q - on[c]
    QF[hc, 9] = np.log2(np.maximum(dt[hc], 1 / 16)) / 3.0
    QF[hc, 10] = np.minimum(dt[hc], 4.0) / 4.0
    ended = hc & (off[c] + L <= Q)
    QF[ended, 11] = 1.0
    QF[ended, 12] = np.log2(np.maximum(Q[ended] - off[c][ended], 1 / 16)) / 3.0
    QF[ended, 13] = np.log2(np.maximum(off[c][ended] - on[c][ended], 1 / 16)) / 3.0
    j = np.searchsorted(on, Q - 1e-9, side='left')
    jj = np.minimum(j, N - 1)
    has = (j < N) & (on[jj] < Q + TOL)
    pc = ((mid[jj] - t) % 12).astype(np.int64)
    zq = np.array([z[kk] if kk < len(z) else -1 for kk in kq], dtype=np.int64)
    return dict(t=t, N=N, F=F, mid=mid, Ys=Ys, Q=Q, ctx=ctx, QF=QF, has=has, jn=jj, pc=pc, zq=zq)


def batch(items, rng, noise):
    """items = [(prep, a, b, heard_mid or None)] -> torch tensors for one step."""
    B = len(items)
    nmax = max(b - a for _, a, b, _ in items)
    X = np.zeros((B, nmax, NF), np.float32)
    gi, QF, has, pc, ct, cm, dj, dm, qs = [], [], [], [], [], [], [], [], []
    for bi, (S, a, b, hm) in enumerate(items):
        F = S['F'][a:b].copy()
        m = (S['mid'] if hm is None else hm)[a:b].copy()
        if noise:
            flip = rng.random(len(m)) < noise
            d = rng.integers(1, 12, len(m))
            m = np.where(flip, m + np.where(d < 6, d, d - 12), m)
        F[np.arange(len(m)), (m - S['t']) % 12] = 1.0
        F[:, 12] = (m - 66) / 12.0
        X[bi, :b - a] = F
        ctx = S['ctx']
        sel = np.nonzero(((ctx >= a) & (ctx < b)) | ((a == 0) & (ctx == -1)))[0]
        if not len(sel):
            continue
        gi.append(np.where(ctx[sel] >= 0, bi * nmax + ctx[sel] - a, B * nmax))
        QF.append(S['QF'][sel])
        h = S['has'][sel]
        has.append(h)
        pc.append(S['pc'][sel])
        jn = S['jn'][sel]
        Yn = S['Ys'][jn]                                     # soft chord target of the note at q
        Yb = np.zeros((len(sel), 7), np.float32)
        zq = S['zq'][sel]
        Yb[zq >= 0, zq[zq >= 0]] = 1.0                       # chord of beat int(q)
        C = np.where(h[:, None], Yn, Yb)
        ct.append(C)
        cm.append(C.sum(1) > 0)
        D = (Yn[:, :, None] * np.eye(12, dtype=np.float32)[S['pc'][sel]][:, None, :]).reshape(len(sel), 84)
        dj.append(D)
        dm.append(h & (Yn.sum(1) > 0))
        qs.append(S['Q'][sel])
    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    return dict(X=torch.from_numpy(X), gi=torch.from_numpy(cat(gi)), QF=torch.from_numpy(cat(QF)),
                has=torch.from_numpy(cat(has).astype(np.float32)), pc=torch.from_numpy(cat(pc)),
                C=torch.from_numpy(cat(ct)), cm=torch.from_numpy(cat(cm)), D=torch.from_numpy(cat(dj)),
                dm=torch.from_numpy(cat(dm)), q=cat(qs))


def forward(net, bt):
    S, _ = net.encode(bt['X'])
    B, n, H = S.shape
    Sf = torch.cat([S.reshape(B * n, H), torch.zeros(1, H)], 0)[bt['gi']]
    Xf = torch.cat([bt['X'].reshape(B * n, NF), torch.zeros(1, NF)], 0)[bt['gi']]
    return net.head(Sf, Xf, bt['QF'])


def losses(out, bt):
    la, lb, lc, ld = out
    has = bt['has']
    a = torch.nn.functional.binary_cross_entropy_with_logits(la, has)
    hb = has > 0
    b = torch.nn.functional.cross_entropy(lb[hb], bt['pc'][hb]) if hb.any() else la.sum() * 0
    cm, dm = bt['cm'], bt['dm']
    c = -(torch.log_softmax(lc[cm], -1) * bt['C'][cm]).sum(-1).mean() if cm.any() else la.sum() * 0
    d = -(torch.log_softmax(ld[dm], -1) * bt['D'][dm]).sum(-1).mean() if dm.any() else la.sum() * 0
    return a, b, c, d


def auc(scores, labels):
    o = np.argsort(scores)
    r = np.empty(len(scores))
    r[o] = np.arange(1, len(scores) + 1)
    npos = labels.sum()
    nneg = len(labels) - npos
    return float((r[labels].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def val_metrics(net, vsets):
    """Forecaster accuracy on val (bench3's heard melody, seed1), plan points q >= J only."""
    net.eval()
    res = {}
    with torch.no_grad():
        for ds, items in vsets.items():
            P_on, Y_on, B1, B3, C1, nb, nc, L4 = [], [], 0, 0, 0, 0, 0, np.zeros(4)
            nl = 0
            for i in range(0, len(items), 24):
                chunk = items[i:i + 24]
                bt = batch(chunk, None, 0.0)
                out = forward(net, bt)
                a, b, c, d = losses(out, bt)
                L4 += np.array([a.item(), b.item(), c.item(), d.item()])
                nl += 1
                keep = torch.from_numpy(bt['q'] >= bench3.J)
                la, lb, lc, _ = out
                P_on.append(torch.sigmoid(la[keep]).numpy())
                Y_on.append(bt['has'][keep].numpy() > 0)
                hb = (bt['has'] > 0) & keep
                top = torch.topk(lb[hb], 3, -1).indices
                y = bt['pc'][hb]
                B1 += int((top[:, 0] == y).sum())
                B3 += int((top == y[:, None]).any(-1).sum())
                nb += int(hb.sum())
                # chord top-1 against the hard chord of beat int(q)
                zc = bt['C'].argmax(-1)
                cmk = bt['cm'] & keep
                C1 += int((lc[cmk].argmax(-1) == zc[cmk]).sum())
                nc += int(cmk.sum())
            p, yv = np.concatenate(P_on), np.concatenate(Y_on)
            res[ds] = {'on_auc': auc(p, yv), 'on_acc': float(((p >= 0.5) == yv).mean()), 'on_rate': float(yv.mean()),
                       'pc_top1': B1 / max(nb, 1), 'pc_top3': B3 / max(nb, 1), 'chord_top1': C1 / max(nc, 1),
                       'loss_a': L4[0] / nl, 'loss_b': L4[1] / nl, 'loss_c': L4[2] / nl, 'loss_d': L4[3] / nl}
    res['mean'] = {k: float(np.mean([r[k] for r in res.values()])) for k in next(iter(res.values()))}
    net.train()
    return res


def show(res, head):
    m = res['mean']
    print(f"{head} val mean: on AUC {m['on_auc']:.3f} acc {m['on_acc']:.3f} | pc top1 {m['pc_top1']:.3f} top3 "
          f"{m['pc_top3']:.3f} | chord top1 {m['chord_top1']:.3f} | losses a {m['loss_a']:.3f} b {m['loss_b']:.3f} "
          f"c {m['loss_c']:.3f} d {m['loss_d']:.3f} || " + ' '.join(
              f"{ds} {r['on_auc']:.3f}/{r['pc_top1']:.3f}/{r['pc_top3']:.3f}/{r['chord_top1']:.3f}"
              for ds, r in res.items() if ds != 'mean'), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secs', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--init')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out, logp = os.path.abspath(a.out), os.path.abspath(a.log)
    assert out.startswith(MODELS + os.sep) and not os.path.exists(out), out
    assert logp.startswith(HERE + os.sep) and not os.path.exists(logp), logp
    torch.set_num_threads(3)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    prng = random.Random(a.seed)
    t0 = time.time()
    tr = {ds: [p for p in (prep(s) for s in songs) if p is not None] for ds, songs in bench3.train_sets().items()}
    vsets = {}
    for ds, songs in bench3.eval_sets('val').items():
        items = []
        for s in songs:
            p = prep(s)
            if p is None:
                continue
            hm = np.array([int(n[2]) for n in sorted(bench3.B2.heard(s)['sop'], key=lambda n: n[0])], dtype=np.int64)
            items.append((p, 0, p['N'], hm))
        vsets[ds] = sorted(items, key=lambda x: x[0]['N'])
    print(f"prep {time.time() - t0:.0f}s: train " + ' '.join(f'{k} {len(v)}' for k, v in tr.items())
          + ' | val ' + ' '.join(f'{k} {len(v)}' for k, v in vsets.items()), flush=True)
    net = torch_net()
    if a.init:
        net.load_state_dict(torch.load(a.init))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    names = list(tr)
    log = {'args': vars(a), 'crop': CROP, 'batch': BATCH, 'noise': NOISE, 'lr': LR, 'evals': []}
    res = val_metrics(net, vsets)
    show(res, 'it 0')
    log['evals'].append({'it': 0, 'secs': 0, 'val': res})
    ts, it, last_eval, run = time.time(), 0, time.time(), np.zeros(4)
    while time.time() - ts < a.secs:
        frac = (time.time() - ts) / a.secs
        for g in opt.param_groups:
            g['lr'] = LR * (1.0 - 0.9 * max(0.0, (frac - 0.6) / 0.4))
        items = []
        for _ in range(BATCH):
            S = prng.choice(tr[prng.choice(names)])
            if S['N'] > CROP:
                st = 0 if prng.random() < P_FROM0 else prng.randint(0, S['N'] - CROP)
                items.append((S, st, st + CROP, None))
            else:
                items.append((S, 0, S['N'], None))
        bt = batch(items, rng, NOISE)
        la, lb, lc, ld = losses(forward(net, bt), bt)
        loss = la + lb + lc + ld
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run = 0.98 * run + 0.02 * np.array([la.item(), lb.item(), lc.item(), ld.item()])
        it += 1
        if it % 100 == 0:
            print(f'  it {it} {time.time() - ts:.0f}s train loss a {run[0]:.3f} b {run[1]:.3f} c {run[2]:.3f} '
                  f'd {run[3]:.3f}', flush=True)
        if time.time() - last_eval > 150:
            res = val_metrics(net, vsets)
            show(res, f'it {it}')
            log['evals'].append({'it': it, 'secs': time.time() - ts, 'val': res})
            last_eval = time.time()
    net.eval()
    res = val_metrics(net, vsets)
    show(res, f'final it {it}')
    log['evals'].append({'it': it, 'secs': time.time() - ts, 'val': res})
    with open(out, 'xb') as fh:
        torch.save(net.state_dict(), fh)
    log['out'] = out
    log['iters'] = it
    log['wall'] = time.time() - t0
    with open(logp, 'x') as fh:
        json.dump(log, fh, indent=1)
    print(f'saved {out} ({os.path.getsize(out) / 1e6:.2f} MB) after {it} steps, wall {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
