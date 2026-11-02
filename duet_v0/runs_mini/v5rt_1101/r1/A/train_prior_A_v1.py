"""Idea A (run v5rt_1101, round 1): fit the melody prior from TRAIN only (bench3.train_sets(): bach, folk, pop, pop909).
Writes runs_mini/v5rt_1101/models/A_prior_v1.npz (mode: refuses to overwrite). No val data is read here.

1. Pitch model: P(her next TRUE pitch | her last heard pitch m_c as a tonic-relative pc, the last heard interval
   m_c - m_p clipped to +-5 (or none), the metric class of the next onset: 0 downbeat, 1 other beat, 2 off-beat).
   Target = true next midi - heard m_c, clipped to -12..12 (25 bins). Contexts are HEARD pitches: bench3's mishearing
   (10 %: another pitch class at the nearest octave) is drawn R times per song with fixed seeds, so the table is
   P(true next | heard context). Smoothed counts with back-off (pc, iv, cls) -> (pc, iv) -> (pc) -> (), each level mixed
   with its parent by n / (n + A) with n the RAW count; probabilities from dataset-balanced weights (each of the 4
   train sets carries the same total weight).
2. Rhythm model at plan points q = 0, 0.5, ... with the bench_rt view at L = 0.1 (her notes heard when onset + L <= q,
   offsets when offset + L <= q). Context: s = q - last heard onset (1/12 beat bins, cap 4 beats), rest = last offset
   known (and how long ago, 0.5-beat bins, cap 3), phase (q on a beat or half), bar class tkey(pos, M) as m1, last
   inter-onset interval (1/12 bins, cap 4, or none). Outcomes averaged per context (same back-off scheme):
     att_new  an attack at q would land on a note she has NOT been heard to start yet: a note sounding at q or starting
              within EPS = 0.01 beat after q (pop909 onsets sit 0.0003 after the grid: musically simultaneous, so they
              count as new, not as the held note)
     att_held an attack at q would land on her last heard note (still sounding at q) and not on a new one
     cov_new  beats of the first L of the first unheard note inside [q, q + 0.5) (the plan note sounds there)
     cov_hold beats her last heard note sounds inside [q, min(q + 0.5, first hear point after q))
     cov_next beats of the first L of the first unheard note inside [q + 0.5, q + 0.5 + L) (if the plan is kept)
"""
import bisect
import os
import random
import sys
import zlib

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'models', 'A_prior_v1.npz')
L = 0.1
EPS = 0.01
R = 3                    # mishearing draws per song for the pitch model
A = 20.0                 # back-off strength (raw counts)
NOISE = 0.10


def tkey(pos, M):
    return int(pos) % 4 if M == 4 else (4 if int(pos) % M == 0 else 5)


def mishear(song, seed):
    rng = random.Random(zlib.crc32(str(song.get('id', song.get('title', ''))).encode()) ^ seed)
    out = []
    for on, off, m in sorted(song['sop'], key=lambda n: n[0]):
        hm = m
        if rng.random() < NOISE:
            pc = (m + rng.randint(1, 11)) % 12
            hm = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        out.append((float(on), float(off), int(m), int(hm)))
    return out


def ncls(on, song):
    k = int(on + 1e-9)
    ph = on - k
    pos = song['beats'][k]['pos'] if k < len(song['beats']) else 0
    if ph < EPS:
        return 0 if int(pos) % song['M'] == 0 else 1
    return 2


def ivb(d):
    return 11 if d is None else int(np.clip(d, -5, 5)) + 5


def rhythm_rows(song):
    notes = sorted(((float(on), float(off)) for on, off, m in song['sop']), key=lambda n: n[0])
    ons = [n[0] for n in notes]
    K, M = song['K'], song['M']
    rows = []
    q = 0.0
    while q < K:
        c = bisect.bisect_right([o + L for o in ons], q) - 1          # last heard note
        if c >= 0:
            on_c, off_c = notes[c]
            s_b = min(48, int(round((q - on_c) * 12)))
            rest_b = 0 if off_c + L > q else 1 + min(3, int((q - off_c) / 0.5))
            phase = 0 if abs(q - round(q)) < 1e-9 else 1
            k = int(q + 1e-9)
            posb = tkey(song['beats'][k]['pos'] if k < len(song['beats']) else k % M, M)
            ioi_b = 49 if c == 0 else min(48, int(round((on_c - notes[c - 1][0]) * 12)))
            u = c + 1 if c + 1 < len(notes) else None
            att_new = att_held = cov_new = cov_hold = cov_next = 0.0
            if u is not None:
                on_u, off_u = notes[u]
                if on_u < q + EPS and off_u > q:
                    att_new = 1.0
                e_u = min(off_u, on_u + L)
                cov_new = max(0.0, min(e_u, q + 0.5) - max(on_u, q))
                cov_next = max(0.0, min(e_u, q + 0.5 + L) - max(on_u, q + 0.5))
                end = min(q + 0.5, on_u + L)
            else:
                end = q + 0.5
            if not att_new and on_c <= q < off_c:
                att_held = 1.0
            cov_hold = max(0.0, min(off_c, end) - q)
            rows.append((s_b, rest_b, phase, posb, ioi_b, att_new, att_held, cov_new, cov_hold, cov_next))
        q += 0.5
    return rows


def backoff(sums, cnt, parent_axes):
    """sums [..., F] weighted outcome sums, cnt[...] = (raw n, weight). Returns the smoothed mean table over the
    full key, backing off by summing out the axes in parent_axes order (last listed axis first)."""
    raise NotImplementedError


def main():
    if os.path.exists(OUT):
        sys.exit(f'refusing to overwrite {OUT}')
    tr = bench3.train_sets()
    nn = {ds: sum(len(s['sop']) for s in songs) for ds, songs in tr.items()}
    tot = sum(nn.values())
    wds = {ds: tot / len(nn) / n for ds, n in nn.items()}
    print('train notes', nn, 'dataset weights', {k: round(v, 3) for k, v in wds.items()}, flush=True)

    # ---------------- pitch model
    Cw = np.zeros((12, 12, 3, 25))
    Cn = np.zeros((12, 12, 3))
    for ds, songs in tr.items():
        for s in songs:
            t = s['tonic']
            for r in range(R):
                ns = mishear(s, 7000 + r)
                for j in range(1, len(ns)):
                    on, off, m_true, _ = ns[j]
                    m_c = ns[j - 1][3]
                    m_p = ns[j - 2][3] if j >= 2 else None
                    pc = (m_c - t) % 12
                    iv = ivb(None if m_p is None else m_c - m_p)
                    cl = ncls(on, s)
                    d = int(np.clip(m_true - m_c, -12, 12)) + 12
                    Cw[pc, iv, cl, d] += wds[ds] / R
                    Cn[pc, iv, cl] += 1.0 / R
    P0 = Cw.sum((0, 1, 2)) + 1e-3
    P0 /= P0.sum()
    W1, N1 = Cw.sum((1, 2)), Cn.sum((1, 2))
    P1 = (W1 / np.maximum(W1.sum(-1, keepdims=True), 1e-12)) * (N1 / (N1 + A))[:, None] + (A / (N1 + A))[:, None] * P0
    W2, N2 = Cw.sum(2), Cn.sum(2)
    P2 = (W2 / np.maximum(W2.sum(-1, keepdims=True), 1e-12)) * (N2 / (N2 + A))[..., None] + (A / (N2 + A))[..., None] * P1[:, None, :]
    P3 = (Cw / np.maximum(Cw.sum(-1, keepdims=True), 1e-12)) * (Cn / (Cn + A))[..., None] + (A / (Cn + A))[..., None] * P2[:, :, None, :]
    P3 /= P3.sum(-1, keepdims=True)
    print('pitch model: contexts with data', int((Cn > 0).sum()), 'of', Cn.size, '; P(repeat) mean',
          round(float((P3[..., 12] * Cn).sum() / Cn.sum()), 3), flush=True)

    # ---------------- rhythm model
    shape = (49, 5, 2, 6, 50)
    Sw = np.zeros(shape + (5,))
    Sn = np.zeros(shape)
    Ww = np.zeros(shape)
    for ds, songs in tr.items():
        for s in songs:
            for row in rhythm_rows(s):
                key = row[:5]
                Sw[key] += wds[ds] * np.array(row[5:])
                Ww[key] += wds[ds]
                Sn[key] += 1.0
    G0 = Sw.sum((0, 1, 2, 3, 4)) / Ww.sum()

    def lvl(axes_keep, parent):
        ax = tuple(i for i in range(5) if i not in axes_keep)
        sw, ww, sn = Sw.sum(ax), Ww.sum(ax), Sn.sum(ax)
        mean = sw / np.maximum(ww, 1e-12)[..., None]
        lam = (sn / (sn + A))[..., None]
        return lam * mean + (1 - lam) * parent

    R1 = lvl((0, 1), G0[None, None, :])
    R2 = lvl((0, 1, 2), R1[:, :, None, :])
    R3 = lvl((0, 1, 2, 3), R2[:, :, :, None, :])
    R4 = lvl((0, 1, 2, 3, 4), R3[:, :, :, :, None, :])
    print('rhythm model: plan points', int(Sn.sum()), 'contexts with data', int((Sn > 0).sum()), 'global means',
          dict(zip(('att_new', 'att_held', 'cov_new', 'cov_hold', 'cov_next'), np.round(G0, 4).tolist())), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'xb') as fh:
        np.savez(fh, mel=P3.astype(np.float32), rhy=R4.astype(np.float32), mel_n=Cn.astype(np.float32),
                 rhy_n=Sn.astype(np.float32), L=np.float32(L), eps=np.float32(EPS), A=np.float32(A))
    print('wrote', OUT, os.path.getsize(OUT), 'bytes', flush=True)


if __name__ == '__main__':
    main()
