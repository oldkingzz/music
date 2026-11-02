"""Feature model (gradient-boosted trees, CPU) for her next beat + an expected-consonance chooser.

At the decision time k - D (jstar = floor(k - D), frac = k - D - jstar) one feature row is built
from her HEARD melody only: beats < jstar fully, beat jstar up to frac. Features, all relative to
the tonic:
  last 6 beat notes (longest pitch class per beat, the partial beat jstar last; 12 = rest),
  signed interval of the last two heard notes, how long the current note has sounded,
  bar position, meter, beats since the last rest (position in phrase),
  per-song memory: for context lengths C in CS, the pitch class that followed earlier places of
  THIS song whose last C beat notes (same bar position) match the current ones, its vote share
  and the number of matches (registered only once beat j is fully heard),
  the song's pitch-class histogram heard so far.
A HistGradientBoostingClassifier predicts a 12-way distribution q of her pitch class in beat k
(trained on train_sets() with 10 % mishearing noise on the inputs, clean time-share targets).
Chooser: her pitch for each class is placed at the octave nearest her last heard note; Duet picks
h in the alto range maximizing
    E[consonant] - wd * E[unison/octave] - vl * |h - h_prev| - vc * |h - alto mid|
(optionally below her last heard note). Duet sings from beat J on, every beat.
Streaming: a row for beat k only needs the melody heard up to k - D, so rows can be built beat by
beat; run() builds them in one pass and predicts them in one batch call for speed.
"""
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np

from model import soprano_obs
from planner import Beat
from theory import ALTO_MID, CONSONANT, CT, N_CHORDS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, 'bench', 'models', 'gbdt')
P = dict(below=1, wd=0.15, vl=0.01, vc=0.004, copies=2, iters=300, lr=0.08, leaves=31, nhist=6,
         noise=0.10, cache=1, max_train=150)
P.update(json.loads(os.environ.get('DUET_GB', '{}')))
CS = (1, 2, 3, 4, 6, 8)
NH = int(P['nhist'])
CONS = np.array([1.0 if i in CONSONANT else 0.0 for i in range(12)])
UNI = np.zeros(12); UNI[0] = 1.0
CTM = np.zeros((N_CHORDS, 12))
for _z in range(N_CHORDS):
    CTM[_z, list(CT[_z])] = 1.0


def add_noise(song, rng, p):
    sop = []
    for on, off, m in song['sop']:
        if rng.random() < p:
            pc = (m + rng.randint(1, 11)) % 12
            m = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        sop.append([on, off, m])
    return dict(song, sop=sop)


def features(song, D):
    """Rows X (one per beat, causal) and per-beat anchor (her last heard MIDI note at the cut)."""
    sop, t, K, M = song['sop'], song['tonic'], song['K'], song.get('M', 4)
    pos = [b['pos'] for b in song['beats']]
    full = [soprano_obs(sop, j)[1] for j in range(K)]
    full = [12 if m is None else (m - t) % 12 for m in full]
    cD = D  # constant per song
    rows, anchors = [], []
    mem = {C: defaultdict(lambda: np.zeros(12)) for C in CS}
    registered = 0
    hist = np.zeros(12)
    hist_upto = 0
    ons = [x[0] for x in sop]
    ni = 0  # notes with on < cut
    for k in range(K):
        cut = k - cD
        jstar = math.floor(cut)
        frac = cut - jstar
        # notes heard so far
        while ni < len(sop) and ons[ni] < cut:
            ni += 1
        # histogram of fully heard beats
        while hist_upto < max(jstar, 0):
            v, _ = soprano_obs(sop, hist_upto)
            hist += np.roll(v, -t)
            hist_upto += 1
        if jstar >= 0:
            pm = soprano_obs(sop, jstar, frac)[1]
            part = 12 if pm is None else (pm - t) % 12
        else:
            part = 12
        seq = [full[j] if j >= 0 else 12 for j in range(jstar - NH + 1, jstar)] + [part]

        def ctx(kk, C):
            js = math.floor(kk - cD)
            if js < C - 1:
                return None
            if kk == k:
                s = seq[-C:] if C <= NH else [full[j] for j in range(js - C + 1, js)] + [part]
            else:
                pmj = soprano_obs(sop, js, kk - cD - js)[1]
                pj = 12 if pmj is None else (pmj - t) % 12
                s = [full[j] for j in range(js - C + 1, js)] + [pj]
            return (pos[kk],) + tuple(s)
        # register beats j fully heard: j + 1 <= cut  -> j <= jstar - 1 (j < cut - 1 + frac)
        while registered < K and registered + 1 <= cut:
            j = registered
            if full[j] != 12:
                for C in CS:
                    key = ctx(j, C)
                    if key is not None:
                        mem[C][key][full[j]] += 1.0 + 0.02 * j
            registered += 1
        rep = []
        for C in CS:
            key = ctx(k, C)
            c = mem[C].get(key) if key is not None else None
            if c is None or c.sum() == 0:
                rep += [12, 0.0, 0.0]
            else:
                rep += [int(np.argmax(c)), float(c.max() / c.sum()), float(c.sum())]
        # last two heard notes and current note duration
        if ni >= 1:
            on, off, m = sop[ni - 1]
            anchor = m
            dur = min(cut, off) - on
            sounding = 1.0 if off > cut else 0.0
            iv = (m - sop[ni - 2][2]) if ni >= 2 else 0
            rest = cut - off if off < cut else 0.0
        else:
            anchor, dur, sounding, iv, rest = None, 0.0, 0.0, 0, 0.0
        # beats since phrase start (last gap >= 0.5 beat)
        ph = 0.0
        for i in range(ni - 1, 0, -1):
            if sop[i][0] - sop[i - 1][1] >= 0.5:
                ph = cut - sop[i][0]
                break
        else:
            ph = cut - (sop[0][0] if sop else 0.0)
        hs = hist.sum()
        hn = hist / hs if hs > 0 else hist
        row = seq + [max(-12, min(12, iv)), dur, sounding, min(rest, 8.0), pos[k], M, min(ph, 64.0),
                     k] + rep + list(hn)
        rows.append(row)
        anchors.append(anchor)
    return np.array(rows, dtype=float), anchors


def n_cat():
    return NH + 4 + 0  # seq cats and pos? (pos ordinal ok)


def cat_mask(ncol):
    m = np.zeros(ncol, dtype=bool)
    m[:NH] = True
    base = NH + 8
    for i in range(len(CS)):
        m[base + 3 * i] = True
    return m


def targets(song):
    t = song['tonic']
    out = []
    for k in range(song['K']):
        v, _ = soprano_obs(song['sop'], k)
        v = np.roll(v, -t)
        out.append(v / v.sum() if v.sum() > 0 else None)
    return out


class System:
    def __init__(self, train):
        from sklearn.ensemble import HistGradientBoostingClassifier
        D = (0.052 + 0.020) / 0.6          # cfg defaults at 100 bpm (bench uses the same L, delta)
        names = sorted(train)
        key = '_'.join(f'{n}{len(train[n])}' for n in names) + json.dumps(
            {k: P[k] for k in ('copies', 'iters', 'lr', 'leaves', 'nhist', 'noise', 'max_train')}, sort_keys=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, f'model_{abs(hash(key)) % 10**10}.pkl')
        import zlib
        path = os.path.join(MODEL_DIR, f'model_{zlib.crc32(key.encode())}.pkl')
        if P['cache'] and os.path.exists(path):
            self.clf = pickle.load(open(path, 'rb'))
        else:
            t0 = time.time()
            Xs, ys, ws = [], [], []
            rng = random.Random(51600)
            for n in names:
                songs = [s for s in train[n] if 'beats' in s and 'tonic' in s][:int(P['max_train'])]
                for s in songs:
                    tg = targets(s)
                    for _ in range(int(P['copies'])):
                        X, _a = features(add_noise(s, rng, P['noise']), D)
                        for k in range(s['K']):
                            if tg[k] is None:
                                continue
                            for pc in np.nonzero(tg[k] > 0.05)[0]:
                                Xs.append(X[k]); ys.append(pc); ws.append(tg[k][pc])
            X = np.array(Xs); y = np.array(ys); w = np.array(ws)
            self.clf = HistGradientBoostingClassifier(
                max_iter=int(P['iters']), learning_rate=P['lr'], max_leaf_nodes=int(P['leaves']),
                categorical_features=cat_mask(X.shape[1]), early_stopping=False, random_state=0)
            self.clf.fit(X, y, sample_weight=w)
            print(f'[gbdt] trained on {len(y)} rows from {names} in {time.time() - t0:.0f}s')
            pickle.dump(self.clf, open(path, 'wb'))
        self.classes = list(self.clf.classes_)
        self.note_guess = {}

    def predict(self, X):
        pr = self.clf.predict_proba(X)
        q = np.zeros((len(X), 12))
        q[:, self.classes] = pr
        return q

    def run(self, song, cfg):
        t, K = song['tonic'], song['K']
        pos = [b['pos'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        X, anchors = features(song, D)
        Q = self.predict(X)
        lo, hi = cfg.alto
        H = np.arange(lo, hi + 1)
        Hf = H.astype(float)
        out, h_last = [], None
        for k in range(K):
            q = Q[k]
            self.note_guess[k] = int(np.argmax(q))
            bt = Beat(k=k, pos=pos[k], z_true=-1, z_hat=-1, q=float(q.max()))
            a = anchors[k]
            if k >= cfg.J and a is not None:
                # her MIDI pitch per class: nearest octave to her last heard note
                pcs_abs = (np.arange(12) + t) % 12
                d = (pcs_abs - a) % 12
                d = np.where(d > 6, d - 12, d)
                m = a + d
                iv = np.abs(m[None, :] - H[:, None]) % 12
                sc = CONS[iv] @ q - P['wd'] * UNI[iv] @ q - P['vc'] * np.abs(Hf - ALTO_MID)
                if h_last is not None:
                    sc = sc - P['vl'] * np.abs(Hf - h_last)
                if P['below']:
                    ok = H <= a - 1
                    if ok.any():
                        sc = np.where(ok, sc, -np.inf)
                h = int(H[int(np.argmax(sc))])
                hp = (h - t) % 12
                zs = [z for z in range(N_CHORDS) if hp in CT[z]]
                if zs:
                    e4 = CTM @ q
                    bt.z_hat = max(zs, key=lambda z: e4[z])
                bt.action, bt.h = 'sing', h
                h_last = h
            out.append(bt)
        return out


def build(train):
    return System(train)
