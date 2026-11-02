"""Retrieval / memory: guess her next beat note from what followed similar contexts, then pick the
note that is most likely consonant with her.

Index (built from train_sets() only, every dataset incl. melody-only essen): every beat of every
training melody as (context = her last n beat notes, tonic-relative longest pitch class, 12 = rest;
bar position) -> her longest pitch class in the next beat. Keys for n = 0..NMAX, plus "one wildcard"
keys (one context slot replaced by *), so a context with one misheard note (10 % noise) still finds
its continuations. Keys are int64 codes; the index is two sorted numpy arrays (key*16+pc, count),
looked up with one searchsorted per beat (~50 us).
Song memory: the same keys built online from the song's own HEARD beats that are fully heard by the
decision time (beat j's continuation j+1 only once j+1 < floor(k - D)); weighted higher.
Prediction: interpolated backoff over n (corpus), the same over song memory, mixed with a weight
that grows with how much song memory matched; optionally multiplied with the causal chord-HMM note
prior (as in expected_support).
Choice: Duet's MIDI note h in the alto range maximizing sum_pc q[pc] * cons(|m_pc - h|) where m_pc is
the pitch of class pc nearest to her latest heard note (so above/below is judged in real pitch), a
unison/octave discount (avoid DOUBLING), and a small leap penalty. Duet always sings from beat J.
Causal: only uses the heard melody up to k - D (full beats < jstar, the partial beat jstar).
"""
import bisect
import json
import math
import os

import numpy as np

from model import ChordHMM
from planner import Beat
from theory import ALTO_MID, CONSONANT, CT, N_CHORDS

P = dict(nmax=4, beta=1.0, wild=1.0, sbeta=1.0, w_song=1.0, s_b=0.5, hmm=1.0, uni=0.1,
         vl=0.01, vc=0.004, below=1, lo=-1, hi=-1, ess=1, bias=0.0, tv=1)
P.update(json.loads(os.environ.get('DUET_RET', '{}')))
NMAX = int(P['nmax'])
REST, WILD = 12, 13
CONS = np.array([1.0 if i in CONSONANT else 0.0 for i in range(12)])
CTM = np.zeros((N_CHORDS, 12))
for _z in range(N_CHORDS):
    CTM[_z, list(CT[_z])] = 1.0


def beat_longest(sop, K, cut=None):
    """Per-beat longest-sounding MIDI note (None if silent) for beats 0..K-1, notes clipped at cut."""
    best = [None] * K
    blen = [0.0] * K
    for on, off, m in sop:
        if cut is not None:
            if on >= cut:
                break
            off = min(off, cut)
        for k in range(max(0, int(math.floor(on))), min(K, int(math.ceil(off)))):
            ov = min(off, k + 1) - max(on, k)
            if ov > blen[k] + 1e-9:
                best[k], blen[k] = m, ov
    return best


def beat_vecs(sop, K):
    """Per-beat time share of each pitch class (absolute pc), beats 0..K-1."""
    v = np.zeros((K, 12))
    for on, off, m in sop:
        for k in range(max(0, int(math.floor(on))), min(K, int(math.ceil(off)))):
            ov = min(off, k + 1) - max(on, k)
            if ov > 1e-9:
                v[k, m % 12] += ov
    return v


def partial_longest(sop, ons, j, cut):
    """Longest note heard in [j, cut) (same tie rule as soprano_obs), from notes with onset < cut."""
    i = bisect.bisect_left(ons, cut) - 1
    best, bl = None, 0.0
    cand = []
    while i >= 0 and sop[i][0] >= j - 16:
        cand.append(sop[i])
        i -= 1
    for on, off, m in reversed(cand):
        ov = min(off, cut) - max(on, j)
        if ov > bl + 1e-9:
            best, bl = m, ov
    return best


def keys_for(ctx, pos):
    """All lookup keys for a context tuple (oldest..newest), most specific first, grouped by n."""
    out = []
    for n in range(0, len(ctx) + 1):
        c = ctx[len(ctx) - n:]
        grp = []
        for w in range(-1, n):                                   # -1: exact
            code = 0
            for i, x in enumerate(c):
                code = code * 14 + (WILD if i == w else x)
            grp.append(((n * 8 + (w + 1)) * 8 + pos) * (14 ** NMAX) + code)
        out.append(grp)
    return out


class System:
    def __init__(self, train):
        self.model = ChordHMM().fit(train['bach'])
        ks, ws = [], []
        tv = bool(P['tv'])
        for name, songs in train.items():
            if name == 'essen' and not P['ess']:
                continue
            for s in songs:
                K, t = s['K'], s['tonic']
                pos = [b['pos'] % 8 for b in s['beats']] if s.get('beats') else [k % 4 for k in range(K)]
                lo = beat_longest(s['sop'], K)
                o = [REST if m is None else (m - t) % 12 for m in lo]
                V = np.roll(beat_vecs(s['sop'], K), -t, axis=1) if tv else None
                for k in range(K):
                    if o[k] == REST:
                        continue
                    ctx = tuple(o[max(0, k - NMAX):k])
                    if tv:
                        tg = [(pc, V[k, pc] / V[k].sum()) for pc in np.nonzero(V[k] > 1e-9)[0]]
                    else:
                        tg = [(o[k], 1.0)]
                    for grp in keys_for(ctx, pos[k]):
                        for key in grp:
                            for pc, w in tg:
                                ks.append(key * 16 + int(pc))
                                ws.append(w)
        a = np.array(ks, dtype=np.int64)
        self.kp, inv = np.unique(a, return_inverse=True)
        self.cnt = np.bincount(inv, weights=np.array(ws))
        self.note_guess = {}

    def _lookup(self, groups):
        flat = [k for g in groups for k in g]
        q = np.array(flat, dtype=np.int64) * 16
        lo = np.searchsorted(self.kp, q, 'left')
        hi = np.searchsorted(self.kp, q + 16, 'left')
        res, i = [], 0
        for g in groups:
            vs = []
            for _ in g:
                v = np.zeros(12)
                a, b = lo[i], hi[i]
                if b > a:
                    v[self.kp[a:b] % 16] = self.cnt[a:b]
                vs.append(v)
                i += 1
            res.append(vs)
        return res

    @staticmethod
    def _backoff(vecs_by_n, beta, wild):
        p = np.full(12, 1.0 / 12)
        Nlong = 0.0
        for n, vs in enumerate(vecs_by_n):
            c = vs[0] + (wild * sum(vs[1:]) / max(1, len(vs) - 1) if len(vs) > 1 else 0.0)
            N = c.sum()
            if N <= 0:
                continue
            lam = N / (N + beta)
            p = lam * c / N + (1 - lam) * p
            Nlong = N if n >= 2 else Nlong
        return p, Nlong

    def run(self, song, cfg):
        model, t, M, K, sop = self.model, song['tonic'], song['M'], song['K'], song['sop']
        pos = [b['pos'] for b in song['beats']]
        ztrue = [b['z'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        lo_a, hi_a = cfg.alto
        if P['lo'] >= 0:
            lo_a, hi_a = int(P['lo']), int(P['hi'])
        H = np.arange(lo_a, hi_a + 1)
        Hf = H.astype(float)
        ons = [n[0] for n in sop]
        full = beat_longest(sop, K)
        Vs = np.roll(beat_vecs(sop, K), -t, axis=1) if P['tv'] else None               # o_full[j] used only for j < jstar
        o_full = [REST if m is None else (m - t) % 12 for m in full]
        use_hmm = P['hmm'] > 0
        if use_hmm:
            def ll(pc):
                return model.loglik(pc, 'long', cfg.lam)
            pi, alpha = [], None
            for j in range(K):
                p = model.init if j == 0 else model.predict(alpha, pos[j], M)
                pi.append(p)
                alpha = model.update(p, ll(None if o_full[j] == REST else o_full[j]))
        mem = {}                                  # song memory: key -> 12 counts
        mem_upto = 0                              # beats whose target was added: targets j < mem_upto
        out, h_last, her_last = [], None, None
        for k in range(K):
            when = k - D
            jstar = math.floor(when)
            frac = when - jstar
            if jstar >= 0:
                part = partial_longest(sop, ons, jstar, when)              # partial beat jstar
                o = o_full[:jstar] + [REST if part is None else (part - t) % 12]
                for m in ([part] + [full[j] for j in range(jstar - 1, -1, -1)]):
                    if m is not None:
                        her_last = m
                        break
            else:
                o = []
            o = o + [REST] * (k - len(o))          # beats in between not heard yet (only if D > 1)
            # song memory: add targets j that are fully heard (j <= jstar - 1)
            while mem_upto <= jstar - 1:
                j = mem_upto
                if o_full[j] != REST:
                    ctx = tuple(o_full[max(0, j - NMAX):j])
                    for grp in keys_for(ctx, pos[j] % 8):
                        for key in grp:
                            v = mem.get(key)
                            if v is None:
                                v = mem[key] = np.zeros(12)
                            if Vs is not None:
                                v += Vs[j] / Vs[j].sum()
                            else:
                                v[o_full[j]] += 1.0
                mem_upto += 1
            ctx = tuple(o[max(0, k - NMAX):k])
            groups = keys_for(ctx, pos[k] % 8)
            pc_, _ = self._backoff(self._lookup(groups), P['beta'], P['wild'])
            sv = [[mem.get(key, np.zeros(12)) for key in g] for g in groups]
            ps, Ns = self._backoff(sv, P['sbeta'], P['wild'])
            ws = P['w_song'] * Ns / (Ns + P['s_b'])
            q = (1 - ws) * pc_ + ws * ps
            if use_hmm:
                if jstar >= 0:
                    post = model.update(pi[jstar], ll(None if o[jstar] == REST else o[jstar]))
                    for i in range(jstar + 1, k + 1):
                        post = model.predict(post, pos[i], M)
                else:
                    post = pi[k]
                m_hmm = post @ model.B
                q = q * (m_hmm + 1e-6) ** P['hmm']
            q = q / q.sum()
            self.note_guess[k] = int(np.argmax(q))
            # ---- choose h: expected consonance in real pitch
            ref = her_last if her_last is not None else 72
            mpc = np.array([ref + ((pc + t - ref + 6) % 12) - 6 for pc in range(12)])   # nearest pitch
            iv = np.abs(mpc[None, :] - H[:, None]) % 12
            gain = CONS[iv] - P['uni'] * (iv == 0) + P['bias'] * 0
            sc = gain @ q - P['vc'] * np.abs(Hf - ALTO_MID)
            if h_last is not None:
                sc = sc - P['vl'] * np.abs(Hf - h_last)
            if P['below'] and her_last is not None:
                ok = H <= her_last - 1
                if ok.any():
                    sc = np.where(ok, sc, -np.inf)
            h0 = int(H[int(np.argmax(sc))])
            hpc = (h0 - t) % 12
            e4 = CTM @ q
            zc = [z for z in range(N_CHORDS) if hpc in CT[z]]
            z0 = max(zc, key=lambda z: e4[z]) if zc else -1
            bt = Beat(k=k, pos=pos[k], z_true=ztrue[k], z_hat=z0, q=float(e4[z0]) if z0 >= 0 else 0.0)
            if k >= cfg.J:
                bt.action, bt.h = 'sing', h0
                h_last = h0
            out.append(bt)
        return out


def build(train):
    return System(train)
