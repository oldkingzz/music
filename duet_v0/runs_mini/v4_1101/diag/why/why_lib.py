"""Shared code: why does m1_filter_v5lo's real-time chord guess not 'lock on'? (v4_1101 R1 analyst; edits nothing)

Controlled variants of m1_filter_v5lo's Model / Filter. The chooser (m1.Filter.step) is m1's, unchanged; only the
chord belief it is handed changes. Replays bench3.run_song exactly (heard notes, 10 % mishearing, seed ^ 5160),
val only (the bundle has no test split). music_nd exactly as rescore_nd.py. Accuracy / note-index curve exactly
as runs_mini/v4_1101/tools/diag_rt.py (argmax belief at her note onset vs the chord of the onset beat).
"""
import json
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402
from systems import m1_filter_v5lo as m1  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v4_1101', 'diag', 'why')
GOOD = CONSONANT - {0}
J = bench3.J
S = m1.S
DS = ('bach', 'folk', 'pop', 'pop909')
EXPECT_RT = 0.620637625406759
EXPECT_SMOOTH = 0.622872585374413
OFF_DOWN = (1, 2, 3, 5)          # tkeys that are not a downbeat (4/4 beats 2-4; other meters' off-beats)


# ------------------------------------------------------------------------------------------------ model
class ModelV:
    """m1.Model's counting with its knobs as arguments. only = set of train datasets to count (each song weight 1)."""

    def __init__(self, train, bal=1.0, kappa=0.5, noise=0.1, only=None):
        A = np.zeros((6, S, S))
        B = np.zeros((S, 12))
        pi = np.zeros(S)
        nmax = max(len(v) for v in train.values())
        for name, songs in train.items():
            if only is not None and name not in only:
                continue
            wd = 1.0 if only is not None else (nmax / len(songs)) ** bal
            for s in songs:
                t, M = s['tonic'], s['M']
                z = [b['z'] for b in s['beats']]
                K = len(z)
                rel = [[on, off, (m - t) % 12] for on, off, m in s['sop']]
                obs = m1.beat_obs(rel, K)
                first = True
                for k in range(K):
                    if z[k] < 0:
                        continue
                    B[z[k]] += wd * obs[k]
                    if first:
                        pi[z[k]] += wd
                        first = False
                    if k >= 1 and z[k - 1] >= 0:
                        A[m1.tkey(s['beats'][k]['pos'], M), z[k - 1], z[k]] += wd
        self.Acount = A
        self.A = (A + kappa) / (A + kappa).sum(-1, keepdims=True)
        Bc = (B + 1.0) / (B + 1.0).sum(-1, keepdims=True)
        self.Bc = Bc
        self.logB = np.log((1 - noise) * Bc + noise / 12)
        self.logBc = np.log(Bc)
        self.pi = (pi + 1) / (pi + 1).sum()


def sticky(A, s, keys=OFF_DOWN):
    """Raise the mean P(stay) of the off-downbeat tables to s by shrinking every row's off-diagonal mass."""
    A = A.copy()
    for q in keys:
        d = float(np.diag(A[q]).mean())
        if d >= s:
            continue
        f = (1 - s) / (1 - d)
        off = A[q] * (1 - np.eye(S)) * f
        A[q] = off + np.diag(1 - off.sum(1))
    return A


def pstay(A):
    return [round(float(np.diag(A[q]).mean()), 3) for q in range(6)]


# ------------------------------------------------------------------------------------------------ filter
class FilterV(m1.Filter):
    """m1.Filter with lam, cur_w, the current-note assumption and the transition table as settings.
    cur_mode: 'eob' (m1: her current note lasts to the end of the beat), 'true' (ORACLE: its real length),
    'med' (causal: the median length of her last 8 notes). gamma given -> belief = offline smoothing."""

    def __init__(self, model, lam=1.5, cur_w=1.0, cur_mode='eob'):
        super().__init__(model)
        self.lam, self.cur_w, self.cur_mode = lam, cur_w, cur_mode
        self.A, self.gamma, self.durs, self.last = model.A, None, None, None

    def set_song(self, A=None, gamma=None, durs=None):
        self.A = self.m.A if A is None else A
        self.gamma, self.durs = gamma, durs

    def _cur_end(self, k, on):
        if self.cur_mode == 'eob':
            return k + 1
        if self.cur_mode == 'true':
            return on + self.durs[len(self.notes) - 1]
        if self.cur_mode == 'med':
            d = [n[1] - n[0] for n in self.notes[-9:-1] if n[1] is not None]
            return on + (float(np.median(d)) if d else 1.0)
        raise ValueError(self.cur_mode)

    def _beat_v(self, k):
        v = np.zeros(12)
        for on, off, pc in reversed(self.notes):
            e = off if off is not None else self._cur_end(k, on)
            if e <= k:
                break
            ov = min(e, k + 1) - max(on, k)
            if ov > 0:
                v[pc] += ov
        return v

    def _beat_ll(self, k, cur_end=None):
        v = self._beat_v(k)
        if v.sum() <= 0:
            return np.zeros(S)
        return self.lam * (self.m.logB @ v)

    def _advance(self, alpha, k, ll, a2=None):
        pr = self.m.pi.copy() if alpha is None else alpha @ self.A[m1.tkey(self._pos(k), self.M)]
        w = pr * np.exp(ll - ll.max())
        return w / w.sum()

    def belief(self, k):
        if self.gamma is not None and k < len(self.gamma):
            b = self.gamma[k]
        else:
            ll = self._beat_ll(k) * self.cur_w
            b = self._advance(self.alpha, k, ll)
        self.last = b
        return b


class MixFilter(FilterV):
    """CAUSAL mixture of HMMs: one forward filter per (model, table); weight of each = P(it | her notes so far),
    updated beat by beat with its predictive likelihood (eta tempers it). Belief = weighted mix."""

    def __init__(self, models, tables, lam=1.5, cur_w=1.0, cur_mode='eob', eta=1.0, logprior=None):
        super().__init__(models[0], lam, cur_w, cur_mode)
        self.models, self.tables, self.eta = models, tables, eta
        n = len(tables)
        self.logprior = np.full(n, -np.log(n)) if logprior is None else np.asarray(logprior, float)
        self.lastw = None

    def set_song(self, A=None, gamma=None, durs=None):
        self.gamma, self.durs = gamma, durs

    def start(self, meta):
        super().start(meta)
        self.alphas = [None] * len(self.tables)
        self.logw = self.logprior.copy()

    def _step_j(self, j, k, v, wcur=1.0):
        mo = self.models[j]
        ll = (self.lam * wcur) * (mo.logB @ v) if v.sum() > 0 else np.zeros(S)
        a = self.alphas[j]
        pr = mo.pi.copy() if a is None else a @ self.tables[j][m1.tkey(self._pos(k), self.M)]
        w = pr * np.exp(ll - ll.max())
        c = w.sum()
        return w / c, np.log(c) + ll.max()

    def _finish_beats(self, upto):
        while self.done < upto:
            k = self.done
            v = self._beat_v(k)
            for j in range(len(self.tables)):
                self.alphas[j], lp = self._step_j(j, k, v)
                self.logw[j] += self.eta * lp
            self.done += 1

    def belief(self, k):
        if self.gamma is not None and k < len(self.gamma):
            b = self.gamma[k]
            self.last = b
            return b
        v = self._beat_v(k)
        bs, lw = [], []
        for j in range(len(self.tables)):
            bj, lp = self._step_j(j, k, v, self.cur_w)
            bs.append(bj)
            lw.append(self.logw[j] + self.eta * lp)
        lw = np.array(lw)
        wt = np.exp(lw - lw.max())
        wt /= wt.sum()
        b = wt @ np.array(bs)
        self.last, self.lastw = b, wt
        return b


# ------------------------------------------------------------------------------------------------ smoothing
def smooth(model, A, lam, hrel, pos, M):
    """HMM forward-backward over the whole HEARD song (NOT causal). Returns (gamma, log-likelihood)."""
    K = len(pos)
    obs = m1.beat_obs(hrel, K)
    ll = np.array([lam * (model.logB @ o) if o.sum() > 0 else np.zeros(S) for o in obs])
    mx = ll.max(1, keepdims=True)
    e = np.exp(ll - mx)
    al = np.zeros((K, S))
    lik = 0.0
    for k in range(K):
        pr = model.pi if k == 0 else al[k - 1] @ A[m1.tkey(pos[k], M)]
        w = pr * e[k]
        c = w.sum()
        lik += np.log(c) + mx[k, 0]
        al[k] = w / c
    be = np.ones((K, S))
    for k in range(K - 2, -1, -1):
        w = A[m1.tkey(pos[k + 1], M)] @ (e[k + 1] * be[k + 1])
        be[k] = w / w.sum()
    g = al * be
    return g / g.sum(1, keepdims=True), lik


# ------------------------------------------------------------------------------------------------ data
def prep():
    data = {}
    for ds, songs in bench3.eval_sets('val').items():
        L = []
        for song in songs:
            h = bench3.B2.heard(song)
            heard = sorted(h['sop'], key=lambda n: n[0])
            true = sorted(song['sop'], key=lambda n: n[0])
            t = song['tonic']
            z = [b['z'] for b in song['beats']]
            pos = [b['pos'] for b in song['beats']]
            span, sid = [], -1
            for k in range(len(z)):
                if z[k] < 0:
                    span.append(None)
                    continue
                if k == 0 or z[k - 1] != z[k]:
                    sid += 1
                span.append(sid)
            L.append(dict(heard=heard, true=true, t=t, z=z, pos=pos, M=song['M'], span=span,
                          meta={'tonic': t, 'M': song['M'], 'K': song['K'], 'pos': pos},
                          hrel=[[on, off, (m - t) % 12] for on, off, m in h['sop']],
                          trel=[[on, off, (m - t) % 12] for on, off, m in song['sop']],
                          durs=[float(off - on) for on, off, m in heard]))
        data[ds] = L
    return data


# ------------------------------------------------------------------------------------------------ scoring
def new_acc():
    return dict(num=0.0, den=0.0, dbl=0.0, n=0, ok=0, pt=0.0, idx=[[0, 0] for _ in range(4)],
                down=[0, 0], other=[0, 0], wid=[0, 0])


def score_song(d, outs, bels, r, wts=None, ds_index=None):
    z, pos, t, span = d['z'], d['pos'], d['t'], d['span']
    seen = {}
    for i, (on, off, m) in enumerate(d['true']):
        k0 = int(on)
        z0 = z[k0] if k0 < len(z) else -1
        if z0 >= 0:
            seen[span[k0]] = seen.get(span[k0], 0) + 1
            if on >= J:
                b = bels[i]
                ok = int(int(np.argmax(b)) == z0)
                r['n'] += 1
                r['ok'] += ok
                r['pt'] += float(b[z0])
                q = r['idx'][min(seen[span[k0]], 4) - 1]
                q[0] += 1
                q[1] += ok
                q = r['down'] if pos[k0] == 0 else r['other']
                q[0] += 1
                q[1] += ok
                if wts is not None and wts[i] is not None:
                    r['wid'][0] += 1
                    r['wid'][1] += int(int(np.argmax(wts[i])) == ds_index)
        h = outs[i]
        for k in range(max(J, k0), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            r['den'] += ov
            if h >= 0:
                iv = abs(m - h) % 12
                r['num'] += ov * (iv in GOOD and (h - t) % 12 in CT[z[k]])
                r['dbl'] += ov * (iv == 0)


def finish(r):
    o = {'music_nd': r['num'] / r['den'], 'double': r['dbl'] / r['den'], 'acc': r['ok'] / r['n'],
         'p_true': r['pt'] / r['n'], 'idx_acc': [q[1] / max(q[0], 1) for q in r['idx']],
         'idx_n': [q[0] for q in r['idx']], 'acc_down': r['down'][1] / max(r['down'][0], 1),
         'acc_other': r['other'][1] / max(r['other'][0], 1), 'notes': r['n']}
    if r['wid'][0]:
        o['style_id_acc'] = r['wid'][1] / r['wid'][0]
    return o


def headline(res):
    n = len(res)
    h = {k: sum(v[k] for v in res.values()) / n for k in ('music_nd', 'acc', 'p_true', 'acc_down', 'acc_other')}
    h['idx_acc'] = [sum(v['idx_acc'][i] for v in res.values()) / n for i in range(4)]
    h['double_max'] = max(v['double'] for v in res.values())
    if all('style_id_acc' in v for v in res.values()):
        h['style_id_acc'] = sum(v['style_id_acc'] for v in res.values()) / n
    return h


def run(data, models, filt_fn, *, tables=None, smooth_fn=None):
    """models / tables: one object or a dict ds -> object (ORACLE: uses the dataset name).
    filt_fn(model) -> a FilterV. smooth_fn(model, table, d) -> gamma (offline) or None (real time)."""
    res = {}
    for di, (ds, L) in enumerate(data.items()):
        mo = models[ds] if isinstance(models, dict) else models
        A = tables[ds] if isinstance(tables, dict) else tables
        f = filt_fn(mo)
        r = new_acc()
        for d in L:
            gamma = smooth_fn(mo, A, d) if smooth_fn else None
            f.set_song(A, gamma, d['durs'])
            f.start(d['meta'])
            outs, bels, wts, prev = [], [], [], None
            for i, (on, off, m) in enumerate(d['heard']):
                outs.append(int(f.step(i, float(on), int(m), None if prev is None else float(prev[1] - prev[0]))))
                bels.append(f.last)
                wts.append(getattr(f, 'lastw', None))
                prev = (on, off)
            score_song(d, outs, bels, r, wts, di)
        res[ds] = finish(r)
    return {'head': headline(res), 'ds': res}


def line(name, out):
    h = out['head']
    s = (f"{name:22s} nd {h['music_nd']:.4f} [" + ' '.join(f"{v['music_nd']:.3f}" for v in out['ds'].values())
         + f"] acc {h['acc']:.3f} [" + ' '.join(f"{v['acc']:.3f}" for v in out['ds'].values())
         + '] idx ' + '/'.join(f"{x:.2f}" for x in h['idx_acc']))
    if 'style_id_acc' in h:
        s += f" styleid {h['style_id_acc']:.2f}"
    return s


def save(name, obj):
    path = os.path.join(OUT, name + '.json')
    with open(path, 'x') as fh:
        json.dump(obj, fh, indent=1, default=float)
    return path


def tick():
    return time.time()
