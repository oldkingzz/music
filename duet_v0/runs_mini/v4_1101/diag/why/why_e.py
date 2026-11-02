"""why_e: two E3 probes as DIAGNOSTIC variants (no system file, no bench3 row), m1's chooser unchanged, val only.
  STYLE: a song-level hidden style (one per train corpus, each with its own transition AND emission table); its
         posterior uses the meter, a per-beat rhythm word (onsets in the beat, onset on the beat, rest) and/or how well
         each style's HMM predicted her pitches; chord belief = style-weighted mix (causal; never uses the dataset name).
  DUR:   chord-duration prior: state (chord, beats since it started, capped at 3), P(stay | bar position, duration).
  plus the ceilings: per-corpus models with the corpus KNOWN (oracle) at lam 1.0. Writes why_e_v1.json (mode 'x')."""
import sys
import time

import numpy as np

from why_lib import DS, S, FilterV, MixFilter, ModelV, bench3, headline, line, m1, prep, run, save

NAME = 'why_e_v1'
NW = 16
DMAX = 4


def rword(notes, k, end=None):
    n_on, at_k, cover = 0, 0, 0.0
    for j in range((len(notes) if end is None else end) - 1, -1, -1):
        on, off, pc = notes[j]
        if off is not None and off <= k:
            break
        if off is None or on >= k + 1:
            continue
        cover += min(off, k + 1) - max(on, k)
        if on >= k - 1e-9:
            n_on += 1
            at_k |= int(abs(on - k) < 1e-6)
    return min(n_on, 3) * 4 + at_k * 2 + int(1 - cover > 0.25)


def rhythm_model(train):
    lw, lm, metas = [], [], sorted({s['M'] for v in train.values() for s in v})
    for ds in DS:
        c = np.ones(NW)
        mc = {M: 1.0 for M in metas}
        for s in train[ds]:
            notes = sorted([list(n) for n in s['sop']], key=lambda n: n[0])
            mc[s['M']] += 1
            K = len(s['beats'])
            end = 0
            for k in range(K):
                while end < len(notes) and notes[end][0] < k + 1:
                    end += 1
                c[rword(notes, k, end)] += 1
        lw.append(np.log(c / c.sum()))
        tot = sum(mc.values())
        lm.append({M: np.log(v / tot) for M, v in mc.items()})
    rh = np.array(lw)
    mprior = {M: np.array([lm[j].get(M, np.log(1e-3)) for j in range(len(DS))]) for M in metas}
    return rh, mprior


class StyleFilter(MixFilter):
    def __init__(self, models, tables, rh, mprior, lam=1.0, eta_c=1.0, eta_r=1.0, use_m=True):
        super().__init__(models, tables, lam=lam, eta=eta_c)
        self.rh, self.mprior, self.eta_r, self.use_m = rh, mprior, eta_r, use_m

    def start(self, meta):
        super().start(meta)
        if self.use_m and meta['M'] in self.mprior:
            self.logw = self.logw + self.mprior[meta['M']]

    def _finish_beats(self, upto):
        if self.eta_r:
            for k in range(self.done, upto):
                self.logw = self.logw + self.eta_r * self.rh[:, rword(self.notes, k)]
        super()._finish_beats(upto)


def dur_model(train, A, only=None, bal=1.0, a=0.5):
    st = np.zeros((6, DMAX))
    n = np.zeros((6, DMAX))
    nmax = max(len(v) for v in train.values())
    for name, songs in train.items():
        if only is not None and name not in only:
            continue
        wd = 1.0 if only is not None else (nmax / len(songs)) ** bal
        for s in songs:
            z = [b['z'] for b in s['beats']]
            run_len = 0
            for k in range(1, len(z)):
                run_len = run_len + 1 if (k >= 2 and z[k - 1] >= 0 and z[k - 2] == z[k - 1]) else 0
                if z[k - 1] < 0 or z[k] < 0:
                    continue
                q = m1.tkey(s['beats'][k]['pos'], s['M'])
                d = min(run_len, DMAX - 1)
                n[q, d] += wd
                st[q, d] += wd * (z[k] == z[k - 1])
    ps = (st + a) / (n + 2 * a)
    C = A * (1 - np.eye(S))[None]
    C = C / C.sum(-1, keepdims=True)
    return ps, C


class DurFilter(FilterV):
    def __init__(self, model, ps, C, lam=1.5):
        super().__init__(model, lam)
        self.ps, self.C = ps, C

    def _advance(self, alpha, k, ll, a2=None):
        q = m1.tkey(self._pos(k), self.M)
        pr = np.zeros((S, DMAX))
        if alpha is None:
            pr[:, 0] = self.m.pi
        else:
            stay = alpha * self.ps[q][None, :]
            pr[:, 1:] += stay[:, :-1]
            pr[:, DMAX - 1] += stay[:, DMAX - 1]
            pr[:, 0] += (alpha * (1 - self.ps[q][None, :])).sum(1) @ self.C[q]
        w = pr * np.exp(ll - ll.max())[:, None]
        return w / w.sum()

    def belief(self, k):
        ll = self._beat_ll(k) * self.cur_w
        b = self._advance(self.alpha, k, ll).sum(1)
        self.last = b
        return b


def main():
    t0 = time.time()
    train = bench3.train_sets()
    base = ModelV(train)
    dsm = {ds: ModelV(train, only={ds}) for ds in DS}
    data = prep()
    rh, mprior = rhythm_model(train)
    print(f'rhythm model {time.time() - t0:.0f}s', flush=True)
    out = {'what': __doc__.strip().split('\n')[0], 'notes': 'val only (no test split); style order ' + ' '.join(DS),
           'meter_prior_per_style': {str(M): np.round(np.exp(v), 3).tolist() for M, v in mprior.items()},
           'variants': {}}
    models = [dsm[ds] for ds in DS]
    tabs = [dsm[ds].A for ds in DS]
    V = [('ORACLE_dsAB_lam1.0', dsm, lambda mo_: FilterV(mo_, 1.0)),
         ('STYLE_models_meter_rhythm_pitch_lam1.0', base,
          lambda mo_: StyleFilter(models, tabs, rh, mprior, 1.0, 1.0, 1.0)),
         ('STYLE_models_meter_rhythm_pitch_lam1.5', base,
          lambda mo_: StyleFilter(models, tabs, rh, mprior, 1.5, 1.0, 1.0)),
         ('STYLE_models_meter_rhythm_lam1.0', base,
          lambda mo_: StyleFilter(models, tabs, rh, mprior, 1.0, 0.0, 1.0)),
         ('STYLE_models_meter_rhythm_lam1.5', base,
          lambda mo_: StyleFilter(models, tabs, rh, mprior, 1.5, 0.0, 1.0)),
         ('STYLE_tables_meter_rhythm_lam1.0', base,
          lambda mo_: StyleFilter([base] * 4, tabs, rh, mprior, 1.0, 0.0, 1.0))]
    psP, CP = dur_model(train, base.A)
    out['dur_pstay_pooled_by_tkey_and_duration'] = np.round(psP, 3).tolist()
    V += [('DUR_pooled_lam1.5', base, lambda mo_: DurFilter(mo_, psP, CP, 1.5)),
          ('DUR_pooled_lam1.0', base, lambda mo_: DurFilter(mo_, psP, CP, 1.0))]
    dur_ds = {ds: dur_model(train, dsm[ds].A, only={ds}) for ds in DS}
    out['dur_pstay_per_dataset'] = {ds: np.round(v[0], 3).tolist() for ds, v in dur_ds.items()}
    for name, mo, fn in V:
        r = run(data, mo, fn)
        out['variants'][name] = {'rt': r}
        print(line(name, r), f'{time.time() - t0:.0f}s', flush=True)
    # duration prior with the corpus KNOWN (oracle): per-dataset P(stay | bar position, duration) and change table
    res = {}
    for ds in DS:
        sub = {ds: data[ds]}
        ps, C = dur_ds[ds]
        res[ds] = run(sub, base, lambda mo_, ps=ps, C=C: DurFilter(mo_, ps, C, 1.5))['ds'][ds]
    r = {'head': headline(res), 'ds': res}
    out['variants']['DUR_ORACLE_ds_lam1.5'] = {'rt': r}
    print(line('DUR_ORACLE_ds_lam1.5', r), flush=True)
    out['secs'] = round(time.time() - t0, 1)
    print('wrote', save(NAME, out))


if __name__ == '__main__':
    sys.exit(main())
