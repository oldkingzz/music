"""why_c: is it the EMISSION? m1's HMM (pooled transitions, chooser unchanged) with other evidence models, val only.
  tmplS: P(pc | chord) = S/3 on each chord tone, (1-S)/9 elsewhere (then m1's 10 % mishearing floor);
  accW:  notes that START off the beat count W x their length (beat-onset / on-beat notes count fully);
  blendA: learned B and the template mixed in log space. Real time and offline smoothing; per-beat emission-only
  accuracy with the chord-frequency prior. Writes why_c_v1.json (mode 'x')."""
import sys
import time

import numpy as np

from why_lib import (CT, DS, EXPECT_RT, FilterV, J, ModelV, S, bench3, line, m1, prep, run, save, sticky)

NAME = 'why_c_v1'
CTM = m1.CTM


class Em:
    def __init__(self, base, Q, noise=0.1, A=None):
        self.A = base.A if A is None else A
        self.pi = base.pi
        self.Bc = Q
        self.logB = np.log((1 - noise) * Q + noise / 12)
        self.logBc = np.log(Q)


def tmpl(s):
    return CTM * (s / 3) + (1 - CTM) * ((1 - s) / 9)


def blend(base, s, a):
    Q = np.exp(a * np.log(base.Bc) + (1 - a) * np.log(tmpl(s)))
    return Q / Q.sum(1, keepdims=True)


def off_beat(on):
    return abs(on - round(on)) > 1e-6


class AccFilter(FilterV):
    def __init__(self, model, lam=1.5, w_off=0.5):
        super().__init__(model, lam)
        self.w_off = w_off

    def _beat_v(self, k):
        v = np.zeros(12)
        for on, off, pc in reversed(self.notes):
            e = off if off is not None else self._cur_end(k, on)
            if e <= k:
                break
            ov = min(e, k + 1) - max(on, k)
            if ov > 0:
                v[pc] += ov * (self.w_off if off_beat(on) else 1.0)
        return v


def beat_obs_w(rel, K, w_off):
    obs = np.zeros((K, 12))
    for on, off, m in rel:
        k0, k1 = int(on), int(np.ceil(off))
        for k in range(max(k0, 0), min(k1, K)):
            ov = min(off, k + 1) - max(on, k)
            if ov > 0:
                obs[k, m % 12] += ov * (w_off if off_beat(on) else 1.0)
    return obs


def smooth_w(model, A, lam, hrel, pos, M, w_off=1.0):
    K = len(pos)
    obs = beat_obs_w(hrel, K, w_off)
    ll = np.array([lam * (model.logB @ o) if o.sum() > 0 else np.zeros(S) for o in obs])
    e = np.exp(ll - ll.max(1, keepdims=True))
    al = np.zeros((K, S))
    for k in range(K):
        pr = model.pi if k == 0 else al[k - 1] @ A[m1.tkey(pos[k], M)]
        w = pr * e[k]
        al[k] = w / w.sum()
    be = np.ones((K, S))
    for k in range(K - 2, -1, -1):
        w = A[m1.tkey(pos[k + 1], M)] @ (e[k + 1] * be[k + 1])
        be[k] = w / w.sum()
    g = al * be
    return g / g.sum(1, keepdims=True)


def em_prior_acc(data, models, prior):
    """per-beat emission-only accuracy (HEARD melody) with the chord-frequency prior."""
    out = {}
    lp = np.log(prior)
    for ds, L in data.items():
        n = 0
        ok = {k: 0 for k in models}
        for d in L:
            z = d['z']
            oh = m1.beat_obs(d['hrel'], len(d['pos']))
            for k in range(J, len(z)):
                if z[k] < 0 or oh[k].sum() <= 0:
                    continue
                n += 1
                for nm, mo in models.items():
                    ok[nm] += int(np.argmax(lp + 1.5 * (mo.logB @ oh[k])) == z[k])
        out[ds] = {nm: v / n for nm, v in ok.items()}
    return out


def main():
    t0 = time.time()
    train = bench3.train_sets()
    base = ModelV(train)
    dsm = {ds: ModelV(train, only={ds}) for ds in DS}
    data = prep()
    tm = {s: Em(base, tmpl(s)) for s in (0.7, 0.8, 0.9)}
    prior = base.Bc.sum(1) * 0
    tot = sum(dsm[ds].Acount.sum((0, 1)) for ds in DS)            # chord frequency (unweighted beats)
    prior = (tot + 1) / (tot + 1).sum()
    out = {'what': __doc__.strip().split('\n')[0], 'notes': 'val only (no test split)',
           'learned_Bc_rows_I_ii_iii_IV_V_vi_vii': np.round(base.Bc, 3).tolist(),
           'learned_Bc_chord_tone_mass': np.round((base.Bc * CTM).sum(1), 3).tolist(),
           'chord_prior_used': np.round(prior, 3).tolist()}
    out['emission_with_prior_beat_acc'] = em_prior_acc(data, {'learned': base, 'tmpl70': tm[0.7], 'tmpl80': tm[0.8],
                                                               'tmpl90': tm[0.9]}, prior)
    print('Bc chord-tone mass', out['learned_Bc_chord_tone_mass'], flush=True)
    print('emission+prior per-beat acc', {ds: {k: round(x, 3) for k, x in v.items()}
                                          for ds, v in out['emission_with_prior_beat_acc'].items()}, flush=True)
    st85 = sticky(base.A, 0.85)
    dsA = {ds: dsm[ds].A for ds in DS}
    V = [('base', base, None, 1.5, None)]
    for s in (0.7, 0.8, 0.9):
        for lam in (1.0, 1.5, 2.0):
            V.append((f'tmpl{int(s * 100)}_lam{lam}', tm[s], None, lam, None))
    for a in (0.5,):
        for s in (0.8, 0.9):
            V.append((f'blend{int(s * 100)}a{a}_lam1.5', Em(base, blend(base, s, a)), None, 1.5, None))
    for w in (0.25, 0.5):
        V.append((f'acc{w}_learned_lam1.5', base, None, 1.5, w))
        V.append((f'acc{w}_tmpl80_lam1.5', tm[0.8], None, 1.5, w))
        V.append((f'acc{w}_tmpl90_lam1.5', tm[0.9], None, 1.5, w))
    V += [('tmpl80_sticky85_lam1.5', tm[0.8], st85, 1.5, None),
          ('tmpl80_dsA_oracle_lam1.5', tm[0.8], dsA, 1.5, None),
          ('tmpl90_dsA_oracle_lam1.5', tm[0.9], dsA, 1.5, None)]
    out['variants'] = {}
    for name, mo, tb, lam, w in V:
        if time.time() - t0 > 520:
            out['stopped_early_before'] = name
            break
        if w is None:
            fn = (lambda mo_, lam=lam: FilterV(mo_, lam))
            sm_fn = (lambda mo_, A_, d, lam=lam: smooth_w(mo_, mo_.A if A_ is None else A_, lam, d['hrel'], d['pos'],
                                                            d['M']))
        else:
            fn = (lambda mo_, lam=lam, w=w: AccFilter(mo_, lam, w))
            sm_fn = (lambda mo_, A_, d, lam=lam, w=w: smooth_w(mo_, mo_.A if A_ is None else A_, lam, d['hrel'],
                                                                 d['pos'], d['M'], w))
        rt = run(data, mo, fn, tables=tb)
        sm = run(data, mo, fn, tables=tb, smooth_fn=sm_fn)
        out['variants'][name] = {'lam': lam, 'w_off': w, 'rt': rt, 'smooth': sm}
        print(line(name + ' rt', rt), f'{time.time() - t0:.0f}s', flush=True)
        print(line(name + ' sm', sm), flush=True)
    out['reproduces_m1_rt'] = abs(out['variants']['base']['rt']['head']['music_nd'] - EXPECT_RT) < 1e-9
    out['secs'] = round(time.time() - t0, 1)
    print('reproduces m1 rt:', out['reproduces_m1_rt'], out['secs'], 's')
    print('wrote', save(NAME, out))


if __name__ == '__main__':
    sys.exit(main())
