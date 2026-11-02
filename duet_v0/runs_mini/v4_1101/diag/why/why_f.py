"""why_f: the E3 probe that why_e points to, as a DIAGNOSTIC variant (no system file, no bench3 row), val only.
  'Pop-aware HMM': styles = {m1's pooled HMM, an HMM trained on pop only, one on pop909 only} (or {pooled, pop+pop909});
  song-level style posterior from the meter and the rhythm of finished beats (+ optionally how well each style's HMM
  predicted her pitches); chord belief = style-weighted mix; m1's chooser unchanged. Causal: never uses the dataset name.
  Writes why_f_v1.json (mode 'x')."""
import sys
import time

import numpy as np

from why_e import NW, StyleFilter, rword
from why_lib import DS, ModelV, bench3, finish, headline, line, new_acc, prep, save, score_song

NAME = 'why_f_v1'


def rhythm_groups(train, groups):
    metas = sorted({s['M'] for v in train.values() for s in v})
    lw, lm = [], []
    for g in groups:
        c = np.ones(NW)
        mc = {M: 1.0 for M in metas}
        for ds in g:
            for s in train[ds]:
                notes = sorted([list(n) for n in s['sop']], key=lambda n: n[0])
                mc[s['M']] += 1
                end = 0
                for k in range(len(s['beats'])):
                    while end < len(notes) and notes[end][0] < k + 1:
                        end += 1
                    c[rword(notes, k, end)] += 1
        lw.append(np.log(c / c.sum()))
        tot = sum(mc.values())
        lm.append({M: np.log(v / tot) for M, v in mc.items()})
    return np.array(lw), {M: np.array([lm[j][M] for j in range(len(groups))]) for M in metas}


def run_style(data, fn, smap):
    res = {}
    for ds, L in data.items():
        f = fn()
        r = new_acc()
        for d in L:
            f.set_song(None, None, d['durs'])
            f.start(d['meta'])
            outs, bels, wts, prev = [], [], [], None
            for i, (on, off, m) in enumerate(d['heard']):
                outs.append(int(f.step(i, float(on), int(m), None if prev is None else float(prev[1] - prev[0]))))
                bels.append(f.last)
                wts.append(f.lastw)
                prev = (on, off)
            score_song(d, outs, bels, r, wts, smap[ds])
        res[ds] = finish(r)
    return {'head': headline(res), 'ds': res}


def main():
    t0 = time.time()
    train = bench3.train_sets()
    base = ModelV(train)
    pop = ModelV(train, only={'pop'})
    p909 = ModelV(train, only={'pop909'})
    popall = ModelV(train, only={'pop', 'pop909'})
    data = prep()
    out = {'what': __doc__.strip().split('\n')[0], 'notes': 'val only (no test split)', 'variants': {}}
    g3 = [('bach', 'folk'), ('pop',), ('pop909',)]
    g2 = [('bach', 'folk'), ('pop', 'pop909')]
    rh3, mp3 = rhythm_groups(train, g3)
    rh2, mp2 = rhythm_groups(train, g2)
    s3 = {'bach': 0, 'folk': 0, 'pop': 1, 'pop909': 2}
    s2 = {'bach': 0, 'folk': 0, 'pop': 1, 'pop909': 1}
    M3, T3 = [base, pop, p909], [base.A, pop.A, p909.A]
    M2, T2 = [base, popall], [base.A, popall.A]
    V = []
    for lam in (1.0, 1.5):
        for ec in (0.0, 1.0):
            V.append((f'POP3_lam{lam}_pitch{int(ec)}', lambda lam=lam, ec=ec: StyleFilter(M3, T3, rh3, mp3, lam, ec, 1.0), s3))
            V.append((f'POP2_lam{lam}_pitch{int(ec)}', lambda lam=lam, ec=ec: StyleFilter(M2, T2, rh2, mp2, lam, ec, 1.0), s2))
    for name, fn, smap in V:
        r = run_style(data, fn, smap)
        out['variants'][name] = {'rt': r}
        print(line(name, r), f'{time.time() - t0:.0f}s', flush=True)
    out['secs'] = round(time.time() - t0, 1)
    print('wrote', save(NAME, out))


if __name__ == '__main__':
    sys.exit(main())
