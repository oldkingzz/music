"""Compare tempo predictors on exact anchors (+50 ms hearing delay) in the rubato scenario."""
import numpy as np
import perform, singer
from follow import analyse, offline_notes
from run_v0 import SCENARIOS, find_gaps, match

xs, true_s = singer.render(singer.score_notes())
F = analyse(xs); gaps = find_gaps(F)
R = np.array([n['on'] for n in true_s])
JOIN = 9.88
DELAY = 0.05

def truth(perf):
    return np.array([perf.ref_to_live(r) for r in R])

def known(T, i, lat=DELAY):
    return [j for j in range(i) if T[j] + lat <= T[i] - 0.005]

def lsq(x, t, w):
    xm, tm = np.average(x, weights=w), np.average(t, weights=w)
    b = np.sum(w * (x - xm) * (t - tm)) / np.sum(w * (x - xm) ** 2)
    return tm - b * xm, b

def p_line(T, i, U=R, win=4.0, maxn=8, span=0.8, wmin=0.5):
    k = known(T, i); pts = [j for j in k if T[k[-1]] - T[j] <= win][-maxn:]
    x, t = U[pts], T[pts]; w = np.linspace(wmin, 1, len(pts))
    if len(pts) >= 2 and x.max() - x.min() >= span:
        a, b = lsq(x, t, w); b = np.clip(b, 0.6, 1.8)
    else:
        b = 1.0
    a = np.average(t - b * x, weights=w)
    return a + b * U[i]

def p_ioi(T, i, alpha=0.5):
    k = known(T, i); b = 1.0
    for j0, j1 in zip(k[:-1], k[1:]):
        if R[j1] - R[j0] > 0.15:
            b = alpha * (T[j1] - T[j0]) / (R[j1] - R[j0]) + (1 - alpha) * b
    j = k[-1]
    return T[j] + np.clip(b, 0.6, 1.8) * (R[i] - R[j])

def profile_from(perf_reh, smooth=0.6):
    r = np.arange(0, R[-1] + 2, 0.01)
    t = np.array([perf_reh.ref_to_live(x) for x in r])
    ok = ~np.isnan(t)
    rho = np.gradient(np.interp(r, r[ok], t[ok]), r)
    k = int(smooth / 0.01); rho = np.convolve(rho, np.ones(k) / k, 'same')
    u = np.cumsum(rho) * 0.01
    return np.interp(R, r, u)

def score(fn, T):
    e = [(fn(T, i) - T[i]) * 1000 for i in range(len(R)) if R[i] >= JOIN]
    a = np.abs(e); return f'median {np.median(a):4.0f}  p90 {np.percentile(a, 90):4.0f}  <=30 {np.mean(a<=30):4.0%}'

for sc in ('steady', 'rubato'):
    perf = perform.make(xs, gaps, **SCENARIOS[sc]); T = truth(perf)
    print(f'== {sc}')
    for name, fn in [('line 4s/8 (current)', p_line),
                     ('line 2s/4', lambda T, i: p_line(T, i, win=2.0, maxn=4, span=0.5)),
                     ('line 1.5s/3', lambda T, i: p_line(T, i, win=1.5, maxn=3, span=0.4)),
                     ('ioi a=0.5', p_ioi), ('ioi a=0.7', lambda T, i: p_ioi(T, i, 0.7))]:
        print(f'  {name:22s}', score(fn, T))
    for reh_phase, reh_rit, tag in ((0.7, 0.30, 'same sway phase'), (2.5, 0.25, 'sway phase differs, rit 0.25'), (0.7, 0.0, 'rehearsal had no rit')):
        reh = perform.make(xs, gaps, sway=0.07, period=8.0, rit=reh_rit, phase=reh_phase, seed=9)
        U = profile_from(reh)
        print(f'  rehearsal profile ({tag}):')
        for name, fn in [('  + line 4s/8', lambda T, i: p_line(T, i, U=U)),
                         ('  + line 2s/4', lambda T, i: p_line(T, i, U=U, win=2.0, maxn=4, span=0.5))]:
            print(f'    {name:20s}', score(fn, T))
