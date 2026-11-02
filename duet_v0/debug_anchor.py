"""Debug: how accurate are the anchors, and how good would prediction be with perfect anchors?"""
import sys
import numpy as np
import perform, singer
from dsp import HOP, SR
from follow import Accompanist, FeatureTracker, Follower, analyse, offline_notes, plan_harmony
from run_v0 import SCENARIOS, calibrate, find_gaps, match

name = sys.argv[1] if len(sys.argv) > 1 else 'rubato'
xs, true_s = singer.render(singer.score_notes())
F = analyse(xs); det = offline_notes(F)
rb, lb, _ = calibrate(xs, true_s, F, det)
true_on = [n['on'] for n in true_s]
truth_on = [true_on[k] if k is not None else None for k in match([n['on'] for n in det], true_on)]
for nt in det: nt['on'] -= rb.get(nt['kind'], 0)
plan, key = plan_harmony(det); gaps = find_gaps(F)
perf = perform.make(xs, gaps, **SCENARIOS[name]); y = perf.audio
ft, fol = FeatureTracker(), Follower(F)
acc = Accompanist(det, plan, 9.88, bias_live=lb)
for k in range(len(y)//HOP):
    now=(k+1)*HOP/SR; ft.step(y[k*HOP:(k+1)*HOP]); fol.step(ft.m[-1], ft.v[-1], ft.nov[-1]); acc.step(now, ft, fol, 0.0002)
# anchor accuracy
r2k = {nt['on']: i for i, nt in enumerate(det)}
errs = []
for t, r in acc.anchors:
    i = r2k[r]; tt = perf.ref_to_live(truth_on[i])
    errs.append((t-tt)*1000)
errs = np.array(errs)
print(f'anchors {len(acc.anchors)}/{len(det)}  err median {np.median(errs):.1f} ms  |err| p90 {np.percentile(abs(errs),90):.1f}  max {abs(errs).max():.0f}')
# oracle: perfect anchors for every note, same fitting rule, predict next notes 1 note ahead with info cutoff
T_true = np.array([perf.ref_to_live(truth_on[i]) for i in range(len(det))])
R = np.array([nt['on'] for nt in det])
def predict(i, lat, win=4.0, maxn=8, span=0.8):
    known = [j for j in range(i) if T_true[j] + lat <= T_true[i] - 0.005]  # heard before it's needed
    pts = [j for j in known if T_true[known[-1]] - T_true[j] <= win][-maxn:]
    Tt, Rr = T_true[pts], R[pts]; w = np.linspace(0.5,1,len(pts)); b=1.0
    if len(pts)>=2 and Rr.max()-Rr.min()>=span:
        rm,tm=np.average(Rr,weights=w),np.average(Tt,weights=w); b=np.clip(np.sum(w*(Rr-rm)*(Tt-tm))/np.sum(w*(Rr-rm)**2),.6,1.8)
    a=np.average(Tt-b*Rr,weights=w); return a+b*R[i]
for lat in (0.0, 0.05, 0.09):
    e=[ (predict(i,lat)-T_true[i])*1000 for i in range(5,len(det)) if R[i]>=9.88]
    print(f'oracle anchors, detect delay {lat*1000:.0f} ms: median |e| {np.median(np.abs(e)):.0f}  p90 {np.percentile(np.abs(e),90):.0f}')
for win,maxn in ((2.0,4),(1.5,3),(3.0,6)):
    e=[ (predict(i,0.05,win,maxn,0.5)-T_true[i])*1000 for i in range(5,len(det)) if R[i]>=9.88]
    print(f'oracle, delay 50, window {win}s/{maxn}: median |e| {np.median(np.abs(e)):.0f}  p90 {np.percentile(np.abs(e),90):.0f}')
