"""Debug: per planned note, what did the predictor know when it committed?"""
import sys
import numpy as np
import perform, singer
from dsp import HOP, SR
from follow import Accompanist, FeatureTracker, Follower, analyse, offline_notes, plan_harmony
from run_v0 import SCENARIOS, calibrate, find_gaps, match

name = sys.argv[1] if len(sys.argv) > 1 else 'natural'
xs, true_s = singer.render(singer.score_notes())
F = analyse(xs); det = offline_notes(F)
rb, lb, _ = calibrate(xs, true_s, F, det, [(a+b)/2 for a,b in find_gaps(F)])
true_on = [n['on'] for n in true_s]
truth_on = [true_on[k] for k in match([n['on'] for n in det], true_on)]
for nt in det: nt['on'] -= rb.get(nt['kind'], 0)
plan, key = plan_harmony(det); gaps = find_gaps(F)
spec = dict(SCENARIOS[name]); spec.pop('rehearsal', None)
perf = perform.make(xs, gaps, **spec); y = perf.audio
ft, fol = FeatureTracker(), Follower(F)
acc = Accompanist(det, plan, 9.88, bias_live=lb, breaths=[(a+b)/2 for a,b in gaps])
for k in range(len(y)//HOP):
    now=(k+1)*HOP/SR; ft.step(y[k*HOP:(k+1)*HOP]); fol.step(ft.m[-1], ft.v[-1], ft.nov[-1]); acc.step(now, ft, fol, 0.0002)
T = {i: perf.ref_to_live(truth_on[i]) for i in range(len(det))}
anch = {r: t for t, r in acc.anchors}
for j, p in enumerate(acc.plan):
    i = p['note']; v = acc.pred.get(j, {})
    known = [(round(t,2), det.index(next(n for n in det if n['on']==r))) for t, r in acc.anchors if t + 0.0 <= v.get('sent', 0)]
    last = known[-3:]
    e = (v['at'] - T[i]) * 1000 if v.get('status') == 'played' and T[i]==T[i] else None
    ae = (anch[det[i]['on']] - T[i]) * 1000 if det[i]['on'] in anch and T[i]==T[i] else None
    print(f"{v.get('status','-'):7s} j{j:2d} note{i:2d} {det[i]['kind']:6s} truth {T[i]:6.2f} sent {v.get('sent',0):6.2f} at {v.get('at',0):6.2f} "
          f"err {str(None if e is None else round(e)):>5} b {v.get('b',0):.3f} anchor_err {str(None if ae is None else round(ae)):>4} last anchors(note) {[n for _, n in last]}")
