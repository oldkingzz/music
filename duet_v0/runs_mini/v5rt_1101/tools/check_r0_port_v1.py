"""v5rt_1101 phase 1: does rt_systems/r0_predict_live_v1 forecast the same chord belief and plan the same note as
futures/v4_selfiter/live_predict.py PredictDuet (W_CUR 0, CHOICE 'mimic') from the same filter state? Only PredictDuet's
two pure methods (_forecast, _mimic) are called, on a stand-in object: no engine, no tracker, no audio (live_predict's
import chain imports sounddevice only inside its live functions). Val songs, the first N per dataset, L = 0.1.

  python check_r0_port_v1.py <bench file> [N]
"""
import importlib.util
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location('bench_under_test', sys.argv[1])
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
sys.path.insert(0, os.path.join(B.ABS, 'futures', 'v4_selfiter'))
import bench3  # noqa: E402  (the bench module's own bench3)
import live_predict as LP  # noqa: E402
from rt_systems import r0_predict_live_v1 as R0  # noqa: E402


class Stand:
    """The attributes PredictDuet._forecast / _mimic read."""


port = R0.build(bench3.train_sets())
plan0, hear0 = port.plan, port.hear
st = dict(plan_calls=0, planned=0, cmp=0, diff_h=0, max_db=0.0, kept=0, switched=0, notes=0)


def plan(q, heard):
    st['plan_calls'] += 1
    h_before = port.h
    out = plan0(q, heard)
    if out is not None:
        st['planned'] += 1
        s = Stand()
        s.system, s.W_CUR, s.cur, s.notes, s.h = port.f, 0.0, None, [list(heard[-1])], h_before
        b_live = LP.PredictDuet._forecast(s, int(q))
        h_live = LP.PredictDuet._mimic(s, b_live)
        b_port = port._forecast(int(q))
        st['cmp'] += 1
        st['max_db'] = max(st['max_db'], float(np.max(np.abs(b_live - b_port))))
        st['diff_h'] += int(h_live != out)
    return out


def hear(i, onset, midi, prev_dur):
    out = hear0(i, onset, midi, prev_dur)
    st['notes'] += 1
    st['kept' if out is None else 'switched'] += 1
    return out


port.plan, port.hear = plan, hear
for ds, songs in bench3.eval_sets('val').items():
    for s in songs[:N]:
        notes = [(float(on), float(off), int(m)) for on, off, m in sorted(bench3.B2.heard(s)['sop'], key=lambda n: n[0])]
        meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
        B.simulate(port, meta, notes, 0.1)
    print(ds, dict(st), flush=True)
ok = st['cmp'] > 0 and st['diff_h'] == 0 and st['max_db'] == 0.0
print('PORT MATCHES live_predict (forecast bit-identical, same planned note at every plan)' if ok else 'PORT DIFFERS')
sys.exit(0 if ok else 1)
