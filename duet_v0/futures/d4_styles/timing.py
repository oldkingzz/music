"""Compute time of the D4 inference path (numpy only): chord model per song and the 3 style arrangers.
Run plainly (M4) and on an efficiency core (board estimate = 2 x E-core time, per BRIEF):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 taskpolicy -c background .venv/bin/python futures/d4_styles/timing.py ecore
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import styles  # noqa: E402
from chord_board import NpBiGRU, StyleHMM  # noqa: E402
from feats import features, load_all  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else 'm4'
data = load_all()
g = NpBiGRU(os.path.join(HERE, 'models', 'bigru.npz'))
songs = data['pop909']['val'][:20]
h = StyleHMM(2.0, 0.5).fit(data['pop909']['train'][:50])
beats = sum(s['K'] for s in songs)
t0 = time.perf_counter()
for s in songs:
    L = g.logp(features(s, 'pop909'))
    h.decode(s, 0, 'vit', extra_ll=L + 0.3 * h.ll(s, 1.0), beta=0.25)     # the 'ens' path (superset of bigru_vit)
dt = time.perf_counter() - t0
mels = styles.melodies()
t1 = time.perf_counter()
for m in mels:
    for fn in (styles.style_bach, styles.style_pop, styles.style_jazz):
        fn(m)
dt2 = time.perf_counter() - t1
secs = sum(m.notes[-1][1] - m.notes[0][0] for m in mels)
out = {'tag': tag, 'chord_model_us_per_beat': 1e6 * dt / beats, 'chord_model_ms_per_255beat_song': 1e3 * dt / beats * 255,
       'arrange_3_styles_s_total': dt2, 'melody_seconds_total': secs, 'arrange_s_per_melody_3_styles': dt2 / len(mels)}
p = os.path.join(HERE, 'out', 'timing.json')
allr = json.load(open(p)) if os.path.exists(p) else {}
allr[tag] = out
json.dump(allr, open(p, 'w'), indent=1)
print(json.dumps(out))
