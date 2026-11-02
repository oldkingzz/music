"""m2_memory_v4 (final) = m2_memory_v3 with tuned defaults: metric-aware chord filter + fixed-lag
hindsight + song memory (weights 0.5/0.35/0.2 for 4/3/2-note context matches), online transition
adaptation off (it had no measurable effect), vl 0.01, doubling penalty 0.12. Method: see m2_memory_v3.
DUET_HIST limits both the filter (re-run from the first note inside the window) and the memory
(only entries stored from notes inside the window are used)."""
import json
import os
os.environ.setdefault('DUET_M2', json.dumps({'vl': 0.01, 'dbl': 0.12, 'wmem': [0.5, 0.35, 0.2], 'use_adapt': 0}))
from systems.m2_memory_v3 import build, P  # noqa: E402,F401
