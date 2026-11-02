"""m2_memory_v2 = m2_memory_v1 (filter + song memory + online transition adaptation) with retuned
voice-leading weights (vl 0.01 per semitone, doubling penalty 0.12). See m2_memory_v1 for the method."""
import json
import os
os.environ.setdefault('DUET_M2', json.dumps({'vl': 0.01, 'dbl': 0.12}))
from systems.m2_memory_v1 import build, P  # noqa: E402,F401
