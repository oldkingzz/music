"""m3_seq_s1: second 96-unit GRU (seed 1, 200 s training), v3 voice-leading; mainly a member for ensembling."""
import os
os.environ.setdefault('M3_TAG', 's1')
os.environ.setdefault('M3_SEED', '1')
os.environ.setdefault('M3_SECS', '200')
for _k, _v in dict(DBL='0.2', STEP='0.015', BIG='0.1', PAR='0.3', ABOVE='0.0').items():
    os.environ.setdefault('M3_' + _k, _v)
from systems.m3_seq_v1 import build  # noqa: E402,F401
