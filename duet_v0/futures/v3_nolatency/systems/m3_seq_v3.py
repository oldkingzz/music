"""m3_seq_v3: m3_seq_v1's GRU (96 units) with voice-leading weights tuned on val (doubling penalty 0.2,
step cost 0.015/semitone, big-leap 0.1, parallel 0.3, no penalty for going above her). DUET_HIST respected."""
import os
for _k, _v in dict(DBL='0.2', STEP='0.015', BIG='0.1', PAR='0.3', ABOVE='0.0').items():
    os.environ.setdefault('M3_' + _k, _v)
from systems.m3_seq_v1 import build  # noqa: E402,F401
