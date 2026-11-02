"""m3_seq_v3 with history limited to the last 0 beats (DUET_HIST=0), for the history ablation."""
import os
os.environ['DUET_HIST'] = '0'
from systems.m3_seq_v3 import build  # noqa: E402,F401
