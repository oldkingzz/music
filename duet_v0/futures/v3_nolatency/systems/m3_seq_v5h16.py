"""m3_seq_v5 with history limited to the last 16 beats (DUET_HIST=16), for the history ablation."""
import os
os.environ['DUET_HIST'] = '16'
from systems.m3_seq_v5 import build  # noqa: E402,F401
