"""m3_seq_v5 with history limited to the last 4 beats (DUET_HIST=4), for the history ablation."""
import os
os.environ['DUET_HIST'] = '4'
from systems.m3_seq_v5 import build  # noqa: E402,F401
