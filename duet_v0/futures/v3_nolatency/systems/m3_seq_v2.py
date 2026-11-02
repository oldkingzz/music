"""m3_seq_v2: same method as m3_seq_v1 but a wider GRU (128 units) trained longer (270 s)."""
import os
os.environ.setdefault('M3_TAG', 'big')
os.environ.setdefault('M3_HID', '128')
os.environ.setdefault('M3_SECS', '270')
from systems.m3_seq_v1 import build  # noqa: E402,F401
