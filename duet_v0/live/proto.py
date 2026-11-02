"""Duet USB serial protocol, host side.

The wire format has ONE Python implementation: firmware/tools/proto.py (it mirrors the
firmware's C++ parser byte for byte and is cross-checked against it in firmware/tools/test_proto.py).
This module loads it under a private name (both files are called proto.py) and adds what the
live app needs on top:

* the old live/ names (MIC_AUDIO, encode, Parser.feed -> (type, seq, payload) tuples, stats()),
  so the rest of live/ did not have to change;
* float <-> int16 audio payload helpers (scale 32768, the same as soundfile);
* SeqCounters / SeqTracker: sequence numbers are PER MESSAGE TYPE in both directions
  (firmware/README.md section 1.5): MIC_AUDIO seq is assigned at capture (gap = lost mic frame),
  STATUS has its own counter, PONG carries the seq of the PING it answers, and the device counts
  gaps in the host's SPK_AUDIO seq. So the host keeps one outgoing counter per type and checks
  incoming gaps per type.

Frame: 0xA5 0x5A | type u8 | seq u8 | len u16 (LE) | payload[len] | sum u8
sum = (type + seq + len_lo + len_hi + sum(payload)) mod 256.
"""
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

_FW = Path(__file__).resolve().parent.parent / 'firmware' / 'tools' / 'proto.py'
if 'duet_fw_proto' in sys.modules:
    fw = sys.modules['duet_fw_proto']
else:
    if not _FW.exists():
        raise ImportError(f'firmware protocol codec not found at {_FW} (live/ uses it as the single source of truth)')
    _spec = importlib.util.spec_from_file_location('duet_fw_proto', _FW)
    fw = importlib.util.module_from_spec(_spec)
    sys.modules['duet_fw_proto'] = fw          # dataclasses need the module registered
    _spec.loader.exec_module(fw)

SYNC = fw.SYNC
HEADER = fw.HEADER_LEN
T = fw.T
EXPECTED_LEN = fw.EXPECTED_LEN
checksum = fw.checksum
encode = fw.encode
seq_gap = fw.seq_gap
u16_delta = fw.u16_delta

MIC_AUDIO = int(T.MIC_AUDIO)
SPK_AUDIO = int(T.SPK_AUDIO)
SERVO = int(T.SERVO)
LED = int(T.LED)
PING = int(T.PING)
STATUS = int(T.STATUS)
DIAG = int(T.DIAG)
PONG = int(T.PONG)

NAMES = {int(t): t.name for t in T}
LED_OFF, LED_ON, LED_BLINK = 0, 1, 2


# ---------------------------------------------------------------- payload helpers
def audio_payload(x):
    """float32 [-1, 1] (or int16) block -> little-endian int16 bytes (scale 32768, like soundfile)."""
    x = np.asarray(x)
    if x.dtype != np.int16:
        x = np.round(np.clip(x, -1.0, 32767 / 32768) * 32768.0).astype(np.int16)
    return x.astype('<i2').tobytes()


def audio_from_payload(p):
    return np.frombuffer(p, '<i2').astype(np.float32) / 32768.0


def servo_payload(angle, speed=0):
    return struct.pack('<BB', int(np.clip(round(angle), 0, 180)), int(np.clip(round(speed), 0, 255)))


def led_payload(state):
    return struct.pack('<B', int(state))


def ping_payload(host_us):
    return struct.pack('<I', int(host_us) & 0xFFFFFFFF)


def status_payload(spk_buffer_frames, mic_overruns, spk_underruns):
    return struct.pack('<BHH', spk_buffer_frames & 0xFF, mic_overruns & 0xFFFF, spk_underruns & 0xFFFF)


def parse_status(p):
    b, o, u = struct.unpack('<BHH', p[:5])
    return {'spk_buffer_frames': b, 'mic_overruns': o, 'spk_underruns': u}


def pong_payload(host_us, dev_us):
    return struct.pack('<II', host_us & 0xFFFFFFFF, dev_us & 0xFFFFFFFF)


def parse_pong(p):
    h, d = struct.unpack('<II', p[:8])
    return {'host_us': h, 'dev_us': d}


def parse_diag(p):
    return fw.decode_payload(fw.Frame(DIAG, 0, bytes(p)))


# ---------------------------------------------------------------- streaming parser
class Parser:
    """Incremental decoder (the firmware's policy: unknown type or wrong length for the type ->
    discard one byte and rescan). feed(bytes) -> list of (type, seq, payload)."""

    def __init__(self):
        self._p = fw.Parser()

    def feed(self, data):
        return [(f.type, f.seq, f.payload) for f in self._p.feed(data)]

    def stats(self):
        s = self._p.stats
        return {'frames': s.frames_ok, 'bad_sum': s.bad_checksum, 'bad_len': s.bad_header, 'skipped': s.bytes_skipped}


# ---------------------------------------------------------------- sequence numbers (per type)
class SeqCounters:
    """Outgoing: one 8-bit counter per frame type (the device checks SPK_AUDIO continuity)."""

    def __init__(self):
        self.c = {}

    def next(self, ftype):
        s = self.c.get(ftype, 0)
        self.c[ftype] = (s + 1) & 0xFF
        return s


class SeqTracker:
    """Incoming: gaps counted per type. For MIC_AUDIO a gap of n = n mic frames lost on the device
    or the link (the seq is assigned at capture). PONG is excluded: its seq echoes the PING's."""

    def __init__(self, ignore=(PONG,)):
        self.last = {}
        self.gaps = {}          # type -> number of gap events
        self.lost = {}          # type -> frames missing in total
        self.ignore = set(ignore)

    def see(self, ftype, seq):
        if ftype in self.ignore:
            return 0
        g = seq_gap(self.last.get(ftype), seq)
        self.last[ftype] = seq
        if g:
            self.gaps[ftype] = self.gaps.get(ftype, 0) + 1
            self.lost[ftype] = self.lost.get(ftype, 0) + g
        return g

    def summary(self):
        return {NAMES.get(t, hex(t)): {'gaps': self.gaps.get(t, 0), 'lost': self.lost.get(t, 0)}
                for t in sorted(set(self.last) | set(self.gaps))}
