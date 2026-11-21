"""Mac-side codec of the Duet USB framed protocol, version 2.

Frame:  A5 5A | type u8 | seq u8 | len u16 LE | payload[len] | sum u8
        sum = (type + seq + len_lo + len_hi + sum(payload)) mod 256

This file mirrors firmware/libraries/DuetCore/src/duet_protocol.{h,cpp} and duet_notes.h byte for byte.
host/tests/test_host.py checks that by talking to the C++ core compiled for the Mac (board_sim).

Version 1 (duet_v0/docs/PLAN.md): audio, servo, LED, ping, status. Version 2 keeps those codes and lengths
and adds HELLO/INFO, CONFIG/CONFIG_ACK, NOTE_IN/NOTE_OUT, LOG and AUDIO_CTRL, and (2026-11-21)
AUDIO_QUERY/AUDIO_STATS, the device's audio counters.
"""
from __future__ import annotations

import enum
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence

SYNC = b"\xA5\x5A"
PROTO_VERSION = 2
HEADER_LEN = 6
MAX_TEXT = 240
CONFIG_HEADER_LEN = 8
MAX_POS = 1000
MAX_PAYLOAD = CONFIG_HEADER_LEN + MAX_POS
FRAME_SAMPLES = 240        # 10 ms at 24 kHz
FS = 24000


class T(enum.IntEnum):
    MIC_AUDIO = 0x01    # dev->host 240 x int16
    SPK_AUDIO = 0x02    # host->dev 240 x int16
    SERVO = 0x10        # host->dev angle u8, speed u8 (deg per 10 ms, 0 = immediate)
    LED = 0x11          # host->dev state u8 (0 off, 1 on, 2 blink)
    PING = 0x12         # host->dev u32 host us
    HELLO = 0x13        # host->dev empty; the device answers INFO
    CONFIG = 0x14       # host->dev plugin u8, tonic u8, M u8, flags u8, K u16, n_pos u16, pos[n_pos] u8
    NOTE_IN = 0x15      # host->dev i u16, midi u8, flags u8 (bit0: prev_dur valid), onset f32, prev_dur f32
    AUDIO_CTRL = 0x16   # host->dev flags u8 (bit0: send MIC_AUDIO, bit1: play SPK_AUDIO)
    AUDIO_QUERY = 0x17  # host->dev empty; the audio firmware answers AUDIO_STATS (same seq)
    STATUS = 0x20       # dev->host spk_buffer_frames u8, mic_overruns u16, spk_underruns u16
    DIAG = 0x21         # dev->host 8 x u16 (optional)
    PONG = 0x22         # dev->host u32 echoed host us, u32 device micros()
    INFO = 0x23         # dev->host ASCII "key=value;key=value"
    NOTE_OUT = 0x24     # dev->host i u16, midi i8 (-1 = silent), status u8, compute_us u32
    LOG = 0x25          # dev->host ASCII text
    CONFIG_ACK = 0x26   # dev->host status u8 (0 = ok), plugin u8, n_pos u16
    AUDIO_STATS = 0x27  # dev->host 48 bytes of audio counters (layout: AUDIO_STATS_FIELDS)


# (min, max) payload length per type; mirrors duet::len_range().
LEN_RANGE = {
    T.MIC_AUDIO: (480, 480), T.SPK_AUDIO: (480, 480), T.SERVO: (2, 2), T.LED: (1, 1), T.PING: (4, 4),
    T.STATUS: (5, 5), T.DIAG: (16, 16), T.PONG: (8, 8),
    T.HELLO: (0, 0), T.CONFIG: (CONFIG_HEADER_LEN, CONFIG_HEADER_LEN + MAX_POS), T.NOTE_IN: (12, 12),
    T.AUDIO_CTRL: (1, 1), T.INFO: (1, MAX_TEXT), T.NOTE_OUT: (8, 8), T.LOG: (1, MAX_TEXT),
    T.CONFIG_ACK: (4, 4), T.AUDIO_QUERY: (0, 0), T.AUDIO_STATS: (48, 48),
}
_RANGE_BY_CODE = {int(k): v for k, v in LEN_RANGE.items()}

ACK_OK, ACK_UNKNOWN_PLUGIN, ACK_BAD_CONFIG = 0, 1, 2
NOTE_OK, NOTE_NOT_CONFIGURED = 0, 1

# AUDIO_STATS (mirrors duet::AudioStats in duet_protocol.h). Counters are u32 since boot and wrap.
AUDIO_STATS_FORMAT = "<BBBBIIIIIIIIIHHI"
AUDIO_STATS_FIELDS = ("mode", "flags", "spk_level", "spk_state", "mic_sent", "mic_dropped", "spk_rx", "spk_played",
                      "spk_silent", "spk_underruns", "spk_dropped", "spk_seq_gaps", "spk_crc", "spk_peak",
                      "reserved", "dev_us")
AUDIO_MODE_NONE, AUDIO_MODE_I2S, AUDIO_MODE_STANDIN = 0, 1, 2
AUDIO_MODE_NAMES = {AUDIO_MODE_NONE: "none", AUDIO_MODE_I2S: "i2s", AUDIO_MODE_STANDIN: "standin"}
AS_MIC_ON, AS_SPK_ON, AS_RUNNING, AS_SPK_SUMS = 0x01, 0x02, 0x04, 0x08


def checksum(type_: int, seq: int, payload: bytes) -> int:
    n = len(payload)
    return (type_ + seq + (n & 0xFF) + (n >> 8) + sum(payload)) & 0xFF


def encode(type_: int, seq: int, payload: bytes = b"") -> bytes:
    payload = bytes(payload)
    rng = _RANGE_BY_CODE.get(int(type_) & 0xFF)
    if rng is None or not rng[0] <= len(payload) <= rng[1]:
        raise ValueError(f"type 0x{int(type_):02X}: payload of {len(payload)} bytes is not allowed")
    return (SYNC + struct.pack("<BBH", type_ & 0xFF, seq & 0xFF, len(payload)) + payload
            + bytes([checksum(type_ & 0xFF, seq & 0xFF, payload)]))


@dataclass(frozen=True)
class Frame:
    type: int
    seq: int
    payload: bytes

    @property
    def name(self) -> str:
        try:
            return T(self.type).name
        except ValueError:
            return f"0x{self.type:02X}"


@dataclass
class ParserStats:
    frames_ok: int = 0
    bad_checksum: int = 0
    bad_header: int = 0
    bytes_skipped: int = 0
    stale_drops: int = 0


class Parser:
    """Streaming decoder with resynchronisation, the same policy as the C++ parser: find A5 5A; reject an
    unknown type or a length outside the type's range; verify the checksum; on any failure discard ONE byte
    and rescan. Feed arbitrary chunks."""

    def __init__(self):
        self.buf = bytearray()
        self.stats = ParserStats()

    def reset(self) -> None:
        self.buf.clear()

    @property
    def pending(self) -> int:
        return len(self.buf)

    def drop_stale(self) -> List[Frame]:
        """Give up on the first pending byte (a header that promised bytes which never came) and rescan."""
        if not self.buf:
            return []
        del self.buf[:1]
        self.stats.bytes_skipped += 1
        self.stats.stale_drops += 1
        return self.feed(b"")

    def feed(self, data: bytes) -> List[Frame]:
        self.buf += data
        out: List[Frame] = []
        buf = self.buf
        pos = 0
        n = len(buf)
        st = self.stats
        while True:
            i = buf.find(SYNC, pos)
            if i < 0:
                # keep a trailing A5 (it could be the first half of a sync)
                keep = 1 if n > pos and buf[-1] == 0xA5 else 0
                st.bytes_skipped += (n - pos) - keep
                pos = n - keep
                break
            st.bytes_skipped += i - pos
            pos = i
            if pos + HEADER_LEN > n:
                break
            type_, seq, ln = buf[pos + 2], buf[pos + 3], buf[pos + 4] | (buf[pos + 5] << 8)
            rng = _RANGE_BY_CODE.get(type_)
            if rng is None or not rng[0] <= ln <= rng[1]:
                st.bad_header += 1
                st.bytes_skipped += 1
                pos += 1
                continue
            flen = HEADER_LEN + ln + 1
            if pos + flen > n:
                break
            payload = bytes(buf[pos + HEADER_LEN: pos + HEADER_LEN + ln])
            if checksum(type_, seq, payload) != buf[pos + HEADER_LEN + ln]:
                st.bad_checksum += 1
                st.bytes_skipped += 1
                pos += 1
                continue
            st.frames_ok += 1
            out.append(Frame(type_, seq, payload))
            pos += flen
        del buf[:pos]
        return out


# ------------------------------------------------------------------ builders (host -> device)

def ping(seq: int, host_us: int) -> bytes:
    return encode(T.PING, seq, struct.pack("<I", host_us & 0xFFFFFFFF))


def hello(seq: int) -> bytes:
    return encode(T.HELLO, seq)


def led(seq: int, state: int) -> bytes:
    return encode(T.LED, seq, bytes([int(state) & 0xFF]))


def servo(seq: int, angle_deg: int, speed_deg_per_10ms: int = 0) -> bytes:
    return encode(T.SERVO, seq, bytes([max(0, min(180, int(angle_deg))), max(0, min(255, int(speed_deg_per_10ms)))]))


def audio_ctrl(seq: int, mic: bool, spk: bool) -> bytes:
    return encode(T.AUDIO_CTRL, seq, bytes([(1 if mic else 0) | (2 if spk else 0)]))


def config(seq: int, plugin: int, tonic: int, M: int, K: int = 0, pos: Sequence[int] = ()) -> bytes:
    """pos longer than MAX_POS is cut: the device cannot hold more (and says how many it took in the ACK)."""
    pos = bytes(int(p) & 0xFF for p in pos[:MAX_POS])
    head = struct.pack("<BBBBHH", plugin & 0xFF, int(tonic) % 256, M & 0xFF, 0, K & 0xFFFF, len(pos))
    return encode(T.CONFIG, seq, head + pos)


def note_in(seq: int, i: int, midi: int, onset: float, prev_dur: Optional[float]) -> bytes:
    flags = 0 if prev_dur is None else 1
    return encode(T.NOTE_IN, seq, struct.pack("<HBBff", i & 0xFFFF, midi & 0xFF, flags, float(onset),
                                              0.0 if prev_dur is None else float(prev_dur)))


def spk_audio(seq: int, pcm_bytes: bytes) -> bytes:
    """pcm_bytes: 240 samples, int16 little-endian (480 bytes)."""
    return encode(T.SPK_AUDIO, seq, pcm_bytes)


def audio_query(seq: int) -> bytes:
    return encode(T.AUDIO_QUERY, seq)


def decode_audio_stats(p: bytes) -> dict:
    d = dict(zip(AUDIO_STATS_FIELDS, struct.unpack(AUDIO_STATS_FORMAT, p)))
    d.pop("reserved")
    f = d["flags"]
    d["mode_name"] = AUDIO_MODE_NAMES.get(d["mode"], str(d["mode"]))
    d["mic_on"], d["spk_on"] = bool(f & AS_MIC_ON), bool(f & AS_SPK_ON)
    d["running"], d["spk_sums"] = bool(f & AS_RUNNING), bool(f & AS_SPK_SUMS)
    return d


# ------------------------------------------------------------------ decoding (device -> host)

def decode_payload(fr: Frame):
    t = fr.type
    p = fr.payload
    if t == T.PONG:
        h, d = struct.unpack("<II", p)
        return {"host_us": h, "dev_us": d}
    if t == T.INFO:
        text = p.decode("ascii", "replace")
        out = {}
        for part in text.split(";"):
            k, _, v = part.partition("=")
            if k:
                out[k] = v
        return out
    if t == T.LOG:
        return p.decode("ascii", "replace")
    if t == T.NOTE_OUT:
        i, midi, status, us = struct.unpack("<HbBI", p)
        return {"i": i, "midi": midi, "status": status, "compute_us": us}
    if t == T.CONFIG_ACK:
        status, plugin, n_pos = struct.unpack("<BBH", p)
        return {"status": status, "plugin": plugin, "n_pos": n_pos}
    if t == T.STATUS:
        b, mo, su = struct.unpack("<BHH", p)
        return {"spk_buffer_frames": b, "mic_overruns": mo, "spk_underruns": su}
    if t == T.DIAG:
        keys = ("bad_checksum", "bad_header", "spk_overflows", "spk_seq_gaps", "spk_trims",
                "mic_dma_overflows", "mic_queue_usb_drops", "audio_init_err")
        return dict(zip(keys, struct.unpack("<8H", p)))
    if t == T.AUDIO_STATS:
        return decode_audio_stats(p)
    return p


def f32(x: float) -> float:
    """The value the device sees after x has crossed the wire as a 32-bit float."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]
