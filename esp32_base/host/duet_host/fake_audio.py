"""Python mirror of the stand-in microphone and speaker in firmware/libraries/DuetCore/src/duet_standin.{h,cpp}.

The board has no microphone or speaker wired yet. Firmware built with DUET_AUDIO_STANDIN=1 (and the Mac simulator)
makes up its microphone signal with integer arithmetic only, so this file produces the very same int16 numbers:
the Mac can check every sample the board sends.

The melody: "Twinkle, Twinkle" in C major, 14 notes on a 0.5 s grid (the 7th and the 14th last 1.0 s), looped
every 8.0 s (192000 samples at 24 kHz). Each note sounds for its slot minus a 50 ms silent gap, with a 5 ms
linear attack and a 10 ms linear release. The tone is a parabolic sine (no sin(), no floats) of the fundamental
plus the 2nd and 3rd harmonic, weights 6 : 3 : 2, scaled to a peak of about 12000.

    sample(n)               one sample, scalar integer code (the definition, line for line as in C++)
    samples(n0, count)      the same with numpy (fast), checked against sample() by the tests
    crc32(x, prev)          the stand-in speaker's running checksum: zlib CRC-32 of the int16 little-endian bytes

Rules that keep it bit-exact with C++: no division or modulo of a negative number (C truncates, Python floors),
products kept below 2^63, phases wrapped to 32 bits explicitly.
"""
from __future__ import annotations

import zlib

import numpy as np

RATE = 24000
FRAME = 240                      # samples per 10 ms frame

# (midi, phase increment per sample = round(f * 2^32 / 24000), slot length in samples). Same table in C++.
INC = {60: 46819719, 62: 52553357, 64: 58989149, 65: 62496826, 67: 70150316, 69: 78741067}
SHORT, LONG = 12000, 24000       # 0.5 s and 1.0 s slots
MELODY = [(60, SHORT), (60, SHORT), (67, SHORT), (67, SHORT), (69, SHORT), (69, SHORT), (67, LONG),
          (65, SHORT), (65, SHORT), (64, SHORT), (64, SHORT), (62, SHORT), (62, SHORT), (60, LONG)]
GAP = 1200                       # 50 ms of silence at the end of every slot
ATTACK = 120                     # 5 ms
RELEASE = 240                    # 10 ms
W1, W2, W3 = 6, 3, 2             # harmonic weights
GAIN_MUL, GAIN_SHIFT = 5333, 31  # out = mix * env * GAIN_MUL >> GAIN_SHIFT  (peak about 12000)

STARTS = []
_t = 0
for _m, _len in MELODY:
    STARTS.append(_t)
    _t += _len
LOOP = _t                        # 192000 samples = 8.0 s


def psin(phase: int) -> int:
    """Parabolic sine of a 32-bit phase: -16384..16384."""
    p = (phase >> 16) & 0xFFFF
    x = p & 0x7FFF
    y = (x * (32768 - x)) >> 14
    return -y if p & 0x8000 else y


def sample(n: int) -> int:
    pos = n % LOOP               # n >= 0
    k = 0
    while k + 1 < len(MELODY) and STARTS[k + 1] <= pos:
        k += 1
    midi, slot = MELODY[k]
    t = pos - STARTS[k]
    sound = slot - GAP
    if t >= sound:
        return 0
    if t < ATTACK:
        env = (t * 32768) // ATTACK
    elif t >= sound - RELEASE:
        env = ((sound - t) * 32768) // RELEASE
    else:
        env = 32768
    ph = (t * INC[midi]) & 0xFFFFFFFF
    mix = W1 * psin(ph) + W2 * psin((ph * 2) & 0xFFFFFFFF) + W3 * psin((ph * 3) & 0xFFFFFFFF)
    v = abs(mix) * env * GAIN_MUL
    out = v >> GAIN_SHIFT
    return -out if mix < 0 else out


def note_table():
    """[(start_sample, sound_samples, midi)] of one loop."""
    return [(s, ln - GAP, m) for s, (m, ln) in zip(STARTS, MELODY)]


def _psin_np(ph: np.ndarray) -> np.ndarray:
    p = (ph >> 16) & 0xFFFF
    x = p & 0x7FFF
    y = (x * (32768 - x)) >> 14
    return np.where(p & 0x8000, -y, y)


def samples(n0: int, count: int) -> np.ndarray:
    """int16 samples n0 .. n0+count-1 of the stand-in microphone."""
    n = np.arange(int(n0), int(n0) + int(count), dtype=np.int64)
    pos = n % LOOP
    starts = np.array(STARTS, dtype=np.int64)
    k = np.searchsorted(starts, pos, side="right") - 1
    slot = np.array([ln for _, ln in MELODY], dtype=np.int64)[k]
    inc = np.array([INC[m] for m, _ in MELODY], dtype=np.int64)[k]
    t = pos - starts[k]
    sound = slot - GAP
    env = np.full(len(n), 32768, dtype=np.int64)
    att = t < ATTACK
    env[att] = (t[att] * 32768) // ATTACK
    rel = (t >= sound - RELEASE) & ~att
    env[rel] = ((sound[rel] - t[rel]) * 32768) // RELEASE
    ph = (t * inc) & 0xFFFFFFFF          # t < 2^15, inc < 2^27: the product fits in int64
    mix = W1 * _psin_np(ph) + W2 * _psin_np((ph * 2) & 0xFFFFFFFF) + W3 * _psin_np((ph * 3) & 0xFFFFFFFF)
    v = (np.abs(mix) * np.where(t < sound, env, 0) * GAIN_MUL) >> GAIN_SHIFT
    return np.where(mix < 0, -v, v).astype(np.int16)


def frame(k: int) -> np.ndarray:
    """Mic frame k after the gate opened (the generator restarts at sample 0 every time the mic is turned on)."""
    return samples(k * FRAME, FRAME)


def crc32(x, prev: int = 0) -> int:
    """Continue the stand-in speaker's checksum over int16 samples x (little-endian bytes), from value prev."""
    return zlib.crc32(np.asarray(x, dtype="<i2").tobytes(), prev) & 0xFFFFFFFF


def peak(x) -> int:
    a = np.asarray(x, dtype=np.int32)
    return int(np.max(np.abs(a))) if len(a) else 0
