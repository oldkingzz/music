"""The two ends of the note path on the Mac: sound -> notes ("listen") and notes -> sound ("sing").

Both are the code the first Duet repository already has (vendor/dsp.py, vendor/follow.py), used unchanged:
  listen: YIN pitch every 10 ms, then offline_notes splits the recording into notes (robust to vibrato)
  sing:   render_voice, the soft machine voice
Everything here works on finished recordings (clips). The live, frame-by-frame versions are the next step
and will plug into the same places.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

_VENDOR = str(Path(__file__).resolve().parent / "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
import dsp      # noqa: E402  (vendor/dsp.py)
import follow   # noqa: E402  (vendor/follow.py)

SR = dsp.SR     # 24 kHz, the project's audio rate


def clip_to_notes(x: np.ndarray) -> List[dict]:
    """x: mono float audio at 24 kHz. Returns [{'on': s, 'off': s, 'midi': int}] in time order."""
    F = follow.analyse(np.asarray(x, dtype=float))
    notes = follow.offline_notes(F)
    # The tracker reports times about one analysis window early, so a note at the very start can come out a
    # few milliseconds before 0: clamp it.
    out = [{"on": max(0.0, float(n["on"])), "off": float(n["off"]), "midi": int(n["midi"])} for n in notes
           if n["off"] > max(0.0, n["on"]) and 0 <= int(n["midi"]) < 128]
    out.sort(key=lambda n: n["on"])
    return out


def estimate_tonic(notes: List[dict]) -> int:
    """Home note of the key as a pitch class 0..11 (C = 0), from how long each pitch class is sung.
    The harmony models know major keys only, so a minor key is reported as its relative major
    (same scale notes): A minor -> C."""
    tonic, mode, _ = follow.detect_key(notes)
    return int(tonic if mode == "major" else (tonic + 3) % 12)


def notes_to_voice(notes: List[tuple], n_samples: int, level: float = 0.22) -> np.ndarray:
    """notes: [(start_s, dur_s, midi)]. Returns n_samples of the machine voice at 24 kHz."""
    return dsp.render_voice([(float(s), float(d), float(m)) for s, d, m in notes], int(n_samples), level=level)


def melody_clip(notes: List[tuple], tail_s: float = 0.3, level: float = 0.3) -> np.ndarray:
    """A test recording: the machine voice singing notes [(start_s, dur_s, midi)]. Used by the tests in place
    of a person, because its notes are known exactly."""
    end = max(s + d for s, d, _ in notes) + tail_s
    return notes_to_voice(notes, int(end * SR), level=level).astype(np.float32)


PC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def midi_name(m: int) -> str:
    return "-" if m < 0 else f"{PC_NAMES[m % 12]}{m // 12 - 1}"
