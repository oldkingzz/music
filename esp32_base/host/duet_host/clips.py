"""The clip path's storage: every recording is one WAV file plus one JSON file in esp32_base/runs/clips/.

A clip is a finished piece of sound (a phrase, a whole song, the result of a block). Clips are stored as
24 kHz mono 16-bit, the project's audio format, whatever format they arrived in. Nothing is ever
overwritten or deleted here: every clip gets a new id.
"""
from __future__ import annotations

import io
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SR = 24000
ROOT = Path(__file__).resolve().parents[2]
MIN_FREE_BYTES = 500 * 1024 * 1024      # refuse to store new clips when the disk has less than this free
MAX_SECONDS = 15 * 60                   # longest clip accepted


class ClipError(RuntimeError):
    pass


def to_mono_24k(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x.astype(np.float32)


class ClipStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else ROOT / "runs" / "clips"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------ adding
    def add_array(self, x: np.ndarray, sr: int = SR, *, name: str, source: str, parent: Optional[str] = None,
                  made_by: Optional[str] = None, params: Optional[dict] = None, extra: Optional[dict] = None) -> dict:
        import soundfile as sf
        x = to_mono_24k(x, sr)
        if len(x) == 0:
            raise ClipError("the recording is empty")
        if len(x) > MAX_SECONDS * SR:
            raise ClipError(f"the recording is longer than {MAX_SECONDS // 60} minutes")
        free = shutil.disk_usage(self.root).free
        if free < MIN_FREE_BYTES:
            raise ClipError(f"only {free / 1e9:.2f} GB free on the disk; clips are not stored below "
                            f"{MIN_FREE_BYTES / 1e9:.1f} GB")
        peak = float(np.max(np.abs(x)))
        if peak > 1.0:
            x = x / peak                                   # never clip on the way to 16 bit
        with self._lock:
            while True:
                cid = time.strftime("c%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
                wav = self.root / f"{cid}.wav"
                if not wav.exists():
                    break
            sf.write(str(wav), x, SR, subtype="PCM_16")
            meta = {"id": cid, "name": name, "source": source, "parent": parent, "made_by": made_by,
                    "params": params or {}, "seconds": round(len(x) / SR, 3), "peak": round(min(peak, 1.0), 4),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "bytes": wav.stat().st_size}
            if extra:
                meta.update(extra)
            (self.root / f"{cid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        return meta

    def add_bytes(self, data: bytes, name: str, source: str = "upload") -> dict:
        """data: the bytes of an audio file soundfile can read (WAV, FLAC, OGG, MP3, AIFF)."""
        import soundfile as sf
        try:
            x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        except Exception as e:
            raise ClipError(f"cannot read {name!r} as audio ({type(e).__name__}). WAV, FLAC, OGG, MP3 and AIFF work.")
        return self.add_array(x, sr, name=name, source=source)

    def add_file(self, path, name: Optional[str] = None, source: str = "file") -> dict:
        p = Path(path)
        return self.add_bytes(p.read_bytes(), name or p.name, source)

    # ------------------------------------------------------------ reading
    def meta(self, cid: str) -> dict:
        p = self.root / f"{cid}.json"
        if not _safe_id(cid) or not p.exists():
            raise ClipError(f"no clip {cid!r}")
        return json.loads(p.read_text())

    def wav_path(self, cid: str) -> Path:
        p = self.root / f"{cid}.wav"
        if not _safe_id(cid) or not p.exists():
            raise ClipError(f"no clip {cid!r}")
        return p

    def load(self, cid: str) -> Tuple[np.ndarray, dict]:
        import soundfile as sf
        x, sr = sf.read(str(self.wav_path(cid)), dtype="float32")
        assert sr == SR
        return x, self.meta(cid)

    def list(self) -> List[dict]:
        out = []
        for p in self.root.glob("c*.json"):
            try:
                out.append(json.loads(p.read_text()))
            except Exception:
                pass
        out.sort(key=lambda m: m["id"], reverse=True)
        return out


def _safe_id(cid: str) -> bool:
    return cid.replace("_", "").isalnum() and len(cid) < 40
