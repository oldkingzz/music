"""Audio over the link: record the board's microphone, play to the board's speaker, or both at once.

    from duet_host.audio_stream import record_from_board, play_to_board, record_and_play
    rec = record_from_board(link, 5.0)       # rec.samples (int16), rec.frames, rec.seq_gaps, rec.missing ...
    res = play_to_board(link, samples)       # res["frames_played"], res["crc_match"], ... (the board's counters)
    out = record_and_play(link, samples, 5.0, note_every=0.1)   # both, plus a NOTE_IN every 100 ms

Format on the wire: 24 kHz mono int16, 240 samples (10 ms) per MIC_AUDIO / SPK_AUDIO frame. The board's
gates are opened with AUDIO_CTRL and closed again at the end; nothing here touches the Mac's own sound devices.

Playing: one silent frame is put in front of the samples (the jitter buffer fades in the first frame after it
primes, and fading zeros changes nothing, so the board's checksum is exactly of what was sent), LEAD_FRAMES frames
go at once, the rest on an absolute 10 ms schedule of the Mac clock. The board's counters are
read before, right after the last frame went (underruns so far), and once every frame has been played.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from . import fake_audio, proto
from .link import Link, LinkError
from .proto import T

FRAME = proto.FRAME_SAMPLES       # 240
RATE = proto.FS                   # 24000
# Frames sent at once (the silent one included) before the 10 ms pace starts. The board starts playing at 4 queued
# (SPK_PRIME_FRAMES) and drops a frame if more than 6 (SPK_TRIM_ABOVE) stay queued for 0.5 s, so 5 keeps one
# frame of margin on both sides: the level seen right after a send is 5, after a pop 4.
LEAD_FRAMES = 5


@dataclass
class Recording:
    samples: np.ndarray           # int16; a missing frame is filled with zeros
    frames: int                   # MIC_AUDIO frames received
    seconds: float                # wall time the mic gate was open
    first_seq: Optional[int]      # seq of the first frame (the stand-in starts every recording at 0)
    seq_gaps: int                 # places where seq did not go up by exactly 1
    missing: int                  # frames lost inside the recording (from the seq numbers)
    frame_index: List[int] = field(default_factory=list)   # frame number of each received frame (0 = first)
    arrival: List[float] = field(default_factory=list)     # perf_counter() at arrival

    @property
    def frames_per_second(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def to_int16(x) -> np.ndarray:
    """float audio in -1..1 (or int16 already) -> int16."""
    a = np.asarray(x)
    if a.dtype == np.int16:
        return a
    if np.issubdtype(a.dtype, np.integer):
        return np.clip(a, -32768, 32767).astype(np.int16)
    return np.clip(np.round(a.astype(np.float64) * 32767.0), -32768, 32767).astype(np.int16)


class BoardAudio:
    """One user of the board's audio on one link. Keeps the two AUDIO_CTRL gates, so recording and playing at
    the same time never closes the other's gate."""

    def __init__(self, link: Link):
        self.link = link
        self._glock = threading.Lock()
        self._mic = False
        self._spk = False

    def _gates(self, mic: Optional[bool] = None, spk: Optional[bool] = None) -> None:
        with self._glock:
            if mic is not None:
                self._mic = mic
            if spk is not None:
                self._spk = spk
            self.link.audio_ctrl(self._mic, self._spk)

    def stats(self) -> dict:
        return self.link.audio_stats()

    # ------------------------------------------------------------ microphone
    def record(self, seconds: float, settle: float = 0.25) -> Recording:
        got = []                                   # (seq, payload, t)
        lock = threading.Lock()
        prev_cb = self.link.on_frame

        def hook(fr: proto.Frame) -> None:
            if fr.type == T.MIC_AUDIO:
                with lock:
                    got.append((fr.seq, fr.payload, time.perf_counter()))
            if prev_cb is not None:
                prev_cb(fr)

        self.link.on_frame = hook
        try:
            self._gates(mic=True)
            t0 = time.perf_counter()
            time.sleep(max(0.0, float(seconds)))
            self._gates(mic=False)
            t1 = time.perf_counter()
            time.sleep(settle)                     # frames already on their way
        finally:
            self.link.on_frame = prev_cb
        with lock:
            got = list(got)
        return _assemble(got, t1 - t0)

    # ------------------------------------------------------------ speaker
    def play(self, samples, lead_frames: int = LEAD_FRAMES, drain_timeout: float = 2.0) -> dict:
        if lead_frames < 1:
            raise ValueError("lead_frames must be at least 1")
        x = to_int16(samples)
        n = (len(x) + FRAME - 1) // FRAME
        pcm = np.zeros((n + 1) * FRAME, dtype=np.int16)        # one silent frame first
        pcm[FRAME:FRAME + len(x)] = x
        frames = [pcm[k * FRAME:(k + 1) * FRAME].astype("<i2").tobytes() for k in range(n + 1)]

        before = self.stats()
        self._gates(spk=True)
        late = []
        t0 = time.perf_counter()
        try:
            for k, fb in enumerate(frames):
                due = t0 + max(0, k - (lead_frames - 1)) * 0.010
                now = time.perf_counter()
                if due > now:
                    time.sleep(due - now)
                late.append(time.perf_counter() - due)
                self.link.send(proto.spk_audio(k & 0xFF, fb))
            at_last = self.stats()
            end = time.perf_counter() + drain_timeout + 0.01 * lead_frames
            after = at_last
            while after["spk_played"] - before["spk_played"] < len(frames) and time.perf_counter() < end:
                time.sleep(0.02)
                after = self.stats()
            time.sleep(0.05)                         # let the end-of-stream underrun show
            after = self.stats()
        finally:
            self._gates(spk=False)
        want_crc = fake_audio.crc32(pcm, before["spk_crc"])
        d = lambda key, a=after: (a[key] - before[key]) & 0xFFFFFFFF  # noqa: E731
        return {
            "mode": after["mode_name"],
            "frames_sent": len(frames),
            "frames_played": d("spk_played"),
            "frames_received": d("spk_rx"),
            "underruns_while_sending": (at_last["spk_underruns"] - before["spk_underruns"]) & 0xFFFFFFFF,
            "underruns_after_drain": d("spk_underruns"),   # 1 is normal: the buffer runs dry after the last frame
            "dropped": d("spk_dropped"),
            "seq_gaps": d("spk_seq_gaps"),
            "level_at_last_send": at_last["spk_level"],
            "crc_board": after["spk_crc"],
            "crc_mac": want_crc,
            "sums_kept": after["spk_sums"],
            "crc_match": bool(after["spk_sums"]) and after["spk_crc"] == want_crc,
            "peak_board": after["spk_peak"],
            "peak_sent": fake_audio.peak(pcm),
            "send_late_ms_max": round(1000 * max(late), 2) if late else 0.0,
            "seconds": round(time.perf_counter() - t0, 3),
            "before": before,
            "after": after,
        }


def _assemble(got, seconds: float) -> Recording:
    if not got:
        return Recording(np.zeros(0, dtype=np.int16), 0, seconds, None, 0, 0)
    idx = [0]
    gaps = 0
    for k in range(1, len(got)):
        step = (got[k][0] - got[k - 1][0]) & 0xFF
        if step != 1:
            gaps += 1
        idx.append(idx[-1] + (step if step else 256))      # a repeated seq is taken as a full wrap
    total = idx[-1] + 1
    out = np.zeros(total * FRAME, dtype=np.int16)
    for (seq, payload, _), i in zip(got, idx):
        out[i * FRAME:(i + 1) * FRAME] = np.frombuffer(payload, dtype="<i2")
    return Recording(out, len(got), seconds, got[0][0], gaps, total - len(got), idx, [g[2] for g in got])


# ------------------------------------------------------------------ the module-level calls

def record_from_board(link: Link, seconds: float) -> Recording:
    return BoardAudio(link).record(seconds)


def play_to_board(link: Link, samples, lead_frames: int = LEAD_FRAMES) -> dict:
    return BoardAudio(link).play(samples, lead_frames=lead_frames)


def record_and_play(link: Link, samples, seconds: float, note_every: float = 0.0,
                    note_midi: Callable[[int], int] = lambda i: 60 + (i % 12)) -> dict:
    """Record `seconds` from the mic while playing `samples`; with note_every > 0 also send one NOTE_IN every
    note_every seconds the whole time and time each answer. Returns {"recording", "play", "notes"}."""
    ba = BoardAudio(link)
    out: dict = {}
    errors: List[str] = []

    def do_rec():
        try:
            out["recording"] = ba.record(seconds)
        except Exception as e:                      # noqa: BLE001
            errors.append(f"record: {type(e).__name__}: {e}")

    def do_play():
        try:
            out["play"] = ba.play(samples)
        except Exception as e:                      # noqa: BLE001
            errors.append(f"play: {type(e).__name__}: {e}")

    # The speaker counters are read inside play(); start it first so its "before" is taken before the
    # recording's gate change.
    tp = threading.Thread(target=do_play, name="duet-play")
    tr = threading.Thread(target=do_rec, name="duet-record")
    tp.start()
    tr.start()
    notes = []
    if note_every > 0:
        t0 = time.perf_counter()
        i = 0
        while tr.is_alive():
            due = t0 + i * note_every
            now = time.perf_counter()
            if due > now:
                time.sleep(due - now)
            if not tr.is_alive():
                break
            try:
                r = link.note(i, note_midi(i), onset=i * note_every, prev_dur=note_every if i else None)
                notes.append({"i": i, "midi": r["midi"], "status": r["status"], "rtt_us": r["rtt_us"],
                              "compute_us": r["compute_us"]})
            except LinkError as e:
                notes.append({"i": i, "error": str(e)})
            i += 1
    tr.join()
    tp.join()
    if errors:
        raise LinkError("; ".join(errors))
    out["notes"] = notes
    return out
