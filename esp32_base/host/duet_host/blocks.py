"""The blocks: one block = one feature's own piece of computation.

Every block says the same four things about itself, the ones the base is built around:
    takes / gives   notes, audio (live sound) or clip (a finished recording)
    lane            how long the listener can wait: now, phrase (a few seconds), song (afterwards)
    place           where it computes: board, mac, cloud
    status          ready, or why it cannot run yet
and has one function:  run(ctx, x, params) -> {'audio': samples at 24 kHz, 'summary': text, ...}

Adding a feature = writing one function here, adding it to BLOCKS, and adding one line to host/routes.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np

from . import audio_notes, harmonizers
from .clips import SR


class BlockError(RuntimeError):
    """The block ran and could not do its job (bad input, nothing heard, ...)."""


class BlockUnavailable(RuntimeError):
    """The block cannot run on this machine yet (a tool or hardware is missing)."""


@dataclass
class Context:
    link: object = None            # duet_host.link.Link or None


@dataclass
class Block:
    name: str
    title: str                     # Chinese, shown in the panel
    takes: str
    gives: str
    lane: str
    place: str
    run: Callable
    status: str = "ready"          # ready | needs_tool | needs_board
    note: str = ""                 # Chinese, one line: what it does or what is missing
    params: Dict[str, dict] = field(default_factory=dict)

    def describe(self) -> dict:
        return {k: getattr(self, k) for k in ("name", "title", "takes", "gives", "lane", "place", "status", "note", "params")}

    def defaults(self) -> dict:
        return {k: (int(v["default"]) if v.get("cast") == "int" else v["default"]) for k, v in self.params.items()}

    def clean(self, params: dict) -> dict:
        """Defaults for what is missing; numbers forced into their type and range."""
        out = self.defaults()
        for k, spec in self.params.items():
            if k not in params or params[k] in ("", None):
                continue
            v = params[k]
            if spec["type"] == "int":
                v = int(float(v))
            elif spec["type"] == "float":
                v = float(v)
            elif spec["type"] == "choice":
                v = str(v)
                if v not in [c[0] for c in spec["choices"]]:
                    raise BlockError(f"{k}: {v!r} is not one of the choices")
                if spec.get("cast") == "int":
                    v = int(v)
            if spec["type"] in ("int", "float"):
                if not spec["min"] <= v <= spec["max"]:
                    raise BlockError(f"{k} = {v} is outside {spec['min']}..{spec['max']}")
            out[k] = v
        return out


def _limit(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (0.98 / peak)).astype(np.float32) if peak > 0.98 else x.astype(np.float32)


def shift_pitch(x: np.ndarray, semitones: float) -> np.ndarray:
    """Move the pitch, keep the length. Everything moves together, the voice's colour too: a few semitones
    sound natural, an octave sounds like a different creature."""
    if semitones == 0:
        return np.asarray(x, dtype=np.float32)
    import librosa
    return librosa.effects.pitch_shift(np.asarray(x, dtype=np.float32), sr=SR, n_steps=float(semitones)).astype(np.float32)


# ------------------------------------------------------------------ blocks
def run_canon(ctx: Context, x: np.ndarray, p: dict) -> dict:
    """Round: the same phrase again, delay_beats later, on top of the original."""
    d = int(round(p["delay_beats"] * 60.0 / p["bpm"] * SR))
    y = shift_pitch(x, p["semitones"])
    out = np.zeros(len(x) + d * p["voices"], dtype=np.float64)
    out[:len(x)] += x
    for k in range(1, p["voices"] + 1):
        out[k * d:k * d + len(y)] += p["gain"] * y
    return {"audio": _limit(out),
            "summary": f"{p['voices']} 个跟唱声部，每个晚 {p['delay_beats']:g} 拍（{d / SR:.2f} 秒）"
                       + (f"，移 {p['semitones']:+d} 个半音" if p["semitones"] else "")}


def run_transpose(ctx: Context, x: np.ndarray, p: dict) -> dict:
    return {"audio": _limit(shift_pitch(x, p["semitones"])), "summary": f"整段移 {p['semitones']:+d} 个半音，长度不变"}


def run_harmony(ctx: Context, x: np.ndarray, p: dict) -> dict:
    """Listen (sound -> notes) on the Mac, harmony model on the board (or its Python reference when no board
    is connected), sing (notes -> sound) on the Mac, then mix."""
    notes = audio_notes.clip_to_notes(x)
    if not notes:
        raise BlockError("这段录音里没有听出音符（太轻、太短，或者不是单个人声）")
    tonic = p["tonic"] if p["tonic"] >= 0 else audio_notes.estimate_tonic(notes)
    spb = 60.0 / p["bpm"]                                   # seconds per beat
    t0 = notes[0]["on"]                                     # beat 0 = her first note
    M = p["meter"]
    n_beats = int((notes[-1]["off"] - t0) / spb) + 1
    meta = {"tonic": tonic, "M": M, "K": n_beats, "pos": [k % M for k in range(n_beats)]}
    ref = harmonizers.make(p["plugin"])
    ref.start(dict(meta))
    link = ctx.link if ctx.link is not None and not ctx.link.error else None
    if link is not None:
        link.configure(p["plugin"], tonic, M, n_beats, meta["pos"])
    rows, differ, max_us, prev = [], 0, 0, None
    for i, n in enumerate(notes):
        onset = (n["on"] - t0) / spb
        prev_dur = None if prev is None else (prev["off"] - prev["on"]) / spb
        py = int(ref.step(i, onset, n["midi"], prev_dur))
        if link is not None:
            r = link.note(i, n["midi"], onset, prev_dur)
            h, max_us = int(r["midi"]), max(max_us, r["compute_us"])
            differ += h != py
        else:
            h = py
        rows.append({"on": round(n["on"], 3), "off": round(n["off"], 3), "her": n["midi"], "duet": h})
        prev = n
    voice = audio_notes.notes_to_voice([(r["on"], r["off"] - r["on"], r["duet"]) for r in rows if r["duet"] >= 0], len(x))
    where = "板子" if link is not None and link.kind == "board" else "模拟器" if link is not None else "Mac 上的 Python"
    summary = (f"听出 {len(rows)} 个音，调 {audio_notes.PC_NAMES[tonic]} 大调"
               + ("（自动估计）" if p["tonic"] < 0 else "") + f"，和声模型 {p['plugin']} 在{where}上算")
    if link is not None:
        summary += f"，每个音最多 {max_us} 微秒，和 Python 不一致的音 {differ} 个"
    return {"audio": _limit(np.asarray(x, dtype=np.float64) + p["gain"] * voice), "summary": summary, "notes": rows,
            "tonic": tonic, "where": where, "differ_from_python": differ}


def run_voice_convert(ctx: Context, x: np.ndarray, p: dict) -> dict:
    raise BlockUnavailable("歌声转换工具还没装。这一块需要一个公开的歌声转换模型（在云端或这台 Mac 上跑），选定并装好之后接在这里。")


_NUM = lambda t, default, lo, hi, label: {"type": t, "default": default, "min": lo, "max": hi, "label": label}  # noqa: E731

BLOCKS: Dict[str, Block] = {b.name: b for b in [
    Block("harmony", "配和声", "clip", "clip", "song", "board", run_harmony,
          note="听出你唱的音，交给板子上的和声模型，再用机器嗓音唱出来，和原声叠在一起。没连板子时用 Mac 上的同一个模型。",
          params={"plugin": {"type": "choice", "default": "third_below", "label": "和声模型",
                             "choices": [["third_below", "低三度（third_below）"], ["echo", "原样跟唱（echo，只测通路）"]]},
                  "tonic": {"type": "choice", "default": "-1", "cast": "int", "label": "调（大调的主音）",
                            "choices": [["-1", "自动估计"]] + [[str(k), n] for k, n in enumerate(audio_notes.PC_NAMES)]},
                  "bpm": _NUM("float", 100.0, 30.0, 240.0, "速度（每分钟拍数）"),
                  "meter": _NUM("int", 4, 2, 6, "每小节几拍"),
                  "gain": _NUM("float", 0.8, 0.0, 2.0, "和声音量")}),
    Block("canon", "轮唱", "clip", "clip", "phrase", "mac", run_canon,
          note="同一句晚几拍再进来一遍，叠在原声上。接上麦克风和喇叭之后，这一块搬到板子上。",
          params={"bpm": _NUM("float", 100.0, 30.0, 240.0, "速度（每分钟拍数）"),
                  "delay_beats": _NUM("float", 8.0, 0.5, 64.0, "晚几拍进来"),
                  "voices": _NUM("int", 1, 1, 3, "跟唱声部数"),
                  "semitones": _NUM("int", 0, -12, 12, "跟唱声部移几个半音"),
                  "gain": _NUM("float", 0.8, 0.0, 1.5, "跟唱音量")}),
    Block("transpose", "移调", "clip", "clip", "song", "mac", run_transpose,
          note="整段一起升高或降低，长度不变。嗓音的特点也跟着变，移几个半音自然，移多了会怪。",
          params={"semitones": _NUM("int", -3, -12, 12, "移几个半音（负数是降）")}),
    Block("voice_convert", "换嗓音", "clip", "clip", "song", "cloud", run_voice_convert, status="needs_tool",
          note="把嗓音换成另一种，词和旋律不变。还没装工具，现在不能跑。"),
]}

