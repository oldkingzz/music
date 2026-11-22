"""The panel: a web page on this Mac only, a thin view over the same functions the checks use.

    .venv/bin/python -m duet_host.webapp            (from esp32_base/host; then open http://127.0.0.1:8765)
    .venv/bin/python -m duet_host.webapp --port 9000

Starting it opens nothing: no serial port, no microphone, no browser. The board is opened when you press
"连板子", the microphone only while you press "录音". It listens on 127.0.0.1 only, so other machines cannot
reach it. Stop it with Ctrl-C.

"板子的声音": record from the board's microphone into a clip (/api/board_record) and play a clip to the board's
speaker (/api/board_play), both in real time through duet_host.audio_stream. Until the audio hardware is wired the
board (and the simulator) report audio=standin, and the page says so.

"现场" (sketch duet_live, INFO live=1, or the simulator as duet_live: target "sim-live"): the board listens, finds
her notes, runs the harmony plug-in and sings. /api/live/start, /set, /stop, /scale (speaker test), /record
(the microphone's 16 kHz hops into a clip) and the fast poll /api/live/poll?since=N. Safe defaults everywhere:
the fake microphone, the speaker off, volume 30. Disconnecting, connecting elsewhere and shutting the server down
first tell the board LISTEN off and SING off (best effort).

Standard library only (http.server); no web framework to install.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
import signal
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np

from . import audio_notes, audio_stream, fake_audio, harmonizers, link as linkmod, proto, runlog
from .blocks import BLOCKS, Context
from .clips import SR, ClipError, ClipStore
from .jobs import JobQueue
from .link import Link, LinkError
from .routes import load_routes, run_route

WEB = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = 100 * 1024 * 1024
MAX_BOARD_SECONDS = 60          # longest recording from / playback to the board in one request (it runs in real time)

# "Twinkle, twinkle" as scale degrees above the tonic (semitones), with lengths in beats.
TWINKLE = [(0, 1), (0, 1), (7, 1), (7, 1), (9, 1), (9, 1), (7, 2), (5, 1), (5, 1), (4, 1), (4, 1), (2, 1), (2, 1), (0, 2)]

# ---- the live chain (sketch duet_live)
LIVE_DEFAULTS = {"fake": True, "sing": False, "volume": 30, "plugin": "third_below", "tonic": 0, "bpm": 120}
LIVE_HISTORY = 2000                     # events kept for /api/live/poll
LIVE_POLL_MAX = 500                     # events in one poll reply at most
LIVE_STATE_KEYS = ("heard", "sung", "freq", "conf", "level", "cents", "hops", "heard_n", "sung_n", "spk_blocks",
                   "volume", "listen", "sing", "fake", "monitor", "mic_up", "spk_up", "error")
SCALE_NOTES = [60, 62, 64, 65, 67, 69, 71, 72]     # C4 D4 E4 F4 G4 A4 B4 C5
SCALE_NOTE_S, SCALE_GAP_S, SCALE_VEL = 0.30, 0.06, 100
MAX_LIVE_RECORD = 30                    # seconds
LINK_WAIT_S = 3.0                       # how long a live request waits for another conversation with the board
NO_LINK_LIVE = "还没有连接。先按“连板子”（板子要刷 duet_live），或者按“连模拟器（现场版）”。"


class ApiError(RuntimeError):
    pass


def _int(v, lo: int, hi: int, what: str) -> int:
    try:
        if isinstance(v, bool):
            raise ValueError
        f = float(v)
        n = int(f)
        if n != f:
            raise ValueError
    except (TypeError, ValueError):
        raise ApiError(f"{what}要是 {lo} 到 {hi} 之间的整数（收到 {v!r}）")
    if not lo <= n <= hi:
        raise ApiError(f"{what}要在 {lo} 到 {hi} 之间（收到 {n}）")
    return n


def _bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)


class App:
    def __init__(self, store: Optional[ClipStore] = None):
        self.store = store or ClipStore()
        self.routes = load_routes()
        self.link: Optional[Link] = None
        self.link_info: dict = {}
        self.link_lock = threading.RLock()      # one conversation with the board at a time
        self.jobs = JobQueue(self._run_job)
        self.events = collections.deque(maxlen=80)
        self.recording = False
        self.board_audio_busy = ""      # "" or what the board's audio is doing now (录音 / 放音)
        self.board_audio_lock = threading.Lock()   # one board recording or playback at a time; others are refused
        # The live session. Link conversations (start / set / record's monitor gate) hold link_lock, like every
        # other conversation with the board; live_lock only guards the session's data (history, counters) and is
        # never held across a link call, so the fast poll never waits for the board.
        self.live_lock = threading.Lock()
        self.live_n = 0                                  # the server's own event number, never reset
        self.live_session = 0                            # +1 at every start
        self.live_events = collections.deque(maxlen=LIVE_HISTORY)
        self.live_rec_busy = False
        self.scale_thread: Optional[threading.Thread] = None
        self.scale_stop = threading.Event()
        self._live_reset()
        self.say("面板启动。还没有连接板子，也没有打开麦克风。")

    def _live_reset(self) -> None:
        """Back to the safe defaults: listening off, fake microphone, speaker off, volume 30."""
        with self.live_lock:
            self.live = dict(LIVE_DEFAULTS, listen=False, monitor=False)
            self.live_state: dict = {}
            self.live_now = {"her": -1, "duet": -1}
            self.live_last_seq: Optional[int] = None     # None: no start yet, do not count lost events
            self.live_lost = 0
            self.live_seconds = 0.0
            self.live_events.clear()

    def say(self, text: str) -> None:
        self.events.appendleft({"time": time.strftime("%H:%M:%S"), "text": text})

    # ------------------------------------------------------------ link
    def connect(self, target: str) -> dict:
        with self.link_lock:
            self.disconnect(quiet=True)
            lk = Link.open(target)
            try:
                info = lk.hello()
            except LinkError:
                lk.close()
                raise
            self.link, self.link_info = lk, {k: v for k, v in info.items() if not k.startswith("_")}
            self._live_reset()
            if self.live_available():
                # Opening the port does not reset a board: an earlier session may have left it listening or
                # singing. Say "everything off" once, so the board matches the panel's safe defaults.
                self._live_off_quiet(lk)
                lk.live_clear()
            what = "板子" if lk.kind == "board" else ("模拟器（现场版）" if lk.port == "sim-live" else "模拟器")
            self.say(f"连上{what}（{lk.port}）：固件 {info.get('fw')} {info.get('ver')}，协议版本 {info.get('proto')}")
            return self.link_state()

    def disconnect(self, quiet: bool = False) -> None:
        with self.link_lock:
            if self.link is not None:
                self._live_off_quiet(self.link)          # never leave a speaker on
                self._live_reset()
                self.link.close()
                self.link = None
                self.link_info = {}
                if not quiet:
                    self.say("已断开。")

    def link_state(self) -> Optional[dict]:
        lk = self.link
        if lk is None:
            return None
        return {"kind": lk.kind, "port": lk.port, "info": self.link_info, "error": lk.error, "stats": lk.stats(),
                "logs": [{"time": time.strftime("%H:%M:%S", time.localtime(t)), "text": s} for t, s in list(lk.logs)[-20:]]}

    def need_link(self) -> Link:
        if self.link is None:
            raise ApiError("还没有连接。先按“连板子”或“连模拟器”。")
        if self.link.error:
            raise ApiError(f"连接断了（{self.link.error}）。重新连接一次。")
        return self.link

    def ping(self, n: int = 20) -> dict:
        with self.link_lock:
            lk = self.need_link()
            r = [lk.ping()["rtt_us"] for _ in range(max(1, min(int(n), 500)))]
        out = {"n": len(r), "median_us": round(runlog.pct(r, 50)), "p95_us": round(runlog.pct(r, 95)), "max_us": round(max(r))}
        self.say(f"ping {out['n']} 次：一来一回中位数 {out['median_us']} 微秒，最慢 {out['max_us']} 微秒")
        return out

    def notes_demo(self, plugin: str, tonic: int) -> dict:
        """Send "Twinkle, twinkle" through the note path, one note at a time, and compare with Python."""
        if plugin not in harmonizers.ID_BY_NAME:
            raise ApiError(f"没有叫 {plugin} 的和声模型")
        tonic = int(tonic) % 12
        ref = harmonizers.make(plugin)
        meta = {"tonic": tonic, "M": 4, "K": 16, "pos": [k % 4 for k in range(16)]}
        ref.start(dict(meta))
        rows, onset, prev = [], 0.0, None
        with self.link_lock:
            lk = self.need_link()
            lk.configure(plugin, tonic, 4, 16, meta["pos"])
            for i, (deg, dur) in enumerate(TWINKLE):
                midi = 60 + tonic + deg
                r = lk.note(i, midi, onset, prev)
                py = int(ref.step(i, onset, midi, prev))
                rows.append({"i": i, "her": midi, "her_name": audio_notes.midi_name(midi), "duet": r["midi"],
                             "duet_name": audio_notes.midi_name(r["midi"]), "python": py, "same": r["midi"] == py,
                             "compute_us": r["compute_us"], "rtt_us": round(r["rtt_us"])})
                onset, prev = onset + dur, float(dur)
        same = sum(r["same"] for r in rows)
        self.say(f"《小星星》{len(rows)} 个音走了一遍音符通道（{plugin}，{audio_notes.PC_NAMES[tonic]} 大调）："
                 f"和 Python 一致 {same}/{len(rows)}")
        return {"rows": rows, "same": same, "n": len(rows), "kind": lk.kind}

    # ------------------------------------------------------------ the board's microphone and speaker
    def _board_audio(self) -> str:
        """The board's audio kind from INFO: "standin", "i2s"; raises if this firmware cannot stream audio."""
        self.need_link()
        mode = self.link_info.get("audio")
        if mode is None and self.link_info.get("live") == "1":
            raise ApiError("这个固件（duet_live）的声音在上面的“现场”那一节：在那里听麦克风、让喇叭出声。"
                           "这里的 24 kHz 录音和放音只给 duet_base_audio 用。")
        if mode is None:
            raise ApiError("这个固件没有声音（第一版固件 duet_base）。要录音或放音，先换成 duet_base_audio。")
        if mode in ("0", "1"):
            raise ApiError("这是旧版的声音固件，读不到板子的声音计数。先把 duet_base_audio 更新到新版。")
        return mode

    def _board_audio_begin(self, what: str) -> None:
        """Refuse (do not queue) a second board recording/playback while one runs."""
        if not self.board_audio_lock.acquire(blocking=False):
            raise ApiError(f"板子正在{self.board_audio_busy or '忙'}，等它做完")
        self.board_audio_busy = what

    def _board_audio_end(self) -> None:
        self.board_audio_busy = ""
        self.board_audio_lock.release()

    def board_record(self, seconds: float) -> dict:
        """Record from the board's microphone (the stand-in melody until the hardware is wired) into a clip."""
        seconds = float(seconds)
        if not 1 <= seconds <= MAX_BOARD_SECONDS:
            raise ApiError(f"从板子录音的长度要在 1 到 {MAX_BOARD_SECONDS} 秒之间")
        self._board_audio_begin("录音")
        try:
            with self.link_lock:
                mode = self._board_audio()
                what = "替身麦克风" if mode == "standin" else "麦克风"
                self.say(f"开始从板子的{what}录 {seconds:g} 秒……")
                r = audio_stream.record_from_board(self.link, seconds)
        finally:
            self._board_audio_end()
        if r.frames == 0:
            raise ApiError("板子一帧声音都没有发过来")
        exact = None
        if mode == "standin":
            exact = bool(r.first_seq == 0 and np.array_equal(r.samples, fake_audio.samples(0, len(r.samples))))
        name = f"板子{'替身' if mode == 'standin' else ''}麦克风 {seconds:g} 秒"
        extra = {"board_audio": mode, "frames": r.frames, "seq_gaps": r.seq_gaps, "missing": r.missing,
                 "frames_per_second": round(r.frames_per_second, 2), "bit_exact": exact}
        summary = f"{r.frames} 帧，每秒 {r.frames_per_second:.1f} 帧，丢了 {r.missing} 帧"
        if exact is not None:
            summary += "；每个采样都和 Mac 算的替身旋律一样" if exact else "；和 Mac 算的替身旋律对不上"
        extra["summary"] = summary
        meta = self.store.add_array(r.samples.astype(np.float32) / 32768.0, name=name, source="board_mic", extra=extra)
        self.say(f"从板子录好了：{meta['id']}，{summary}")
        return meta

    def board_play(self, clip_id: str) -> dict:
        """Play a clip to the board's speaker in real time and read back what the board counted."""
        x, meta = self.store.load(clip_id)
        if len(x) > MAX_BOARD_SECONDS * SR:
            raise ApiError(f"这个片段有 {len(x) / SR:.0f} 秒，放给板子最多 {MAX_BOARD_SECONDS} 秒")
        self._board_audio_begin("放音")
        try:
            with self.link_lock:
                mode = self._board_audio()
                self.say(f"开始把《{meta['name']}》放给板子（{len(x) / SR:.1f} 秒）……")
                p = audio_stream.play_to_board(self.link, x)
        finally:
            self._board_audio_end()
        out = {k: v for k, v in p.items() if k not in ("before", "after")}
        out.update(clip=clip_id, name=meta["name"], board_audio=mode)
        ok = p["frames_played"] == p["frames_sent"] and p["underruns_while_sending"] == 0
        text = f"发了 {p['frames_sent']} 帧，板子放了 {p['frames_played']} 帧，中途断档 {p['underruns_while_sending']} 次"
        if mode == "standin":
            text += "；校验和一致" if p["crc_match"] else "；校验和不一致"
        out["ok"] = ok and (p["crc_match"] if mode == "standin" else True)
        out["summary"] = text
        self.say(f"放完了：{text}")
        return out

    # ------------------------------------------------------------ the live chain (sketch duet_live)
    def live_available(self) -> bool:
        return self.link is not None and self.link_info.get("live") == "1"

    def _need_live(self) -> Link:
        if self.link is None:
            raise ApiError(NO_LINK_LIVE)
        if self.link.error:
            raise ApiError(f"连接断了（{self.link.error}）。重新连接一次。")
        if not self.live_available():
            fw = self.link_info.get("fw") or "?"
            raise ApiError(f"连着的固件（{fw}）没有现场部分（live）。要用这一节：把板子刷成 duet_live，"
                           f"或者先按“断开”，再按“连模拟器（现场版）”。")
        return self.link

    def _hold_link(self):
        """link_lock, but give up after LINK_WAIT_S with a Chinese error instead of hanging the page."""
        app = self

        class _Held:
            def __enter__(self):
                if not app.link_lock.acquire(timeout=LINK_WAIT_S):
                    raise ApiError("板子正在做别的事（发《小星星》、录音、放音或者跑任务），等它做完再试。")

            def __exit__(self, *exc):
                app.link_lock.release()
        return _Held()

    def _live_apply(self, lk: Link, wait_hw: bool = True, timeout: float = 1.0) -> dict:
        """Send the whole wanted gate state (never a partial one: live_ctrl's own defaults are listen off and
        volume 100). Returns the LIVE_STATE reply."""
        w = self.live
        try:
            st = lk.live_ctrl(listen=w["listen"], sing=w["sing"], fake=w["fake"], monitor=w["monitor"],
                              volume=w["volume"], bpm=w["bpm"], wait_hw=wait_hw, timeout=timeout)
        except LinkError as e:
            raise ApiError(f"板子没有回应现场的指令（{e}）。看看 USB 线，或者断开再连一次。")
        with self.live_lock:
            # The stream is in order: every periodic LIVE_STATE already queued arrived before this reply and is
            # older than it. Drop those, so the next _live_pump cannot put an older state over the reply.
            # (A periodic frame arriving after this is newer and replaces the reply as usual.)
            while True:
                try:
                    lk.live_states.popleft()
                except IndexError:
                    break
            self.live_state = st
        return st

    def _live_off_quiet(self, lk: Optional[Link]) -> None:
        """LISTEN off, SING off, MONITOR off; best effort, never raises. Volume and source are kept."""
        self._scale_cancel()
        self.live.update(listen=False, sing=False, monitor=False)
        if lk is None or lk.error or self.link_info.get("live") != "1":
            return
        try:
            self._live_apply(lk, wait_hw=False, timeout=0.5)
        except Exception:
            pass

    def _scale_cancel(self) -> None:
        self.scale_stop.set()
        th = self.scale_thread
        if th is not None and th is not threading.current_thread():
            th.join(timeout=1.0)

    def _live_pump(self) -> None:
        """Move what the link collected into the session: numbered events, lost count, current notes, state."""
        lk = self.link
        if lk is None or self.link_info.get("live") != "1":
            return
        with self.live_lock:
            while True:
                try:
                    d = lk.live_notes.popleft()
                except IndexError:
                    break
                seq = int(d["seq"])
                if self.live_last_seq is not None:
                    self.live_lost += (seq - ((self.live_last_seq + 1) & 0xFF)) & 0xFF
                self.live_last_seq = seq
                self.live_n += 1
                kind, midi = int(d["kind"]), int(d["midi"])
                self.live_events.append({"n": self.live_n, "kind": kind, "kind_name": d.get("kind_name"), "midi": midi,
                                         "name": audio_notes.midi_name(midi), "vel": d.get("vel"),
                                         "t_ms": d.get("t_ms"), "fake": bool(d.get("fake")),
                                         "from_host": bool(d.get("from_host")), "session": self.live_session})
                if kind == proto.LN_HEARD_ON:
                    self.live_now["her"] = midi
                elif kind == proto.LN_HEARD_OFF and self.live_now["her"] == midi:
                    self.live_now["her"] = -1
                elif kind == proto.LN_SUNG_ON:
                    self.live_now["duet"] = midi
                elif kind == proto.LN_SUNG_OFF and self.live_now["duet"] == midi:
                    self.live_now["duet"] = -1
            last = None
            while True:
                try:
                    last = lk.live_states.popleft()
                except IndexError:
                    break
            if last is not None:
                self.live_state = last
            if self.live["listen"] and "hops" in self.live_state:
                self.live_seconds = round(int(self.live_state["hops"]) * proto.LIVE_HOP_MS / 1000.0, 2)

    def live_summary(self) -> dict:
        """The small live object of /api/state."""
        self._live_pump()
        lk, w = self.link, self.live
        avail = self.live_available()
        th = self.scale_thread
        return {"available": avail, "running": bool(avail and w["listen"] and not lk.error),
                "firmware": self.link_info.get("fw") if lk is not None else None,
                "source": "fake" if w["fake"] else "mic", "sing": bool(w["sing"]), "volume": w["volume"],
                "plugin": w["plugin"], "tonic": w["tonic"],
                "scale_busy": bool(th is not None and th.is_alive()), "recording": self.live_rec_busy}

    def live_poll(self, since: int = 0) -> dict:
        out = self.live_summary()
        lk, w = self.link, self.live
        with self.live_lock:
            st = {k: self.live_state[k] for k in LIVE_STATE_KEYS if k in self.live_state}
            if "heard" in st:
                st["heard_name"] = audio_notes.midi_name(int(st["heard"]))
            if "sung" in st:
                st["sung_name"] = audio_notes.midi_name(int(st["sung"]))
            events = [e for e in self.live_events if e["n"] > since][-LIVE_POLL_MAX:]
            now = dict(self.live_now)
            out.update(next=self.live_n, lost=self.live_lost, seconds=self.live_seconds, session=self.live_session)
        now["her_name"] = audio_notes.midi_name(now["her"])
        now["duet_name"] = audio_notes.midi_name(now["duet"])
        info = self.link_info if lk is not None else {}
        out.update(kind=lk.kind if lk is not None else None, plugin=w["plugin"], tonic=w["tonic"],
                   key=audio_notes.PC_NAMES[w["tonic"] % 12], bpm=w["bpm"], state=st, now=now, events=events,
                   mic_pins=info.get("mic_pins"), spk_pins=info.get("spk_pins"),
                   link_error=lk.error if lk is not None else None)
        return out

    def live_start(self, body: dict) -> dict:
        fake = _bool(body.get("fake", LIVE_DEFAULTS["fake"]))
        sing = _bool(body.get("sing", LIVE_DEFAULTS["sing"]))
        volume = _int(body.get("volume", LIVE_DEFAULTS["volume"]), 0, 100, "音量")
        plugin = str(body.get("plugin", LIVE_DEFAULTS["plugin"]))
        if plugin not in harmonizers.ID_BY_NAME:
            raise ApiError(f"没有叫 {plugin} 的和声模型")
        tonic = _int(body.get("tonic", LIVE_DEFAULTS["tonic"]), 0, 11, "调")
        bpm = _int(body.get("bpm", LIVE_DEFAULTS["bpm"]), 30, 255, "速度（每分钟拍数）")
        self._scale_cancel()                      # its notes would mix into the new session's numbering
        with self._hold_link():
            lk = self._need_live()
            # 1. listen off first: a start while listening has no rising edge otherwise, and the board's
            #    reply to this comes after every event of the old session (the stream is in order).
            self.live.update(listen=False, monitor=False, fake=fake, sing=sing, volume=volume, plugin=plugin,
                             tonic=tonic, bpm=bpm)
            self._live_apply(lk, wait_hw=False)
            # 2. forget the old session before anything new can arrive
            lk.live_clear()
            with self.live_lock:
                self.live_events.clear()
                self.live_session += 1
                self.live_last_seq = 0xFF        # the first event of a start carries seq 0
                self.live_lost = 0
                self.live_seconds = 0.0
                self.live_state = {}
                self.live_now = {"her": -1, "duet": -1}
            try:
                lk.configure(plugin, tonic, 4)
            except LinkError as e:
                raise ApiError(f"板子没有接受和声模型 {plugin}（{e}）")
            # 3. listen on
            self.live["listen"] = True
            try:
                st = self._live_apply(lk)
            except ApiError:
                self._live_off_quiet(lk)
                raise
        src = "假麦克风《小星星》" if fake else "真麦克风"
        spk = f"喇叭开着（音量 {volume}）" if sing else "喇叭关着"
        self.say(f"现场开始听：{src}，{spk}，和声模型 {plugin}，{audio_notes.PC_NAMES[tonic]} 大调")
        if st.get("error"):
            self.say("板子说声音硬件没起来（LIVE_STATE error）。看设备日志。")
        return self.live_poll(self.live_n)

    def live_set(self, body: dict) -> dict:
        with self._hold_link():
            lk = self._need_live()
            changed = []
            if "sing" in body:
                self.live["sing"] = _bool(body["sing"])
                changed.append("喇叭" + ("打开" if self.live["sing"] else "关上"))
            if "volume" in body:
                self.live["volume"] = _int(body["volume"], 0, 100, "音量")
                changed.append(f"音量 {self.live['volume']}")
            if not self.live["sing"]:
                self.scale_stop.set()
            self._live_apply(lk)
        if changed:
            self.say("现场：" + "，".join(changed))
        return self.live_poll(self.live_n)

    def live_stop(self) -> dict:
        """Everything off. Safety first: if another conversation holds the board for a while, send anyway
        (link calls themselves are thread safe; only their order with that conversation is not kept)."""
        got = self.link_lock.acquire(timeout=LINK_WAIT_S)
        try:
            lk = self.link
            if lk is None:
                self._live_reset()
                return self.live_poll(self.live_n)
            self._live_off_quiet(lk)
        finally:
            if got:
                self.link_lock.release()
        self._live_pump()
        self.say("现场：停了（不听，喇叭关）。")
        return self.live_poll(self.live_n)

    def live_scale(self) -> dict:
        with self._hold_link():
            lk = self._need_live()
            if not self.live["sing"]:
                raise ApiError("先勾上“板子的喇叭出声”，再放音阶。喇叭关着时板子不出声。")
            th = self.scale_thread
            if th is not None and th.is_alive():
                raise ApiError("音阶还在放，等它放完。")
            stop = threading.Event()
            self.scale_stop = stop
            self.scale_thread = threading.Thread(target=self._scale_run, args=(lk, stop), name="duet-scale",
                                                 daemon=True)
            self.scale_thread.start()
        self.say(f"喇叭测试：放音阶 C4 到 C5（8 个音，每个 {SCALE_NOTE_S * 1000:.0f} 毫秒），音量 {self.live['volume']}")
        return {"started": True, "notes": [audio_notes.midi_name(m) for m in SCALE_NOTES],
                "seconds": round(len(SCALE_NOTES) * (SCALE_NOTE_S + SCALE_GAP_S), 2)}

    def _scale_run(self, lk: Link, stop: threading.Event) -> None:
        try:
            for m in SCALE_NOTES:
                if stop.is_set():
                    break
                lk.synth_note(m, SCALE_VEL)
                stop.wait(SCALE_NOTE_S)
                lk.synth_note(m, 0)                  # always end the note, also when stopped
                if stop.wait(SCALE_GAP_S):
                    break
        except Exception as e:
            self.say(f"音阶没放完：{e}")

    def live_record(self, seconds) -> dict:
        """While listening: the monitor on for that many seconds, the hops (16 kHz int16) into a clip."""
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            raise ApiError(f"录音长度要在 1 到 {MAX_LIVE_RECORD} 秒之间")
        if not 1 <= seconds <= MAX_LIVE_RECORD:
            raise ApiError(f"录音长度要在 1 到 {MAX_LIVE_RECORD} 秒之间")
        with self.live_lock:                         # check and set in one step
            if self.live_rec_busy:
                raise ApiError("正在录麦克风听到的声音，等这一段录完")
            self.live_rec_busy = True
        lk, hops, first_hop = None, [], 0
        need = int(math.ceil(seconds * proto.LIVE_RATE / proto.LIVE_HOP_SAMPLES))
        try:
            with self._hold_link():
                lk = self._need_live()
                if not self.live["listen"]:
                    raise ApiError("先按“开始听”，再录。录的是板子正在听的声音。")
                session, fake = self.live_session, self.live["fake"]
                lk.mic_hops.clear()
                self.live["monitor"] = True
                first_hop = int(self._live_apply(lk, wait_hw=False)["hops"])   # the first monitored hop's index
            self.say(f"开始录板子{'假' if fake else ''}麦克风听到的 {seconds:g} 秒……")
            t_end = time.monotonic() + seconds + 3.0
            while len(hops) < need and time.monotonic() < t_end:
                if self.live_session != session or not self.live["listen"] or self.link is not lk:
                    raise ApiError("录到一半，现场停了或重新开始了。没有存片段。")
                try:
                    hops.append(lk.mic_hops.popleft())
                except IndexError:
                    time.sleep(0.02)
        finally:
            if lk is not None and self.live.get("monitor"):
                self.live["monitor"] = False
                if self.link is lk:
                    try:
                        with self._hold_link():
                            self._live_apply(lk, wait_hw=False)
                    except Exception:
                        pass
                lk.mic_hops.clear()
            self.live_rec_busy = False
        if not hops:
            raise ApiError("板子一帧麦克风的声音都没有发过来")
        parts, lost = [], 0
        for k, h in enumerate(hops):
            if k:
                gap = ((h["seq"] - hops[k - 1]["seq"]) & 0xFF) - 1
                if gap:
                    lost += gap
                    parts.append(np.zeros(gap * proto.LIVE_HOP_SAMPLES, dtype=np.int16))
            parts.append(np.frombuffer(h["pcm"], dtype="<i2"))
        short = max(0, need - len(hops) - lost)
        x = np.concatenate(parts)[: int(round(seconds * proto.LIVE_RATE))]
        exact = None
        if fake:
            exact = bool(lost == 0 and np.array_equal(x, fake_audio.samples16(first_hop * proto.LIVE_HOP_SAMPLES, len(x))))
        summary = f"{len(hops)} 帧（每帧 16 毫秒），丢了 {lost} 帧"
        if short:
            summary += f"，少收到 {short} 帧"
        if exact is not None:
            summary += "；每个采样都和 Mac 算的《小星星》一样" if exact else "；和 Mac 算的《小星星》对不上"
        extra = {"live": True, "rate_in": proto.LIVE_RATE, "mic_source": "fake" if fake else "mic", "hops": len(hops),
                 "lost": lost, "short": short, "first_hop": first_hop, "bit_exact": exact, "summary": summary}
        name = f"板子{'假' if fake else ''}麦克风（现场）{seconds:g} 秒"
        meta = self.store.add_array(x.astype(np.float32) / 32768.0, proto.LIVE_RATE, name=name, source="board_mic",
                                    extra=extra)
        self.say(f"录好了：{meta['id']}，{summary}")
        return meta

    # ------------------------------------------------------------ clips and jobs
    def record(self, seconds: float) -> dict:
        """Record from the Mac's default microphone. Only ever called from the panel's record button."""
        seconds = float(seconds)
        if not 1 <= seconds <= 120:
            raise ApiError("录音长度要在 1 到 120 秒之间")
        if self.recording:
            raise ApiError("正在录音，等这一段录完")
        self.recording = True
        try:
            import sounddevice as sd
            self.say(f"开始用 Mac 的麦克风录 {seconds:g} 秒……")
            x = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            meta = self.store.add_array(x[:, 0], name=f"麦克风录音 {seconds:g} 秒", source="mac_mic")
            self.say(f"录好了：{meta['id']}，峰值 {meta['peak']:.2f}")
            return meta
        finally:
            self.recording = False

    def submit(self, route_name: str, clip_id: str, params: dict) -> dict:
        route = next((r for r in self.routes if r["name"] == route_name), None)
        if route is None:
            raise ApiError(f"没有叫 {route_name} 的路线")
        if not route["ready"]:
            raise ApiError(f"这条路线现在跑不了。{route['why_not']}")
        self.store.meta(clip_id)                 # raises if the clip does not exist
        for b, p in (params or {}).items():
            if b in BLOCKS:
                BLOCKS[b].clean(p)               # raises now, not later inside the worker
        return self.jobs.submit(route=route_name, title=route["title"], clip=clip_id, params=params or {})

    def _run_job(self, job: dict) -> dict:
        route = next(r for r in self.routes if r["name"] == job["route"])
        with self.link_lock:
            res = run_route(route, job["clip"], job["params"], self.store, Context(link=self.link))
        self.say(f"任务 {job['id']}（{route['title']}）做完：{res['steps'][-1]['summary']}")
        return {"clip_out": res["clip"]["id"], "steps": res["steps"]}

    # ------------------------------------------------------------ everything the page shows
    def state(self) -> dict:
        try:
            ports = linkmod.list_usb_serial()
        except Exception:
            ports = []
        return {
            "link": self.link_state(),
            "ports": ports,
            "board_ports": [p["device"] for p in ports if (p["vid"], p["pid"]) == (linkmod.USB_VID, linkmod.USB_PID)],
            "sim_built": linkmod.SIM_EXE.exists(),
            "plugins": list(harmonizers.ID_BY_NAME),
            "blocks": {k: b.describe() for k, b in BLOCKS.items()},
            "routes": self.routes,
            "clips": self.store.list()[:60],
            "jobs": self.jobs.list()[:30],
            "events": list(self.events),
            "recording": self.recording,
            "board_audio": (self.link_info.get("audio") if self.link is not None else None),
            "board_audio_busy": self.board_audio_busy,
            "live": self.live_summary(),       # available = this firmware's sound is the live section (duet_live)
            "disk_free_gb": round(shutil.disk_usage(self.store.root).free / 1e9, 2),
        }


class Handler(BaseHTTPRequestHandler):
    app: App = None            # set by serve()
    server_version = "DuetPanel/0.1"

    def log_message(self, fmt, *args):   # quiet: the page polls every 1.5 s
        pass

    # ------------------------------------------------------------ helpers
    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ApiError(f"文件太大（上限 {MAX_UPLOAD // 1024 // 1024} MB）")
        return self.rfile.read(n) if n else b""

    def _wav(self, cid: str) -> None:
        data = self.app.store.wav_path(cid).read_bytes()
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):               # Safari asks for byte ranges before it plays
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, len(data) - int(b))
            end = min(int(b), len(data) - 1) if a and b else len(data) - 1
            if start > end or start >= len(data):
                self._send(416, b"", "audio/wav", {"Content-Range": f"bytes */{len(data)}"})
                return
            self._send(206, data[start:end + 1], "audio/wav",
                       {"Content-Range": f"bytes {start}-{end}/{len(data)}", "Accept-Ranges": "bytes"})
        else:
            self._send(200, data, "audio/wav", {"Accept-Ranges": "bytes"})

    # ------------------------------------------------------------ routes of the web server
    def do_GET(self):
        try:
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(self.app.state())
            elif path == "/api/live/poll":
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                try:
                    since = int((q.get("since") or ["0"])[0])
                except ValueError:
                    since = 0
                self._json(self.app.live_poll(since))
            elif path.startswith("/api/clip/") and path.endswith(".wav"):
                self._wav(path[len("/api/clip/"):-4])
            else:
                self._json({"error": "not found"}, 404)
        except (ApiError, ClipError, LinkError) as e:
            self._json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:      # never leave the page without an answer
            self._json({"error": f"服务器出错：{type(e).__name__}: {e}"}, 500)

    do_HEAD = do_GET

    def do_POST(self):
        app = self.app
        try:
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            if url.path == "/api/upload":
                name = (q.get("name") or ["上传的文件"])[0][:120]
                self._json(app.store.add_bytes(self._body(), name))
                app.say(f"收到文件：{name}")
                return
            body = json.loads(self._body() or b"{}")
            if url.path == "/api/link":
                target = body.get("target", "none")
                if target == "none":
                    app.disconnect()
                    self._json({"link": None})
                else:
                    self._json({"link": app.connect(target)})
            elif url.path == "/api/ping":
                self._json(app.ping(body.get("n", 20)))
            elif url.path == "/api/notes_demo":
                self._json(app.notes_demo(body.get("plugin", "third_below"), body.get("tonic", 0)))
            elif url.path == "/api/record":
                self._json(app.record(body.get("seconds", 5)))
            elif url.path == "/api/board_record":
                self._json(app.board_record(body.get("seconds", 8)))
            elif url.path == "/api/board_play":
                self._json(app.board_play(body.get("clip", "")))
            elif url.path == "/api/live/start":
                self._json(app.live_start(body if isinstance(body, dict) else {}))
            elif url.path == "/api/live/set":
                self._json(app.live_set(body if isinstance(body, dict) else {}))
            elif url.path == "/api/live/stop":
                self._json(app.live_stop())
            elif url.path == "/api/live/scale":
                self._json(app.live_scale())
            elif url.path == "/api/live/record":
                self._json(app.live_record(body.get("seconds", 5) if isinstance(body, dict) else 5))
            elif url.path == "/api/run":
                self._json(app.submit(body.get("route", ""), body.get("clip", ""), body.get("params", {})))
            else:
                self._json({"error": "not found"}, 404)
        except (ApiError, ClipError, LinkError) as e:
            app.say(f"没做成：{e}")
            self._json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:      # a block's own error class, a bad number, ...
            app.say(f"没做成：{e}")
            self._json({"error": f"{e}"}, 400)


def serve(port: int = 8765, app: Optional[App] = None) -> ThreadingHTTPServer:
    Handler.app = app or App()
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    ap = argparse.ArgumentParser(description="Duet base panel (this Mac only)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = serve(a.port)
    print(f"Duet panel: http://127.0.0.1:{a.port}   (Ctrl-C stops it; nothing is connected until you press a button)")

    def _term(*_):                  # a plain kill also ends through the finally below (speaker off, link closed)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.app.disconnect(quiet=True)
        srv.server_close()


if __name__ == "__main__":
    main()
