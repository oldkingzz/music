"""One connection to the board, or to its Mac stand-in.

    from duet_host.link import Link
    link = Link.open("auto")          # the board if it is plugged in, else an error
    link = Link.open("sim")           # firmware/host_tests/build/board_sim (the same C++ core as the board)
    link = Link.open("/dev/cu.usbmodem101")
    link.hello()                      # {'fw': 'duet_base', 'psram': '8388608', ...}
    link.ping()                       # {'rtt_us': ..., 'dev_us': ...}
    link.configure("third_below", tonic=0, M=4)
    link.note(0, midi=64, onset=0.0, prev_dur=None)   # {'midi': 60, 'compute_us': ..., 'rtt_us': ...}
    link.close()

Rules that come from the hardware:
  * The port is opened once and DTR/RTS are never driven: on the ESP32-S3 USB-Serial/JTAG port those lines
    can reset the chip (that is how flashing tools enter download mode).
  * The board is found by USB id (303A:1001), never by port name: the name changes between plugs.
A background thread reads all the time, so the board's transmit buffer never fills.
"""
from __future__ import annotations

import collections
import os
import select
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

from . import harmonizers, proto
from .proto import T

ROOT = Path(__file__).resolve().parents[2]                     # esp32_base/
SIM_EXE = ROOT / "firmware" / "host_tests" / "build" / "board_sim"
USB_VID, USB_PID = 0x303A, 0x1001                              # Espressif USB-Serial/JTAG


class LinkError(RuntimeError):
    pass


def find_board_ports():
    """Serial ports whose USB id is the ESP32-S3 USB-Serial/JTAG controller."""
    from serial.tools import list_ports
    return [p.device for p in list_ports.comports() if p.vid == USB_VID and p.pid == USB_PID]


def list_usb_serial():
    """Every USB serial port with its id, for error messages and for the web panel."""
    from serial.tools import list_ports
    return [{"device": p.device, "vid": p.vid, "pid": p.pid, "product": p.product, "serial": p.serial_number}
            for p in list_ports.comports() if p.vid is not None]


class SerialTransport:
    kind = "board"

    def __init__(self, port: str):
        import termios

        import serial
        s = serial.Serial()
        s.port = port
        s.baudrate = 115200                  # ignored by USB CDC
        s.timeout = 0.02
        s.write_timeout = 0.5
        s.dsrdtr = False
        s.rtscts = False
        s._update_dtr_state = lambda: None   # instance attributes shadow the class methods open() calls,
        s._update_rts_state = lambda: None   # so pyserial neither raises nor lowers DTR/RTS
        s.open()
        try:
            attrs = termios.tcgetattr(s.fd)
            attrs[2] &= ~termios.HUPCL       # do not drop the lines on close either
            termios.tcsetattr(s.fd, termios.TCSANOW, attrs)
        except termios.error:
            pass
        self.s = s
        self.name = port

    def read(self, timeout: float) -> bytes:
        self.s.timeout = timeout
        data = self.s.read(1)
        if data:
            n = self.s.in_waiting
            if n:
                data += self.s.read(n)
        return data

    def write(self, data: bytes) -> None:
        self.s.write(data)
        self.s.flush()

    def close(self) -> None:
        try:
            self.s.close()
        except Exception:
            pass


class SimTransport:
    kind = "sim"

    def __init__(self, exe: Path = SIM_EXE):
        if not Path(exe).exists():
            raise LinkError(f"the simulator is not built: run  sh firmware/host_tests/build_and_test.sh  ({exe})")
        self.p = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.name = "sim"

    def read(self, timeout: float) -> bytes:
        fd = self.p.stdout.fileno()
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            return b""
        data = os.read(fd, 65536)
        if not data:
            raise LinkError("the simulator exited")
        return data

    def write(self, data: bytes) -> None:
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def close(self) -> None:
        try:
            self.p.stdin.close()
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()
        finally:
            try:
                self.p.stdout.close()    # the reader thread has stopped by now (Link.close joins it first)
            except Exception:
                pass


class Link:
    STALE_S = 0.1          # give up on a partial frame after this much silence (as the device does)

    def __init__(self, transport):
        self.t = transport
        self.kind = transport.kind          # "board" or "sim"
        self.port = transport.name
        self.parser = proto.Parser()
        self.logs = collections.deque(maxlen=200)       # (time, text) of LOG frames
        self.last_status: Optional[dict] = None
        self.on_frame: Optional[Callable[[proto.Frame], None]] = None   # every frame, for audio and monitors
        self.rx_frames = 0
        self.tx_frames = 0
        self.error: Optional[str] = None
        self._seq: Dict[int, int] = collections.defaultdict(int)
        self._waiting: Dict[Tuple[int, int], list] = {}
        self._lock = threading.Lock()
        self._wlock = threading.Lock()
        self._stop = False
        self._plugin = None
        self._th = threading.Thread(target=self._reader, name="duet-link-reader", daemon=True)
        self._th.start()

    # ------------------------------------------------------------ opening
    @classmethod
    def open(cls, target: str = "auto") -> "Link":
        if target == "sim":
            return cls(SimTransport())
        if target == "auto":
            ports = find_board_ports()
            if not ports:
                seen = ", ".join(f"{p['device']} ({p['vid']:04X}:{p['pid']:04X} {p['product']})"
                                 for p in list_usb_serial()) or "none"
                raise LinkError(f"no board found (looking for USB id {USB_VID:04X}:{USB_PID:04X}). "
                                f"USB serial ports now: {seen}")
            if len(ports) > 1:
                raise LinkError(f"more than one board found: {ports}. Name the port.")
            target = ports[0]
        return cls(SerialTransport(target))

    def close(self) -> None:
        self._stop = True
        self._th.join(timeout=1.0)
        self.t.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ reader thread
    def _reader(self) -> None:
        last_rx = time.monotonic()
        while not self._stop:
            try:
                data = self.t.read(0.02)
            except Exception as e:            # unplugged, simulator gone
                self.error = f"{type(e).__name__}: {e}"
                break
            if data:
                last_rx = time.monotonic()
                frames = self.parser.feed(data)
            elif self.parser.pending and time.monotonic() - last_rx > self.STALE_S:
                last_rx = time.monotonic()
                frames = self.parser.drop_stale()
            else:
                continue
            for fr in frames:
                self._dispatch(fr)
        with self._lock:                      # wake everyone who is still waiting
            for slot in self._waiting.values():
                slot[0].set()

    def _dispatch(self, fr: proto.Frame) -> None:
        self.rx_frames += 1
        if fr.type == T.LOG:
            self.logs.append((time.time(), proto.decode_payload(fr)))
        elif fr.type == T.STATUS:
            self.last_status = proto.decode_payload(fr)
        with self._lock:
            slot = self._waiting.pop((fr.type, fr.seq), None)
        if slot is not None:
            slot[1] = (fr, time.perf_counter())
            slot[0].set()
        cb = self.on_frame
        if cb is not None:
            try:
                cb(fr)
            except Exception:
                pass

    # ------------------------------------------------------------ sending
    def _next_seq(self, type_: int) -> int:
        s = self._seq[type_]
        self._seq[type_] = (s + 1) & 0xFF
        return s

    def send(self, data: bytes) -> None:
        if self.error:
            raise LinkError(f"link is down: {self.error}")
        with self._wlock:
            self.t.write(data)
            self.tx_frames += 1

    def _request(self, type_: int, build: Callable[[int], bytes], reply: int, timeout: float):
        """Send one frame and wait for the reply that carries the same seq.
        Returns (reply frame, seconds from just before the write to the reply's arrival)."""
        seq = self._next_seq(type_)
        slot = [threading.Event(), None]
        with self._lock:
            self._waiting[(reply, seq)] = slot
        data = build(seq)
        t0 = time.perf_counter()
        self.send(data)
        if not slot[0].wait(timeout) or slot[1] is None:
            with self._lock:
                self._waiting.pop((reply, seq), None)
            why = self.error or f"no {T(reply).name} within {timeout:.1f} s"
            raise LinkError(f"{T(type_).name} seq {seq}: {why}")
        fr, t1 = slot[1]
        return fr, t1 - t0

    # ------------------------------------------------------------ the calls
    def ping(self, timeout: float = 1.0) -> dict:
        host_us = int(time.perf_counter() * 1e6) & 0xFFFFFFFF
        fr, dt = self._request(T.PING, lambda s: proto.ping(s, host_us), T.PONG, timeout)
        d = proto.decode_payload(fr)
        if d["host_us"] != host_us:
            raise LinkError("PONG does not echo the PING")
        return {"rtt_us": dt * 1e6, "dev_us": d["dev_us"]}

    def hello(self, timeout: float = 1.0) -> dict:
        fr, dt = self._request(T.HELLO, proto.hello, T.INFO, timeout)
        d = proto.decode_payload(fr)
        d["_rtt_us"] = dt * 1e6
        return d

    def configure(self, plugin, tonic: int, M: int, K: int = 0, pos: Sequence[int] = (),
                  timeout: float = 1.0) -> dict:
        """Select a plug-in (name or id) and tell it about the song. Raises if the device refuses."""
        pid = harmonizers.ID_BY_NAME[plugin] if isinstance(plugin, str) else int(plugin)
        fr, dt = self._request(T.CONFIG, lambda s: proto.config(s, pid, tonic, M, K, pos), T.CONFIG_ACK, timeout)
        d = proto.decode_payload(fr)
        if d["status"] != proto.ACK_OK:
            raise LinkError(f"CONFIG refused: status {d['status']} for plug-in {plugin!r}")
        self._plugin = pid
        d["rtt_us"] = dt * 1e6
        return d

    def note(self, i: int, midi: int, onset: float, prev_dur: Optional[float], timeout: float = 1.0) -> dict:
        """One of her notes in, Duet's note out (midi = -1 means silent)."""
        fr, dt = self._request(T.NOTE_IN, lambda s: proto.note_in(s, i, midi, onset, prev_dur), T.NOTE_OUT, timeout)
        d = proto.decode_payload(fr)
        if d["i"] != (i & 0xFFFF):
            raise LinkError(f"NOTE_OUT answers note {d['i']}, expected {i}")
        d["rtt_us"] = dt * 1e6
        return d

    def led(self, state: int) -> None:
        self.send(proto.led(self._next_seq(T.LED), state))

    def stats(self) -> dict:
        s = self.parser.stats
        return {"kind": self.kind, "port": self.port, "rx_frames": self.rx_frames, "tx_frames": self.tx_frames,
                "bad_checksum": s.bad_checksum, "bad_header": s.bad_header, "bytes_skipped": s.bytes_skipped,
                "stale_drops": s.stale_drops, "error": self.error}
