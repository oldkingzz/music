"""The panel: a web page on this Mac only, a thin view over the same functions the checks use.

    .venv/bin/python -m duet_host.webapp            (from esp32_base/host; then open http://127.0.0.1:8765)
    .venv/bin/python -m duet_host.webapp --port 9000

Starting it opens nothing: no serial port, no microphone, no browser. The board is opened when you press
"连板子", the microphone only while you press "录音". It listens on 127.0.0.1 only, so other machines cannot
reach it. Stop it with Ctrl-C.

Standard library only (http.server); no web framework to install.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from . import audio_notes, harmonizers, link as linkmod, runlog
from .blocks import BLOCKS, Context
from .clips import SR, ClipError, ClipStore
from .jobs import JobQueue
from .link import Link, LinkError
from .routes import load_routes, run_route

WEB = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = 100 * 1024 * 1024

# "Twinkle, twinkle" as scale degrees above the tonic (semitones), with lengths in beats.
TWINKLE = [(0, 1), (0, 1), (7, 1), (7, 1), (9, 1), (9, 1), (7, 2), (5, 1), (5, 1), (4, 1), (4, 1), (2, 1), (2, 1), (0, 2)]


class ApiError(RuntimeError):
    pass


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
        self.say("面板启动。还没有连接板子，也没有打开麦克风。")

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
            what = "板子" if lk.kind == "board" else "模拟器"
            self.say(f"连上{what}（{lk.port}）：固件 {info.get('fw')} {info.get('ver')}，协议版本 {info.get('proto')}")
            return self.link_state()

    def disconnect(self, quiet: bool = False) -> None:
        with self.link_lock:
            if self.link is not None:
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
            elif path.startswith("/api/clip/") and path.endswith(".wav"):
                self._wav(path[len("/api/clip/"):-4])
            else:
                self._json({"error": "not found"}, 404)
        except (ApiError, ClipError, LinkError) as e:
            self._json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass

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
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.app.disconnect(quiet=True)
        srv.server_close()


if __name__ == "__main__":
    main()
