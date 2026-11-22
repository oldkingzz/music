"""Tests of the panel's live section (sketch duet_live): the /api/live/* endpoints against the simulator in live mode
(target "sim-live"), and the page's script run in node against a hand-written DOM with JSON recorded from the real
server (page_harness.js). Standard library unittest.

    .venv/bin/python host/tests/test_web_live.py            (from esp32_base/)

Touches no hardware: the link is always the simulator ("sim", "sim-live", or the simulator as sketch 1), never "auto"
or a /dev path; no microphone or speaker of this Mac is used; no browser is opened. Each server binds an ephemeral
port on 127.0.0.1 and is shut down; clips go to a temporary directory. One server at a time (serve() sets the
handler's App on the class).
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import test_host  # noqa: E402  (its _Server and Http helpers)
from duet_host import fake_audio, webapp  # noqa: E402
from duet_host.clips import to_mono_24k  # noqa: E402
from duet_host.link import SIM_EXE, Link, SimTransport  # noqa: E402

INDEX = HOST / "duet_host" / "web" / "index.html"
HARNESS = Path(__file__).resolve().parent / "page_harness.js"
TWINKLE = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]
TWINKLE_DUET = [57, 57, 64, 64, 65, 65, 64, 62, 62, 60, 60, 59, 59, 57]     # third_below in C major
SCALE = [60, 62, 64, 65, 67, 69, 71, 72]
CJK = re.compile(r"[一-鿿]")
STATE_KEYS = ("heard", "sung", "heard_name", "sung_name", "freq", "conf", "level", "cents", "hops", "heard_n", "sung_n",
              "spk_blocks", "volume", "listen", "sing", "fake", "monitor", "mic_up", "spk_up", "error")
POLL_KEYS = ("available", "running", "kind", "source", "sing", "volume", "plugin", "tonic", "state", "events", "next",
             "lost", "scale_busy", "now", "session", "seconds")


def find_node():
    for c in ("/usr/local/bin/node", shutil.which("node")):
        if c and Path(c).exists():
            return c
    return None


class LiveClient:
    """A few helpers over test_host.Http for the live endpoints."""

    def __init__(self, http):
        self.h = http

    def post(self, path, body=None):
        return self.h.json("POST", path, {} if body is None else body)

    def poll(self, since=0):
        st, r = self.h.json("GET", f"/api/live/poll?since={since}")
        assert st == 200, r
        return r

    def collect(self, since, until, timeout):
        """Poll as the page does (~120 ms) until until(events) is true or timeout. Returns (events, last reply)."""
        events, r, end = [], None, time.monotonic() + timeout
        while time.monotonic() < end:
            r = self.poll(since)
            events += r["events"]
            since = r["next"]
            if until(events):
                break
            time.sleep(0.12)
        return events, r


def heard(events):
    return [e["midi"] for e in events if e["kind"] == 1]


def sung(events):
    return [e["midi"] for e in events if e["kind"] == 3 and not e["from_host"]]


# ====================================================================== the server
class TestWebLive(unittest.TestCase):
    """One server for the class; the tests run in name order and build on each other."""

    @classmethod
    def setUpClass(cls):
        if not SIM_EXE.exists():
            raise RuntimeError(f"{SIM_EXE} is missing")
        cls.s = test_host._Server().start(cls)
        cls.c = LiveClient(cls.s.http)
        cls.app = cls.s.app

    @classmethod
    def tearDownClass(cls):
        cls.s.stop()

    def state(self):
        st, s = self.s.http.json("GET", "/api/state")
        self.assertEqual(st, 200)
        return s

    def assertChineseError(self, st, r, *must):
        self.assertEqual(st, 400, r)
        self.assertIsInstance(r.get("error"), str)
        self.assertRegex(r["error"], CJK)
        for m in must:
            self.assertIn(m, r["error"])

    def test_01_no_link(self):
        st, r = self.c.post("/api/live/start")
        self.assertChineseError(st, r, "duet_live", "连模拟器（现场版）")
        for path, body in (("/api/live/set", {"sing": True}), ("/api/live/scale", {}), ("/api/live/record", {"seconds": 2})):
            with self.subTest(path=path):
                st, r = self.c.post(path, body)
                self.assertChineseError(st, r)
        st, r = self.c.post("/api/live/stop")             # always safe
        self.assertEqual(st, 200, r)
        p = self.c.poll(0)
        for k in POLL_KEYS:
            self.assertIn(k, p)
        self.assertEqual((p["available"], p["running"], p["events"], p["kind"]), (False, False, [], None))
        self.assertEqual((p["source"], p["sing"], p["volume"]), ("fake", False, 30))
        live = self.state()["live"]
        self.assertEqual((live["available"], live["running"]), (False, False))
        st, _ = self.s.http.json("GET", "/api/live/poll?since=abc")
        self.assertEqual(st, 200)

    def test_02_plain_sim_is_refused(self):
        self.assertEqual(self.c.post("/api/link", {"target": "sim"})[0], 200)
        st, r = self.c.post("/api/live/start")
        self.assertChineseError(st, r, "没有现场部分", "duet_live", "连模拟器（现场版）")
        s = self.state()
        self.assertEqual(s["board_audio"], "standin")          # the old firmware's section is unchanged
        self.assertEqual((s["live"]["available"], s["live"]["running"]), (False, False))
        self.assertEqual(self.c.post("/api/link", {"target": "none"})[0], 200)

    def test_03_start_defaults_and_the_melody(self):
        st, r = self.c.post("/api/link", {"target": "sim-live"})
        self.assertEqual(st, 200, r)
        self.assertEqual(r["link"]["port"], "sim-live")
        s = self.state()
        self.assertEqual(s["live"], {"available": True, "running": False, "firmware": "duet_live", "source": "fake",
                                     "sing": False, "volume": 30, "plugin": "third_below", "tonic": 0,
                                     "scale_busy": False, "recording": False})
        self.assertIsNone(s["board_audio"])                    # no audio= key: the page uses live.available
        st, r = self.c.post("/api/live/start")                 # empty body: the safe defaults
        self.assertEqual(st, 200, r)
        for k in POLL_KEYS:
            self.assertIn(k, r)
        self.assertEqual((r["running"], r["source"], r["sing"], r["volume"], r["plugin"], r["tonic"], r["lost"]),
                         (True, "fake", False, 30, "third_below", 0, 0))
        lk = self.app.link
        self.assertTrue(lk.last_live_state["listen"] and lk.last_live_state["fake"])
        self.assertFalse(lk.last_live_state["sing"] or lk.last_live_state["spk_up"] or lk.last_live_state["mic_up"])
        self.assertEqual(lk.last_live_state["volume"], 30)
        start_next = r["next"]
        events, p = self.c.collect(start_next, lambda ev: len(heard(ev)) >= 14 and len(sung(ev)) >= 14, 12.0)
        self.assertEqual(heard(events)[:14], TWINKLE)
        self.assertEqual(sung(events)[:14], TWINKLE_DUET)
        ns = [e["n"] for e in events]
        self.assertEqual(ns, list(range(start_next + 1, start_next + 1 + len(ns))))   # no gaps, no duplicates
        self.assertEqual(p["lost"], 0)
        self.assertTrue(all(e["fake"] and not e["from_host"] for e in events))
        self.assertEqual(events[0]["name"], "C4")
        self.assertEqual(events[0]["t_ms"], 96)
        for k in ("n", "kind", "kind_name", "midi", "name", "vel", "t_ms", "fake", "from_host"):
            self.assertIn(k, events[0])
        for k in STATE_KEYS:
            self.assertIn(k, p["state"], k)
        self.assertTrue(p["state"]["listen"] and p["state"]["fake"] and not p["state"]["sing"])
        self.assertGreater(p["seconds"], 7.0)
        time.sleep(0.25)                      # the periodic LIVE_STATE (every 96 ms) can trail the events
        self.assertGreaterEqual(self.c.poll(p["next"])["state"]["heard_n"], 14)
        # since: asking again from the same point gives the same events; from the end gives none new
        again = self.c.poll(start_next)["events"]
        self.assertEqual([e["n"] for e in again[:len(ns)]], ns)
        self.assertTrue(all(e["n"] > p["next"] for e in self.c.poll(p["next"])["events"]))
        self.assertTrue(self.state()["live"]["running"])

    def test_04_record(self):
        for bad in (0, 31, "x"):
            st, r = self.c.post("/api/live/record", {"seconds": bad})
            self.assertChineseError(st, r)
        st, m = self.c.post("/api/live/record", {"seconds": 2})
        self.assertEqual(st, 200, m)
        self.assertEqual((m["source"], m["mic_source"], m["lost"], m["bit_exact"], m["seconds"]), ("board_mic", "fake", 0, True, 2.0))
        self.assertTrue(m["summary"])
        self.assertIn(m["id"], [c["id"] for c in self.state()["clips"]])
        x, _ = self.app.store.load(m["id"])
        self.assertEqual(len(x), 48000)
        # the stored 24 kHz clip is the 16 kHz mirror, resampled and rounded to 16 bit
        ref = to_mono_24k(fake_audio.samples16(m["first_hop"] * 256, 32000).astype(np.float32) / 32768.0, 16000)
        self.assertLess(float(np.max(np.abs(x - ref))), 1e-4)
        self.assertGreater(float(np.max(np.abs(x))), 0.2)
        st = self.app.link.last_live_state
        self.assertTrue(st["listen"] and not st["monitor"])      # the monitor is off again, listening goes on
        self.assertEqual(len(self.app.link.mic_hops), 0)

    def test_05_set_and_the_scale(self):
        st, r = self.c.post("/api/live/scale")
        self.assertChineseError(st, r, "喇叭")
        st, r = self.c.post("/api/live/set", {"volume": 55})
        self.assertEqual(st, 200, r)
        self.assertEqual((r["volume"], r["sing"], r["running"], r["source"]), (55, False, True, "fake"))
        lk = self.app.link
        self.assertEqual((lk.last_live_state["volume"], lk.last_live_state["listen"], lk.last_live_state["sing"]), (55, True, False))
        for bad in ({"volume": 101}, {"volume": -1}, {"volume": "x"}):
            st, r = self.c.post("/api/live/set", bad)
            self.assertChineseError(st, r)
        st, r = self.c.post("/api/live/set", {"sing": True})
        self.assertEqual(st, 200, r)
        self.assertEqual((r["sing"], r["volume"], r["running"]), (True, 55, True))
        self.assertTrue(lk.last_live_state["sing"] and lk.last_live_state["spk_up"] and lk.last_live_state["listen"])
        since = self.c.poll(0)["next"]
        st, r = self.c.post("/api/live/scale")
        self.assertEqual(st, 200, r)
        self.assertEqual(r["notes"], ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"])
        st, r2 = self.c.post("/api/live/scale")
        self.assertChineseError(st, r2, "还在放")
        self.assertTrue(self.c.poll(since)["scale_busy"])
        events, p = self.c.collect(since, lambda ev: sum(e["from_host"] for e in ev) >= 16, 6.0)
        host = [(e["kind_name"], e["midi"]) for e in events if e["from_host"]]
        want = []
        for m in SCALE:
            want += [("sung_on", m), ("sung_off", m)]
        self.assertEqual(host, want)
        time.sleep(0.2)
        self.assertFalse(self.c.poll(p["next"])["scale_busy"])
        # the speaker test also works without listening
        self.c.post("/api/live/stop")
        st, r = self.c.post("/api/live/set", {"sing": True})
        self.assertEqual((st, r["running"], r["sing"]), (200, False, True))
        since = r["next"]
        self.assertEqual(self.c.post("/api/live/scale")[0], 200)
        events, _ = self.c.collect(since, lambda ev: sum(e["from_host"] for e in ev) >= 16, 6.0)
        self.assertEqual(sum(e["from_host"] for e in events), 16)

    def test_05b_state_after_set_and_stop_is_never_older_than_the_reply(self):
        # periodic LIVE_STATE frames (every 96 ms) queue up while nobody polls; a reply must not be overwritten
        # by one of them that arrived before it
        st, r = self.c.post("/api/live/start", {})
        self.assertEqual(st, 200, r)
        time.sleep(1.5)                                       # ~15 periodic frames queued, not polled
        self.assertGreater(len(self.app.link.live_states), 5)
        st, r = self.c.post("/api/live/set", {"sing": True, "volume": 20})
        self.assertEqual(st, 200, r)
        self.assertTrue(r["state"]["sing"] and r["state"]["listen"], r["state"])
        self.assertEqual(r["state"]["volume"], 20)
        self.assertTrue(self.c.poll(r["next"])["state"]["sing"])
        time.sleep(1.5)                                       # queue more, now with the speaker on
        self.assertGreater(len(self.app.link.live_states), 5)
        st, r = self.c.post("/api/live/stop")
        self.assertEqual(st, 200, r)
        self.assertEqual((r["running"], r["sing"]), (False, False))
        self.assertFalse(r["state"]["listen"] or r["state"]["sing"], r["state"])
        p = self.c.poll(r["next"])
        self.assertFalse(p["state"]["listen"] or p["state"]["sing"], p["state"])
        time.sleep(0.3)
        p = self.c.poll(p["next"])
        self.assertFalse(p["state"]["listen"] or p["state"]["sing"], p["state"])

    def test_05c_record_busy_is_atomic(self):
        st, r = self.c.post("/api/live/start", {})
        self.assertEqual(st, 200, r)
        res = []
        ths = [threading.Thread(target=lambda: res.append(self.c.post("/api/live/record", {"seconds": 1})))
               for _ in range(4)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        self.assertEqual(sorted(code for code, _ in res), [200, 400, 400, 400], res)
        self.assertFalse(self.app.live_rec_busy)

    def test_06_stop_and_start_again(self):
        st, r = self.c.post("/api/live/stop")
        self.assertEqual(st, 200, r)
        self.assertEqual((r["running"], r["sing"]), (False, False))
        lk = self.app.link
        self.assertFalse(lk.last_live_state["listen"] or lk.last_live_state["sing"] or lk.last_live_state["monitor"])
        time.sleep(0.4)
        nxt = self.c.poll(0)["next"]
        time.sleep(0.5)
        self.assertEqual(self.c.poll(nxt)["events"], [])          # nothing more once stopped
        session = r["session"]
        st, r = self.c.post("/api/live/start", {})
        self.assertEqual(st, 200, r)
        self.assertEqual((r["running"], r["sing"], r["volume"], r["session"]), (True, False, 30, session + 1))
        events, p = self.c.collect(r["next"], lambda ev: len(heard(ev)) >= 2, 4.0)
        self.assertEqual(events[0]["t_ms"], 96)                    # the melody starts again from the top
        self.assertEqual(heard(events)[:2], [60, 60])
        self.assertEqual(p["lost"], 0)

    def test_07_source_change_and_other_keys(self):
        st, r = self.c.post("/api/live/start", {"fake": False, "plugin": "echo", "tonic": 7, "bpm": 90})
        self.assertEqual(st, 200, r)
        self.assertEqual((r["source"], r["plugin"], r["tonic"], r["key"], r["bpm"]), ("mic", "echo", 7, "G", 90))
        lk = self.app.link
        self.assertTrue(lk.last_live_state["mic_up"] and not lk.last_live_state["fake"])
        events, _ = self.c.collect(r["next"], lambda ev: False, 1.0)
        self.assertEqual(heard(events), [])                        # the simulator's real microphone is silence
        for bad in ({"plugin": "nope"}, {"tonic": 12}, {"volume": 200}, {"bpm": 0}):
            st, r = self.c.post("/api/live/start", bad)
            self.assertChineseError(st, r)
        st, r = self.c.post("/api/live/start", {"fake": True})
        self.assertEqual((st, r["source"], r["plugin"]), (200, "fake", "third_below"))

    def test_08_concurrent_polls_and_clicks(self):
        errors, stop = [], threading.Event()

        def poller():
            since = 0
            while not stop.is_set():
                st, r = self.s.http.json("GET", f"/api/live/poll?since={since}")
                if st != 200:
                    errors.append(r)
                else:
                    since = r["next"]
                time.sleep(0.01)
        ths = [threading.Thread(target=poller) for _ in range(4)]
        for t in ths:
            t.start()
        try:
            for v in (10, 20, 30, 40, 30):
                st, r = self.c.post("/api/live/set", {"volume": v})
                if st != 200:
                    errors.append(r)
            st, r = self.c.post("/api/live/start", {})
            if st != 200:
                errors.append(r)
        finally:
            stop.set()
            for t in ths:
                t.join()
        self.assertEqual(errors, [])
        events, p = self.c.collect(r["next"], lambda ev: len(heard(ev)) >= 3, 4.0)
        self.assertEqual(heard(events)[:3], [60, 60, 67])
        self.assertEqual(p["lost"], 0)

    def test_09_board_audio_section_points_to_live(self):
        for path, body in (("/api/board_record", {"seconds": 2}), ("/api/board_play", {"clip": "c20990101_000000_dead"})):
            st, r = self.c.post(path, body)
            self.assertEqual(st, 400)
            if path == "/api/board_record":
                self.assertIn("现场", r["error"])

    def test_10_disconnect_while_listening_stops_the_board(self):
        st, r = self.c.post("/api/live/set", {"sing": True})
        self.assertEqual((st, r["running"], r["sing"]), (200, True, True))
        lk = self.app.link
        self.assertTrue(lk.last_live_state["sing"] and lk.last_live_state["listen"])
        self.assertEqual(self.c.post("/api/link", {"target": "none"})[0], 200)
        self.assertFalse(lk.last_live_state["listen"] or lk.last_live_state["sing"])   # its last answer, before close
        s = self.state()
        self.assertEqual((s["live"]["available"], s["live"]["running"], s["live"]["sing"]), (False, False, False))

    def test_11_connecting_elsewhere_stops_and_resets(self):
        self.assertEqual(self.c.post("/api/link", {"target": "sim-live"})[0], 200)
        st, r = self.c.post("/api/live/start", {"sing": True, "volume": 70})
        self.assertEqual((st, r["sing"], r["volume"]), (200, True, 70))
        lk = self.app.link
        self.assertEqual(self.c.post("/api/link", {"target": "sim-live"})[0], 200)     # a new link replaces it
        self.assertFalse(lk.last_live_state["listen"] or lk.last_live_state["sing"])
        live = self.state()["live"]
        self.assertEqual((live["running"], live["sing"], live["volume"], live["source"]), (False, False, 30, "fake"))
        new = self.app.link.last_live_state           # connect told the new link "everything off" once
        self.assertFalse(new["listen"] or new["sing"])


class TestServerShutdown(unittest.TestCase):
    def test_disconnect_at_shutdown_stops_the_board(self):
        # main() ends with app.disconnect(quiet=True) (Ctrl-C or SIGTERM); _Server.stop does the same
        s = test_host._Server().start(self)
        try:
            c = LiveClient(s.http)
            self.assertEqual(c.post("/api/link", {"target": "sim-live"})[0], 200)
            self.assertEqual(c.post("/api/live/start", {"sing": True})[0], 200)
            lk = s.app.link
        finally:
            s.stop()
        self.assertIsNone(s.app.link)
        self.assertFalse(lk.last_live_state["listen"] or lk.last_live_state["sing"])


# ====================================================================== the page, in node
class TestPageInNode(unittest.TestCase):
    """Records /api/state and /api/live/poll from the real server in each situation, then runs page_harness.js."""

    @classmethod
    def setUpClass(cls):
        cls.node = find_node()
        if cls.node is None:
            return
        s = test_host._Server().start(cls)
        h = s.http
        c = LiveClient(h)

        def snap():
            st1, state = h.json("GET", "/api/state")
            p = c.poll(0)
            assert st1 == 200
            return {"state": state, "poll": p}
        fx = {}
        try:
            fx["none"] = snap()
            assert h.json("POST", "/api/link", {"target": "sim"})[0] == 200
            fx["plain_sim"] = snap()
            app = s.app
            with app.link_lock:                    # what "连接" would do against sketch 1 (as test_audio does)
                app.disconnect(quiet=True)
                lk = Link(SimTransport(args=["--no-audio"]))
                app.link, app.link_info = lk, {k: v for k, v in lk.hello().items() if not k.startswith("_")}
            fx["sketch1"] = snap()
            assert h.json("POST", "/api/link", {"target": "sim-live"})[0] == 200
            fx["live_idle"] = snap()
            st, fx["set_sing_reply"] = c.post("/api/live/set", {"sing": True})
            assert st == 200
            c.post("/api/live/set", {"sing": False})
            st, fx["start_reply"] = c.post("/api/live/start", {})
            assert st == 200
            end = time.monotonic() + 8
            while True:                             # a moment where both notes sound, tested on the kept snapshot
                time.sleep(0.15)
                sn = snap()
                p = sn["poll"]
                if (p["now"]["her"] >= 0 and p["now"]["duet"] >= 0 and len(heard(p["events"])) >= 3) \
                        or time.monotonic() > end:
                    break
            fx["listening"] = sn
            st, fx["stop_reply"] = c.post("/api/live/stop")
            time.sleep(0.3)
            fx["stopped"] = snap()
        finally:
            s.stop()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fix = Path(cls.tmp.name) / "fixtures.json"
        cls.fix.write_text(json.dumps(fx, ensure_ascii=False))
        cls.fx = fx

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "tmp", None):
            cls.tmp.cleanup()

    def setUp(self):
        if self.node is None:
            self.skipTest("node not found (looked at /usr/local/bin/node and PATH): the page's script was not run")

    def test_fixtures_are_the_situations(self):
        fx = self.fx
        self.assertIsNone(fx["none"]["state"]["link"])
        self.assertEqual(fx["plain_sim"]["state"]["board_audio"], "standin")
        self.assertIsNone(fx["sketch1"]["state"]["board_audio"])
        self.assertFalse(fx["sketch1"]["state"]["live"]["available"])
        self.assertTrue(fx["live_idle"]["state"]["live"]["available"])
        self.assertFalse(fx["live_idle"]["state"]["live"]["running"])
        self.assertTrue(fx["listening"]["state"]["live"]["running"])
        self.assertGreaterEqual(fx["listening"]["poll"]["now"]["her"], 0)
        self.assertFalse(fx["stopped"]["poll"]["running"])

    def test_script_syntax(self):
        html = INDEX.read_text(encoding="utf-8")
        scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
        self.assertEqual(len(scripts), 1)
        with tempfile.TemporaryDirectory() as d:
            js = Path(d) / "page.js"
            js.write_text(scripts[0], encoding="utf-8")
            r = subprocess.run([self.node, "--check", str(js)], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_page_in_the_four_situations(self):
        r = subprocess.run([self.node, str(HARNESS), str(INDEX), str(self.fix)], capture_output=True, text=True,
                           timeout=120)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            print(out)
        self.assertEqual(r.returncode, 0, "\n".join(l for l in out.splitlines() if not l.startswith("ok ")))
        self.assertRegex(out, r"\d+ passed, 0 failed")


class TestPageStatic(unittest.TestCase):
    """The same safe defaults, read with Python, in case node is missing."""

    def setUp(self):
        self.html = INDEX.read_text(encoding="utf-8")

    def tag(self, id_):
        m = re.search(r'<[^<>]*\bid="' + re.escape(id_) + r'"[^<>]*>', self.html)
        self.assertIsNotNone(m, id_)
        return m.group(0)

    def test_defaults(self):
        self.assertIn(" checked", self.tag("live-src-fake"))
        self.assertNotIn(" checked", self.tag("live-src-mic"))
        self.assertNotIn(" checked", self.tag("live-sing"))
        self.assertNotIn(" checked", self.tag("live-mac-sound"))
        self.assertIn('value="30"', self.tag("live-vol"))

    def test_texts(self):
        for s in ("现场：你唱，Duet 配和声", "连模拟器（现场版）", "/api/live/start", "/api/live/set", "/api/live/stop",
                  "/api/live/scale", "/api/live/poll", "/api/live/record", "第一次这样试", "sh tools/upload_firmware.sh duet_live --yes-upload",
                  "GPIO 11", "GPIO 4", "GPIO 21", "lane-now"):
            self.assertIn(s, self.html)
        self.assertEqual(self.html.count('"auto"'), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
