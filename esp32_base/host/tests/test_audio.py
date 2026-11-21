"""Tests of the stand-in audio on the Mac side: the Python mirror, the AUDIO_STATS codec, recording from and
playing to the simulator, the audio check, and the panel's board-audio endpoints. Standard library unittest.

    .venv/bin/python host/tests/test_audio.py            (from esp32_base/)

Touches no hardware: the link is always the simulator (firmware/host_tests/build/board_sim), never "auto" or a
/dev path; no Mac microphone or speaker is used. Check records and clips go to a temporary directory.
"""
import contextlib
import io
import json
import math
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(HOST / "checks"))

import numpy as np  # noqa: E402

from duet_host import audio_notes, audio_stream, fake_audio, proto, runlog  # noqa: E402
from duet_host.link import Link, LinkError, SimTransport  # noqa: E402
from duet_host.proto import T  # noqa: E402

# The same golden values are asserted in firmware/host_tests/test_core.cpp (C++ side).
LOOP_CRC = 0x1F16E13D
FIRST = [0, 16, 63, 136, 230, 340, 463, 593]
KNOWN = {100: 9219, 1000: -11476, 5000: -333, 12345: -6028, 191999: 0}
TWINKLE = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]


# ====================================================================== 1. the mirror itself
class TestMirror(unittest.TestCase):
    def test_golden_values_shared_with_cpp(self):
        x = fake_audio.samples(0, fake_audio.LOOP)
        self.assertEqual(fake_audio.LOOP, 192000)
        self.assertEqual([int(v) for v in x[:8]], FIRST)
        for n, v in KNOWN.items():
            self.assertEqual(int(x[n]), v, n)
        self.assertEqual(int(x.astype(np.int64).sum()), 15116)
        self.assertEqual(fake_audio.peak(x), 11999)
        self.assertEqual(fake_audio.crc32(x), LOOP_CRC)

    def test_scalar_equals_numpy(self):
        idx = set(range(0, 600)) | set(range(fake_audio.LOOP - 300, fake_audio.LOOP + 300))
        for s, ln, _ in fake_audio.note_table():
            for d in (-1, 0, 1, 119, 120, 121, ln - 241, ln - 240, ln - 239, ln - 1, ln, ln + 1):
                if s + d >= 0:
                    idx.add(s + d)
        rng = np.random.default_rng(1)
        idx |= set(int(v) for v in rng.integers(0, 3 * fake_audio.LOOP, 3000))
        idx = sorted(idx)
        lo, hi = idx[0], idx[-1]
        x = fake_audio.samples(lo, hi - lo + 1)
        bad = [n for n in idx if fake_audio.sample(n) != int(x[n - lo])]
        self.assertEqual(bad, [])

    def test_loop_wraps_and_gaps_are_silent(self):
        x = fake_audio.samples(0, fake_audio.LOOP)
        self.assertTrue(np.array_equal(fake_audio.samples(fake_audio.LOOP, 5000), x[:5000]))
        for s, sound, _ in fake_audio.note_table():
            self.assertTrue(np.all(x[s + sound:s + sound + fake_audio.GAP] == 0))
            self.assertEqual(int(x[s]), 0)
            self.assertNotEqual(int(x[s + 1]), 0)

    def test_phase_increments_are_the_note_frequencies(self):
        for m, inc in fake_audio.INC.items():
            f = 440.0 * 2 ** ((m - 69) / 12)
            self.assertEqual(inc, round(f * 2 ** 32 / 24000), m)
        self.assertEqual([m for m, _ in fake_audio.MELODY], TWINKLE)

    def test_note_finder_hears_two_loops_exactly(self):
        x = fake_audio.samples(0, 2 * fake_audio.LOOP).astype(np.float32) / 32768
        notes = audio_notes.clip_to_notes(x)
        self.assertEqual([n["midi"] for n in notes], TWINKLE * 2)
        want_on = [s / fake_audio.RATE for s, _, _ in fake_audio.note_table()]
        want_on += [t + 8.0 for t in want_on]
        self.assertLessEqual(max(abs(n["on"] - w) for n, w in zip(notes, want_on)), 0.08)

    def test_crc_continues_and_matches_zlib(self):
        s = np.array([0x3231, 0x3433, 0x3635, 0x3837], dtype=np.int16)
        self.assertEqual(fake_audio.crc32(s), 0x9AE0DAAF)            # zlib.crc32(b"12345678")
        self.assertEqual(fake_audio.crc32(s[2:], fake_audio.crc32(s[:2])), fake_audio.crc32(s))
        self.assertEqual(fake_audio.crc32(np.zeros(0, dtype=np.int16)), 0)


# ====================================================================== 2. protocol additions
class TestProto(unittest.TestCase):
    def test_codes_and_lengths(self):
        self.assertEqual((int(T.AUDIO_QUERY), int(T.AUDIO_STATS)), (0x17, 0x27))
        self.assertEqual(proto.LEN_RANGE[T.AUDIO_QUERY], (0, 0))
        self.assertEqual(proto.LEN_RANGE[T.AUDIO_STATS], (48, 48))
        self.assertEqual(struct.calcsize(proto.AUDIO_STATS_FORMAT), 48)
        self.assertEqual(proto.audio_query(3), bytes([0xA5, 0x5A, 0x17, 3, 0, 0, (0x17 + 3) & 0xFF]))

    def test_decode_known_bytes(self):
        p = bytes([2, 0x0D, 3, 1]) + struct.pack("<9I", 0x01020304, 5, 6, 7, 8, 9, 10, 11, 0xDEADBEEF) \
            + struct.pack("<HHI", 11999, 0, 0xA0B0C0D0)
        d = proto.decode_payload(proto.Frame(T.AUDIO_STATS, 0, p))
        self.assertEqual((d["mode"], d["mode_name"], d["spk_level"], d["spk_state"]), (2, "standin", 3, 1))
        self.assertEqual((d["mic_on"], d["spk_on"], d["running"], d["spk_sums"]), (True, False, True, True))
        self.assertEqual((d["mic_sent"], d["mic_dropped"], d["spk_rx"], d["spk_played"], d["spk_silent"]),
                         (0x01020304, 5, 6, 7, 8))
        self.assertEqual((d["spk_underruns"], d["spk_dropped"], d["spk_seq_gaps"], d["spk_crc"], d["spk_peak"],
                          d["dev_us"]), (9, 10, 11, 0xDEADBEEF, 11999, 0xA0B0C0D0))
        self.assertNotIn("reserved", d)

    def test_assemble_counts_gaps(self):
        pl = [fake_audio.frame(k).astype("<i2").tobytes() for k in range(5)]
        got = [(0, pl[0], 0.0), (1, pl[1], 0.01), (3, pl[3], 0.03), (4, pl[4], 0.04)]
        r = audio_stream._assemble(got, 0.05)
        self.assertEqual((r.frames, r.seq_gaps, r.missing, r.first_seq, r.frame_index), (4, 1, 1, 0, [0, 1, 3, 4]))
        self.assertEqual(len(r.samples), 5 * 240)
        self.assertTrue(np.all(r.samples[480:720] == 0))
        self.assertTrue(np.array_equal(r.samples[720:], fake_audio.samples(720, 480)))

    def test_to_int16(self):
        self.assertEqual(audio_stream.to_int16(np.array([0.0, 1.0, -1.0, 2.0])).tolist(), [0, 32767, -32767, 32767])
        self.assertEqual(audio_stream.to_int16(np.array([5, -40000], dtype=np.int32)).tolist(), [5, -32768])


# ====================================================================== 3. the simulator's stand-in audio
class TestSimAudio(unittest.TestCase):
    def setUp(self):
        self.link = Link.open("sim")
        self.addCleanup(self.link.close)

    def test_info_and_idle_stats(self):
        info = self.link.hello()
        self.assertEqual((info["audio"], info["audio_ready"], info["mic"], info["spk"]), ("standin", "0", "0", "0"))
        s = self.link.audio_stats()
        self.assertEqual((s["mode_name"], s["running"], s["spk_sums"]), ("standin", False, True))
        for k in ("mic_sent", "spk_rx", "spk_played", "spk_silent", "spk_underruns", "spk_dropped", "spk_crc"):
            self.assertEqual(s[k], 0, k)

    def test_record_is_bit_exact_and_restarts(self):
        for _ in range(2):                       # the second recording starts at sample 0 and seq 0 again
            r = audio_stream.record_from_board(self.link, 1.0)
            self.assertAlmostEqual(r.frames, 100 * r.seconds, delta=2)
            self.assertEqual((r.first_seq, r.seq_gaps, r.missing), (0, 0, 0))
            self.assertTrue(np.array_equal(r.samples, fake_audio.samples(0, len(r.samples))))
        s = self.link.audio_stats()
        self.assertFalse(s["mic_on"])
        self.assertTrue(s["running"])
        self.assertGreaterEqual(s["mic_sent"], 2 * r.frames - 4)

    def test_play_counts_and_checksum(self):
        x = fake_audio.samples(12345, 24000)     # 1 s from somewhere inside the melody
        p = audio_stream.play_to_board(self.link, x)
        self.assertEqual(p["frames_sent"], 101)
        self.assertEqual((p["frames_played"], p["frames_received"]), (101, 101))
        self.assertEqual((p["underruns_while_sending"], p["dropped"], p["seq_gaps"]), (0, 0, 0))
        self.assertLessEqual(p["underruns_after_drain"], 1)
        self.assertTrue(p["crc_match"])
        self.assertLessEqual(p["level_at_last_send"], 5)       # one frame below the jitter buffer's trim threshold
        self.assertEqual(p["crc_board"], fake_audio.crc32(np.concatenate([np.zeros(240, np.int16), x])))
        self.assertEqual(p["peak_board"], fake_audio.peak(x))
        # A second clip continues the same running checksum.
        y = np.full(500, -7, dtype=np.int16)     # not a multiple of 240: padded with zeros
        p2 = audio_stream.play_to_board(self.link, y)
        self.assertTrue(p2["crc_match"])
        self.assertEqual(p2["frames_played"], 1 + 3)

    def test_speaker_frames_with_the_gate_closed_are_refused(self):
        self.link.audio_ctrl(mic=True, spk=False)       # audio running, speaker gate closed
        time.sleep(0.05)
        self.link.audio_ctrl(mic=False, spk=False)
        before = self.link.audio_stats()
        for k in range(3):
            self.link.send(proto.spk_audio(k, bytes(480)))
        time.sleep(0.1)
        after = self.link.audio_stats()
        self.assertEqual(after["spk_dropped"] - before["spk_dropped"], 3)
        self.assertEqual(after["spk_rx"], before["spk_rx"])

    def test_both_ways_with_notes(self):
        self.link.configure("echo", 0, 4)
        x = fake_audio.samples(0, 2 * 24000)
        out = audio_stream.record_and_play(self.link, x, 2.0, note_every=0.1)
        r, p, notes = out["recording"], out["play"], out["notes"]
        self.assertTrue(np.array_equal(r.samples, fake_audio.samples(0, len(r.samples))))
        self.assertEqual(r.seq_gaps, 0)
        self.assertTrue(p["crc_match"])
        self.assertEqual(p["frames_played"], p["frames_sent"])
        self.assertEqual(p["underruns_while_sending"], 0)
        self.assertGreaterEqual(len(notes), 18)
        self.assertTrue(all("error" not in n and n["midi"] == 60 + (n["i"] % 12) for n in notes))


class TestSimNoAudio(unittest.TestCase):
    """board_sim --no-audio behaves like sketch 1: AUDIO_QUERY gets no answer, INFO has no audio key."""

    def test_no_reply(self):
        link = Link(SimTransport(args=["--no-audio"]))
        self.addCleanup(link.close)
        self.assertNotIn("audio", link.hello())
        with self.assertRaises(LinkError):
            link.audio_stats(timeout=0.3)
        self.assertGreater(link.ping()["rtt_us"], 0)     # the link is still fine


# ====================================================================== 4. the audio check itself
class TestCheckAudio(unittest.TestCase):
    def run_check(self, *argv):
        import check_audio
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        old_runs, old_argv = runlog.RUNS, sys.argv
        runlog.RUNS = Path(tmp.name)
        sys.argv = ["check_audio.py", *argv]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                rc = check_audio.main()
        finally:
            runlog.RUNS, sys.argv = old_runs, old_argv
        lines = (Path(tmp.name) / "checks.jsonl").read_text().splitlines()
        return rc, out.getvalue(), json.loads(lines[-1])

    def test_passes_on_the_simulator(self):
        rc, text, rec = self.run_check("--target", "sim", "--seconds", "2")
        self.assertEqual(rc, 0, text)
        self.assertTrue(rec["passed"], rec.get("failed"))
        self.assertIn("shorter than the 8 s loop", text)
        self.assertEqual(rec["record"]["samples_differ"], 0)
        self.assertTrue(rec["play"]["crc_match"])
        self.assertEqual(rec["duplex"]["answered"], rec["duplex"]["notes"])

    def test_i2s_firmware_skips_the_bit_exact_parts(self):
        # The simulator only has the stand-in. Relabel its INFO and AUDIO_STATS as "i2s" to run the check's
        # real-hardware branch (the counting parts still run; sample and checksum comparisons must be skipped).
        from unittest import mock
        real_hello, real_stats = Link.hello, Link.audio_stats

        def hello(self, *a, **k):
            d = real_hello(self, *a, **k)
            d["audio"] = "i2s"
            return d

        def stats(self, *a, **k):
            d = real_stats(self, *a, **k)
            d.update(mode=1, mode_name="i2s", spk_sums=False)
            return d

        with mock.patch.object(Link, "hello", hello), mock.patch.object(Link, "audio_stats", stats):
            rc, text, rec = self.run_check("--target", "sim", "--seconds", "1")
        self.assertEqual(rc, 0, text)
        self.assertIn("bit-exact parts (samples = mirror, checksum) are skipped", text)
        self.assertNotIn("samples_differ", rec["record"])
        self.assertNotIn("checksum of what was played", text)
        self.assertIn("note finder heard", text)

    def test_skips_firmware_without_audio(self):
        rc, text, rec = self.run_check("--target", "sim", "--sim-args=--no-audio")
        self.assertEqual(rc, 0)
        self.assertIn("skipped: this firmware has no audio", text)
        self.assertEqual(rec["skipped"], "no audio")

    def test_expected_notes_only_whole_ones(self):
        import check_audio
        self.assertEqual(len(check_audio.expected_notes(fake_audio.LOOP)), 14)
        self.assertEqual(len(check_audio.expected_notes(2 * fake_audio.LOOP)), 28)
        self.assertEqual(len(check_audio.expected_notes(11999)), 1)     # 0.45 s note ends at 10800
        self.assertEqual(len(check_audio.expected_notes(10799)), 0)


# ====================================================================== 5. the panel's board-audio endpoints
class TestPanelBoardAudio(unittest.TestCase):
    """One panel server on an ephemeral 127.0.0.1 port, clips in a temp dir; tests run in name order."""

    @classmethod
    def setUpClass(cls):
        import test_host                          # the same helpers the panel tests use (Http, _Server)
        cls.th = test_host
        cls.s = test_host._Server().start(cls)
        cls.h = cls.s.http

    @classmethod
    def tearDownClass(cls):
        cls.s.stop()

    def test_01_needs_a_link(self):
        st, r = self.h.json("POST", "/api/board_record", {"seconds": 2})
        self.assertEqual(st, 400)
        self.assertIn("error", r)
        st, s = self.h.json("GET", "/api/state")
        self.assertIsNone(s["board_audio"])
        self.assertEqual(s["board_audio_busy"], "")

    def test_02_record_from_the_sim(self):
        self.assertEqual(self.h.json("POST", "/api/link", {"target": "sim"})[0], 200)
        st, s = self.h.json("GET", "/api/state")
        self.assertEqual(s["board_audio"], "standin")
        st, m = self.h.json("POST", "/api/board_record", {"seconds": 2})
        self.assertEqual(st, 200, m)
        self.assertEqual((m["source"], m["board_audio"], m["bit_exact"], m["missing"]), ("board_mic", "standin", True, 0))
        self.assertIn("替身", m["name"])
        self.assertAlmostEqual(m["seconds"], m["frames"] * 0.01, places=3)
        self.assertAlmostEqual(m["frames"], 200, delta=4)
        x, _ = self.s.app.store.load(m["id"])
        self.assertEqual([n["midi"] for n in audio_notes.clip_to_notes(x)][:3], [60, 60, 67])
        self.__class__.rec_id = m["id"]
        for bad in (0, 61, "x"):
            st, r = self.h.json("POST", "/api/board_record", {"seconds": bad})
            self.assertEqual(st, 400, bad)

    def test_03_play_a_clip_to_the_sim(self):
        st, r = self.h.json("POST", "/api/board_play", {"clip": self.rec_id})
        self.assertEqual(st, 200, r)
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["crc_match"])
        self.assertEqual(r["frames_played"], r["frames_sent"])
        self.assertEqual(r["underruns_while_sending"], 0)
        self.assertEqual(r["board_audio"], "standin")
        st, r = self.h.json("POST", "/api/board_play", {"clip": "c20990101_000000_dead"})
        self.assertEqual(st, 400)

    def test_04_firmware_without_audio(self):
        app = self.s.app
        with app.link_lock:                       # what "连接" would do against sketch 1
            app.disconnect(quiet=True)
            lk = Link(SimTransport(args=["--no-audio"]))
            app.link, app.link_info = lk, {k: v for k, v in lk.hello().items() if not k.startswith("_")}
        st, s = self.h.json("GET", "/api/state")
        self.assertIsNone(s["board_audio"])
        self.assertIsNotNone(s["link"])
        st, r = self.h.json("POST", "/api/board_record", {"seconds": 2})
        self.assertEqual(st, 400)
        self.assertIn("没有声音", r["error"])
        st, r = self.h.json("POST", "/api/board_play", {"clip": self.rec_id})
        self.assertEqual(st, 400)
        self.assertEqual(self.h.json("POST", "/api/link", {"target": "none"})[0], 200)

    def test_05_busy_is_refused(self):
        self.assertEqual(self.h.json("POST", "/api/link", {"target": "sim"})[0], 200)
        res = {}
        t = threading.Thread(target=lambda: res.update(a=self.h.json("POST", "/api/board_record", {"seconds": 2})))
        t.start()
        time.sleep(0.5)
        st, s = self.h.json("GET", "/api/state")
        self.assertEqual(s["board_audio_busy"], "录音")
        st, r = self.h.json("POST", "/api/board_record", {"seconds": 2})     # a second one is refused, not queued
        self.assertEqual(st, 400)
        self.assertIn("正在录音", r["error"])
        st, r = self.h.json("POST", "/api/board_play", {"clip": self.rec_id})
        self.assertEqual(st, 400)
        t.join()
        self.assertEqual(res["a"][0], 200)
        self.assertEqual(self.h.json("GET", "/api/state")[1]["board_audio_busy"], "")
        self.assertEqual(self.h.json("POST", "/api/link", {"target": "none"})[0], 200)


class TestPageBoardAudio(unittest.TestCase):
    def test_page_has_the_board_audio_section(self):
        html = (HOST / "duet_host" / "web" / "index.html").read_text(encoding="utf-8")
        for s in ("/api/board_record", "/api/board_play", "board_audio", "board_audio_busy", "ba-standin",
                  "替身", "板子的声音", "board_mic"):
            self.assertIn(s, html)
        self.assertEqual(html.count('"auto"'), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
