"""Tests of the live chain on the Mac side (sketch duet_live): the new frames, the 16 kHz fake-microphone mirror,
the Link methods against the simulator in live mode (board_sim --live, target "sim-live"), and check_live's skip.
Standard library unittest.

    .venv/bin/python host/tests/test_live.py            (from esp32_base/)

Touches no hardware: the link is always the simulator, never "auto" or a /dev path; no Mac microphone or speaker
is used. Check records go to a temporary directory.
"""
import contextlib
import io
import json
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(HOST / "checks"))

import numpy as np  # noqa: E402

from duet_host import fake_audio, proto, runlog  # noqa: E402
from duet_host.link import Link, LinkError  # noqa: E402
from duet_host.proto import T  # noqa: E402

# The same golden values are asserted in firmware/host_tests/test_live.cpp (C++ side).
LOOP16_CRC = 0xA35C56ED
FIRST16 = [0, 36, 136, 283, 463, 660, 858, 1042]
KNOWN16 = {100: -3973, 1000: 3859, 5000: -5844, 12345: 7408, 127999: 0}
TWINKLE = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]


# ====================================================================== 1. the new frames
class TestLiveProto(unittest.TestCase):
    def test_codes_and_lengths(self):
        self.assertEqual((T.LIVE_CTRL, T.SYNTH_NOTE, T.LIVE_NOTE, T.LIVE_STATE, T.MIC_HOP),
                         (0x18, 0x19, 0x28, 0x29, 0x2A))
        for t, n in ((T.LIVE_CTRL, 4), (T.SYNTH_NOTE, 2), (T.LIVE_NOTE, 8), (T.LIVE_STATE, 24), (T.MIC_HOP, 512)):
            self.assertEqual(proto.LEN_RANGE[t], (n, n))
        self.assertEqual(proto.LEN_RANGE[T.MIC_AUDIO], (480, 480))      # the old ones are untouched
        self.assertEqual(proto.LEN_RANGE[T.AUDIO_STATS], (48, 48))

    def test_builders(self):
        fr = proto.live_ctrl(7, listen=True, sing=False, fake=True, monitor=True, volume=150, bpm=90)
        self.assertEqual(fr[2:7], bytes([0x18, 7, 4, 0, 0x0D]))
        self.assertEqual(fr[7:10], bytes([150, 90, 0]))                  # > 100 is clamped by the device
        fr = proto.synth_note(3, 69, 200)
        self.assertEqual(fr[6:8], bytes([69, 127]))
        got = proto.Parser().feed(proto.live_ctrl(1, True, True, False, False) + proto.synth_note(2, 60, 0))
        self.assertEqual([(g.type, g.seq) for g in got], [(T.LIVE_CTRL, 1), (T.SYNTH_NOTE, 2)])

    def test_decoders(self):
        n = proto.decode_live_note(bytes([3, 57, 99, 1, 4, 3, 2, 1]))
        self.assertEqual((n["kind"], n["kind_name"], n["midi"], n["vel"], n["fake"], n["from_host"], n["t_ms"]),
                         (3, "sung_on", 57, 99, True, False, 0x01020304))
        p = struct.pack("<BBbbfBbbBIHHI", 0x81 | 0x20, 100, -1, 57, 261.5, 97, -14, -50, 0, 123456, 14, 13, 0xA0B0C0D0)
        s = proto.decode_live_state(p)
        self.assertEqual((s["listen"], s["spk_up"], s["reply"], s["sing"], s["error"]), (True, True, True, False, False))
        self.assertEqual((s["heard"], s["sung"], s["freq"], s["conf"], s["level"], s["cents"]), (-1, 57, 261.5, 97, -14, -50))
        self.assertEqual((s["hops"], s["heard_n"], s["sung_n"], s["spk_blocks"]), (123456, 14, 13, 0xA0B0C0D0))
        self.assertNotIn("reserved", s)
        hop = proto.decode_mic_hop(struct.pack("<256h", *range(-128, 128)))
        self.assertEqual((len(hop), hop[0], hop[-1]), (256, -128, 127))
        fr = proto.Frame(int(T.LIVE_NOTE), 0, bytes(8))
        self.assertEqual(proto.decode_payload(fr)["kind"], 0)


# ====================================================================== 2. the 16 kHz mirror
class TestMirror16(unittest.TestCase):
    def test_golden_values_shared_with_cpp(self):
        x = fake_audio.samples16(0, fake_audio.LOOP16)
        self.assertEqual(fake_audio.LOOP16, 128000)
        self.assertEqual([int(v) for v in x[:8]], FIRST16)
        for n, v in KNOWN16.items():
            self.assertEqual(int(x[n]), v, n)
        self.assertEqual(int(x.astype(np.int64).sum()), 37410)
        self.assertEqual(fake_audio.peak(x), 11999)
        self.assertEqual(fake_audio.crc32(x), LOOP16_CRC)

    def test_scalar_equals_numpy(self):
        idx = set(range(0, 400)) | set(range(fake_audio.LOOP16 - 300, fake_audio.LOOP16 + 300))
        for s, ln, _ in fake_audio.note_table16():
            for d in (-1, 0, 1, 79, 80, 81, ln - 161, ln - 160, ln - 159, ln - 1, ln, ln + 1):
                if s + d >= 0:
                    idx.add(s + d)
        rng = np.random.default_rng(2)
        idx |= set(int(v) for v in rng.integers(0, 3 * fake_audio.LOOP16, 2000))
        idx = sorted(idx)
        x = fake_audio.samples16(idx[0], idx[-1] - idx[0] + 1)
        for n in idx:
            self.assertEqual(int(x[n - idx[0]]), fake_audio.sample16(n), n)

    def test_melody_gaps_and_the_50ms_variant(self):
        x = fake_audio.samples16(0, fake_audio.LOOP16)
        for s, sound, _ in fake_audio.note_table16():
            self.assertTrue(np.all(x[s + sound: s + sound + fake_audio.GAP16] == 0))
            self.assertGreater(int(np.max(np.abs(x[s: s + sound].astype(np.int32)))), 10000)
        self.assertEqual([m for _, _, m in fake_audio.note_table16()], TWINKLE)
        y = fake_audio.samples16(0, fake_audio.LOOP16, gap=fake_audio.GAP16_50MS)
        self.assertTrue(all(fake_audio.sample16(n, fake_audio.GAP16_50MS) == int(y[n]) for n in range(0, 128000, 997)))
        self.assertEqual(fake_audio.hop16(3).tolist(), x[768:1024].tolist())

    def test_the_24khz_mirror_is_unchanged(self):
        self.assertEqual(fake_audio.crc32(fake_audio.samples(0, fake_audio.LOOP)), 0x1F16E13D)
        self.assertEqual((fake_audio.RATE, fake_audio.GAP, fake_audio.LOOP), (24000, 1200, 192000))


# ====================================================================== 3. the link to the simulator as duet_live
class TestLinkSimLive(unittest.TestCase):
    def setUp(self):
        self.link = Link.open("sim-live")
        self.addCleanup(self.link.close)

    def test_info(self):
        info = self.link.hello()
        self.assertEqual((self.link.kind, self.link.port), ("sim", "sim-live"))
        for k, v in (("fw", "duet_live"), ("ver", "0.3.0"), ("proto", "2"), ("live", "1"), ("mic_rate", "16000"),
                     ("spk_rate", "22050"), ("mic_pins", "11/12/13"), ("spk_pins", "4/5/6"), ("sim", "1")):
            self.assertEqual(info.get(k), v, k)
        self.assertNotIn("audio", info)              # so check_audio skips this firmware

    def test_live_ctrl_reply_and_gates(self):
        st = self.link.live_ctrl()
        self.assertTrue(st["reply"])
        self.assertFalse(st["listen"] or st["sing"] or st["mic_up"] or st["spk_up"])
        st = self.link.live_ctrl(listen=True, sing=True, fake=False, volume=40, bpm=0)
        self.assertTrue(st["listen"] and st["sing"] and st["mic_up"] and st["spk_up"] and not st["fake"])
        self.assertEqual(st["volume"], 40)
        st = self.link.live_ctrl(listen=False, volume=250)
        self.assertEqual(st["volume"], 100)

    def test_periodic_states_never_answer_a_request(self):
        # Periodic LIVE_STATE frames have their own seq (from 0 at every LISTEN edge); a LIVE_CTRL waiting for its
        # reply must only take the frame with LS_REPLY.
        self.link.live_ctrl(listen=True, fake=True)
        for _ in range(60):
            st = self.link.live_ctrl(listen=True, fake=True)
            self.assertTrue(st["reply"])
            time.sleep(0.01)
        states = list(self.link.live_states)
        self.assertGreater(len(states), 5)
        self.assertTrue(all(not s["reply"] for s in states))

    def test_melody_notes_and_drain(self):
        self.link.configure("third_below", 0, 4)
        self.link.live_ctrl(listen=True, fake=True)
        time.sleep(1.1)                              # 68 hops: notes 1, 2 and the start of 3
        got = self.link.live_drain()
        notes = got["notes"]
        self.assertEqual([(n["kind_name"], n["midi"], n["t_ms"]) for n in notes[:6]],
                         [("heard_on", 60, 96), ("sung_on", 57, 96), ("heard_off", 60, 464), ("sung_off", 57, 464),
                          ("heard_on", 60, 592), ("sung_on", 57, 592)])
        self.assertEqual([n["seq"] for n in notes], list(range(len(notes))))
        self.assertTrue(all(n["fake"] for n in notes))
        self.assertGreaterEqual(len(got["states"]), 10)
        self.assertEqual(len(self.link.live_notes), 0)   # drained

    def test_monitor_hops_equal_the_mirror(self):
        self.link.live_ctrl(listen=True, fake=True, monitor=True)
        time.sleep(0.6)
        self.link.live_ctrl(listen=False)
        hops = self.link.live_drain()["hops"]
        self.assertGreater(len(hops), 30)
        self.assertEqual([h["seq"] for h in hops], list(range(len(hops))))
        x = np.frombuffer(b"".join(h["pcm"] for h in hops), dtype="<i2")
        self.assertTrue(np.array_equal(x, fake_audio.samples16(0, len(x))))

    def test_synth_note_needs_sing(self):
        self.link.live_ctrl(sing=False)
        self.link.synth_note(69, 100)
        time.sleep(0.15)
        self.assertEqual(len(self.link.live_notes), 0)
        st0 = self.link.live_ctrl(sing=True)
        self.link.synth_note(69, 100)
        time.sleep(0.4)
        self.link.synth_note(69, 0)
        st1 = self.link.live_ctrl(sing=True)
        ev = self.link.live_drain()["notes"]
        self.assertEqual([(n["kind_name"], n["midi"], n["from_host"], n["t_ms"]) for n in ev],
                         [("sung_on", 69, True, 0), ("sung_off", 69, True, 0)])
        self.assertGreater(st1["spk_blocks"], st0["spk_blocks"] + 20)
        self.assertGreater(int(self.link.hello()["sink_peak"]), 1000)
        self.assertEqual(st1["sung_n"], 0)            # host notes are not plug-in notes

    def test_ping_and_notes_still_work(self):
        self.link.configure("third_below", 0, 4)
        self.link.live_ctrl(listen=True, fake=True, monitor=True, sing=True)
        for _ in range(20):
            self.assertGreater(self.link.ping()["rtt_us"], 0)
        self.assertEqual(self.link.note(0, 64, 0.0, None)["midi"], 60)


class TestOldSimUnchanged(unittest.TestCase):
    def test_plain_sim_has_no_live_chain(self):
        link = Link.open("sim")
        self.addCleanup(link.close)
        info = link.hello()
        self.assertEqual(info["fw"], "board_sim")
        self.assertNotIn("live", info)
        self.assertEqual(link.port, "sim")
        with self.assertRaises(LinkError):
            link.live_ctrl(listen=True, timeout=0.3)
        self.assertGreater(link.ping()["rtt_us"], 0)


# ====================================================================== 4. the live check's skip
class TestCheckLive(unittest.TestCase):
    def run_check(self, *argv):
        import check_live
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        old_runs, old_argv = runlog.RUNS, sys.argv
        runlog.RUNS = Path(tmp.name)
        sys.argv = ["check_live.py", *argv]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                rc = check_live.main()
        finally:
            runlog.RUNS, sys.argv = old_runs, old_argv
        lines = (Path(tmp.name) / "checks.jsonl").read_text().splitlines()
        return rc, out.getvalue(), json.loads(lines[-1])

    def test_skips_firmware_without_live(self):
        rc, text, rec = self.run_check("--target", "sim")
        self.assertEqual(rc, 0, text)
        self.assertIn("skipped: this firmware has no live chain", text)
        self.assertEqual(rec["skipped"], "no live chain")

    def test_expected_list_matches_reference(self):
        import check_live
        notes = [{"kind": k, "midi": m, "t_ms": t} for k, m, t in check_live.EXPECTED]
        heard = [m for k, m, _ in check_live.EXPECTED if k == proto.LN_HEARD_ON]
        sung = [m for k, m, _ in check_live.EXPECTED if k == proto.LN_SUNG_ON]
        self.assertEqual(heard, TWINKLE + [60])
        self.assertEqual(check_live.reference_sung(notes), sung)


if __name__ == "__main__":
    unittest.main(verbosity=2)
