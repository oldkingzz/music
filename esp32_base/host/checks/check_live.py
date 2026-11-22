"""Check 4, the live chain (sketch duet_live): does the board hear the melody, sing the plug-in's notes, and keep
the link answering while it does?

    .venv/bin/python host/checks/check_live.py --target sim-live          (the Mac simulator as duet_live)
    .venv/bin/python host/checks/check_live.py --target auto              (the board: fake melody, speaker OFF)
    .venv/bin/python host/checks/check_live.py --target auto --speaker    (also turns the speaker on: it sounds)
    .venv/bin/python host/checks/check_live.py --target auto --real-mic   (the real microphone instead of the fake)

By default it uses the board's built-in FAKE microphone (a known melody generated on the board) with the speaker
OFF: no microphone I2S and no speaker I2S is started unless --real-mic / --speaker is given. In the simulator the
speaker is a counting sink, so the speaker part always runs there. A firmware whose INFO has no live=1 (sketch 1,
sketch 2, the plain simulator) -> "skipped", exit 0.

What it checks, in order:
  a. INFO: fw, protocol 2, live=1, the rates and pins of the team's wiring; the gates start off.
  b. CONFIG third_below in C, LISTEN + FAKE for one 8 s loop and a bit: the heard notes are the 14 melody notes
     in order (plus the first of the next loop), each sung note equals the Python reference of the plug-in for
     that heard note, every event equals the pinned list (EXPECTED; t_ms exactly in the simulator, within 2 hops
     = 32 ms on a board), no lost event (seq), LIVE_STATE every 6 hops at a 96 ms pace.
  c. MONITOR: the MIC_HOP samples equal the Python mirror of the fake melody exactly, no hop lost.
  d. Speaker (simulator always, board only with --speaker): SING on -> speaker up, its block counter runs at
     22050 / 256 = 86.1 blocks/s, SYNTH_NOTE plays (SUNG events with from_host), the simulator's sink made sound.
  e. LISTEN off / on and FAKE off / on (FAKE off = real mic: simulator or --real-mic only) work without a reconnect:
     sounding notes end, seq and t_ms restart at 0, the melody starts again.
  f. PING and NOTE_IN still answer while the chain runs; round trips are reported.
Exit code 0 = all passed (or skipped). One line is appended to runs/checks.jsonl.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duet_host import fake_audio, harmonizers, proto, runlog  # noqa: E402
from duet_host.link import Link, LinkError  # noqa: E402

MELODY = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]
HOPS = 530                       # one 8 s loop (500 hops) + the first note of the next loop
# The LIVE_NOTE events of those 530 hops with third_below in C: (kind, midi, t_ms); kind 1 heard_on, 2 heard_off,
# 3 sung_on, 4 sung_off. Taken from the lifted tracker and pinned; the same list is PIN_CHAIN in
# firmware/host_tests/test_live.cpp.
EXPECTED = [
    (1, 60, 96), (3, 57, 96), (2, 60, 464), (4, 57, 464), (1, 60, 592), (3, 57, 592), (2, 60, 976), (4, 57, 976),
    (1, 67, 1088), (3, 64, 1088), (2, 67, 1472), (4, 64, 1472), (1, 67, 1584), (3, 64, 1584), (2, 67, 1968),
    (4, 64, 1968), (1, 69, 2096), (3, 65, 2096), (2, 69, 2480), (4, 65, 2480), (1, 69, 2592), (3, 65, 2592),
    (2, 69, 2976), (4, 65, 2976), (1, 67, 3088), (3, 64, 3088), (2, 67, 3968), (4, 64, 3968), (1, 65, 4096),
    (3, 62, 4096), (2, 65, 4464), (4, 62, 4464), (1, 65, 4592), (3, 62, 4592), (2, 65, 4976), (4, 62, 4976),
    (1, 64, 5088), (3, 60, 5088), (2, 64, 5472), (4, 60, 5472), (1, 64, 5584), (3, 60, 5584), (2, 64, 5968),
    (4, 60, 5968), (1, 62, 6096), (3, 59, 6096), (2, 62, 6464), (4, 59, 6464), (1, 62, 6592), (3, 59, 6592),
    (2, 62, 6976), (4, 59, 6976), (1, 60, 7088), (3, 57, 7088), (2, 60, 7968), (4, 57, 7968), (1, 60, 8096),
    (3, 57, 8096), (2, 60, 8464), (4, 57, 8464),
]
BLOCKS_PER_S = 22050 / 256       # the speaker task's pace


def seq_gaps(seqs):
    """Number of missing frames in a run of u8 seq numbers."""
    return sum(((b - a) & 0xFF) - 1 for a, b in zip(seqs, seqs[1:]))


def reference_sung(notes, bpm=120):
    """The Python reference of third_below in C on the heard notes, with onset / prev_dur in beats as the device
    computes them. Returns the list of sung notes (one per heard_on)."""
    ref = harmonizers.make("third_below")
    ref.start({"tonic": 0, "M": 4, "K": 0, "pos": []})
    out, i, on_ms, prev = [], 0, None, None
    for n in notes:
        if n["kind"] == proto.LN_HEARD_OFF and on_ms is not None:
            prev = (n["t_ms"] - on_ms) / 1000 * bpm / 60
        if n["kind"] == proto.LN_HEARD_ON:
            onset = n["t_ms"] / 1000 * bpm / 60
            out.append(ref.step(i, proto.f32(onset), n["midi"], None if prev is None else proto.f32(prev)))
            on_ms = n["t_ms"]
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sim-live", help="sim-live, sim, auto, or a port such as /dev/cu.usbmodem101")
    ap.add_argument("--real-mic", action="store_true", help="use the board's real microphone (starts its I2S)")
    ap.add_argument("--speaker", action="store_true", help="turn the board's speaker on (it sounds)")
    ap.add_argument("--seconds", type=float, default=HOPS * 0.016 + 0.25, help="length of part b")
    a = ap.parse_args()

    fails, rec = [], {"check": "live", "target": a.target, "real_mic": a.real_mic, "speaker": a.speaker}

    def need(ok, what):
        print(("  ok    " if ok else "  FAIL  ") + what)
        if not ok:
            fails.append(what)
        return ok

    def finish(skipped=None):
        rec["passed"], rec["failed"] = not fails, fails
        if skipped:
            rec["skipped"] = skipped
        path = runlog.append("checks", rec)
        print(("PASS" if not fails else f"FAIL ({len(fails)})") + (f" (skipped: {skipped})" if skipped else "")
              + f"  -> {path}")
        return 0 if not fails else 1

    try:
        link = Link.open(a.target)
    except LinkError as e:
        print(f"cannot open: {e}")
        return 2
    try:
        rec["kind"], rec["port"] = link.kind, link.port
        board = link.kind == "board"
        fake = not a.real_mic
        speaker = a.speaker or not board
        tol = 32 if board else 0
        print(f"connected to {link.kind} at {link.port}; microphone: {'FAKE melody' if fake else 'REAL'}, "
              f"speaker: {'ON' if speaker else 'off'}{' (simulator sink)' if speaker and not board else ''}")

        # ---------------------------------------------------------------- a. INFO
        info = link.hello()
        rec["info"] = {k: v for k, v in info.items() if not k.startswith("_")}
        print(f"  INFO  fw={info.get('fw')} ver={info.get('ver')} live={info.get('live')}")
        if info.get("live") != "1":
            print("  skipped: this firmware has no live chain (INFO has no live=1; upload duet_live to check it)")
            return finish("no live chain")
        need(info.get("proto") == str(proto.PROTO_VERSION), f"protocol version {info.get('proto')}")
        need(info.get("fw") == "duet_live", f"firmware name {info.get('fw')}")
        need(info.get("mic_rate") == "16000" and info.get("spk_rate") == "22050",
             f"rates mic {info.get('mic_rate')} Hz, speaker {info.get('spk_rate')} Hz")
        need(info.get("mic_pins") == "11/12/13" and info.get("spk_pins") == "4/5/6",
             f"pins mic {info.get('mic_pins')} (BCLK/WS/SD), speaker {info.get('spk_pins')} (BCLK/LRC/DIN)")
        if board:
            need(int(info.get("flash", 0)) == 16 * 1024 * 1024, f"flash is 16 MB ({info.get('flash')})")
        st = link.live_ctrl(listen=False, sing=False, fake=True)
        need(st["reply"] and not (st["listen"] or st["sing"] or st["mic_up"] or st["spk_up"] or st["error"]),
             f"all gates off (flags 0x{st['flags']:02x}), reply rtt {st['rtt_us']:.0f} us")

        # ---------------------------------------------------------------- b. the melody through the chain
        link.configure("third_below", 0, 4)
        link.live_clear()
        st = link.live_ctrl(listen=True, sing=False, fake=fake)
        need(st["listen"] and st["fake"] == fake and not st["error"],
             f"LISTEN on (flags 0x{st['flags']:02x}, mic_up {st['mic_up']})")
        time.sleep(a.seconds)
        got = link.live_drain()
        notes, states = got["notes"], got["states"]
        rec["b"] = {"notes": len(notes), "states": len(states)}
        print(f"  {len(notes)} events and {len(states)} LIVE_STATE frames in {a.seconds:.2f} s")
        need(seq_gaps([n["seq"] for n in notes]) == 0 and (not notes or notes[0]["seq"] == 0),
             f"no lost event (seq from {notes[0]['seq'] if notes else '-'}, {seq_gaps([n['seq'] for n in notes])} gaps)")
        heard = [n["midi"] for n in notes if n["kind"] == proto.LN_HEARD_ON]
        sung = [n["midi"] for n in notes if n["kind"] == proto.LN_SUNG_ON]
        rec["b"]["heard"], rec["b"]["sung"] = heard, sung
        print("  heard " + " ".join(map(str, heard)))
        print("  sung  " + " ".join(map(str, sung)))
        if fake:
            need(heard[:14] == MELODY, "the heard notes are the 14 melody notes in order")
            need(sung == reference_sung(notes), "each sung note = Python third_below of the heard note")
            need(all(n["fake"] and not n["from_host"] for n in notes), "every event is flagged fake")
            inside = [n for n in notes if n["t_ms"] <= HOPS * 16 + tol]   # the next real event is at 8592 ms
            kinds_ok = [(n["kind"], n["midi"]) for n in inside] == [(k, m) for k, m, _ in EXPECTED]
            worst = max((abs(n["t_ms"] - t) for n, (_, _, t) in zip(inside, EXPECTED)), default=99999)
            rec["b"]["worst_t_ms"] = worst
            need(kinds_ok and worst <= tol,
                 f"{len(inside)} events = the pinned list ({len(EXPECTED)}), times off by at most {worst} ms "
                 f"(allowed {tol})")
        else:
            print("  NOTE  real microphone: the notes are not compared with anything")
        hops = [s["hops"] for s in states]
        n_exp = int(states[-1]["hops"] / proto.LIVE_STATE_EVERY) if states else 0
        need(len(states) >= 80 and seq_gaps([s["seq"] for s in states]) == 0 and
             hops == [proto.LIVE_STATE_EVERY * (k + 1) for k in range(len(hops))],
             f"LIVE_STATE every 6 hops: {len(states)} frames, hops {hops[:2]}..{hops[-1:]} ({n_exp} expected)")
        if len(states) > 10:
            per = (states[-1]["rx_time"] - states[0]["rx_time"]) / (len(states) - 1) * 1000
            rec["b"]["state_period_ms"] = round(per, 2)
            need(abs(per - 96) <= 96 * 0.05, f"LIVE_STATE pace {per:.2f} ms (want 96 +/- 5 %)")

        # ---------------------------------------------------------------- c. monitor
        link.live_ctrl(listen=False, fake=fake)
        link.live_clear()
        st = link.live_ctrl(listen=True, fake=fake, monitor=True)
        time.sleep(2.0)
        hops_rx = link.live_drain()["hops"]
        link.live_ctrl(listen=True, fake=fake, monitor=False)
        seqs = [h["seq"] for h in hops_rx]
        need(len(hops_rx) >= 100 and seqs[0] == 0 and seq_gaps(seqs) == 0,
             f"{len(hops_rx)} MIC_HOP frames in 2 s, seq from {seqs[0] if seqs else '-'}, {seq_gaps(seqs)} lost")
        if fake and hops_rx:
            x = np.frombuffer(b"".join(h["pcm"] for h in hops_rx), dtype="<i2")
            bad = int(np.count_nonzero(x != fake_audio.samples16(0, len(x))))
            rec["c"] = {"hops": len(hops_rx), "samples_differ": bad}
            need(bad == 0, f"all {len(x)} samples equal the Python mirror ({bad} differ)")
        elif hops_rx:
            x = np.frombuffer(b"".join(h["pcm"] for h in hops_rx), dtype="<i2").astype(np.float64)
            print(f"  NOTE  real microphone: rms {np.sqrt(np.mean(x ** 2)):.0f}, peak {np.max(np.abs(x)):.0f} (int16)")

        # ---------------------------------------------------------------- d. speaker
        if speaker:
            link.live_ctrl(listen=False, fake=fake)
            link.live_clear()
            st0 = link.live_ctrl(sing=True, volume=60)
            t0 = time.perf_counter()
            need(st0["sing"] and st0["spk_up"] and not st0["error"], f"SING on: speaker up (flags 0x{st0['flags']:02x})")
            time.sleep(1.0)
            st1 = link.live_ctrl(sing=True, volume=60)
            t1 = time.perf_counter()
            rate = (st1["spk_blocks"] - st0["spk_blocks"]) / (t1 - t0)
            rec["d"] = {"blocks_per_s": round(rate, 2)}
            need(abs(rate - BLOCKS_PER_S) <= BLOCKS_PER_S * 0.05,
                 f"speaker blocks {rate:.1f} per second (want {BLOCKS_PER_S:.1f} +/- 5 %)")
            link.synth_note(69, 100)
            time.sleep(0.3)
            link.synth_note(69, 0)
            time.sleep(0.2)
            hn = [n for n in link.live_drain()["notes"] if n["from_host"]]
            need([(n["kind"], n["midi"], n["vel"]) for n in hn] == [(proto.LN_SUNG_ON, 69, 100), (proto.LN_SUNG_OFF, 69, 0)],
                 f"SYNTH_NOTE 69 on / off reported as from_host sung_on / sung_off ({len(hn)} events)")
            if not board:
                i2 = link.hello()
                peak = int(i2.get("sink_peak", 0))
                rec["d"]["sink_peak"], rec["d"]["sink_blocks"] = peak, int(i2.get("sink_blocks", 0))
                need(0 < peak <= 32767 and int(i2.get("sink_blocks", 0)) >= st1["spk_blocks"],
                     f"the simulator's speaker sink got sound: peak {peak}, {i2.get('sink_blocks')} blocks, "
                     f"crc {i2.get('sink_crc')}")
            # Duet's notes from the plug-in go to the speaker too
            link.live_clear()
            link.live_ctrl(listen=True, sing=True, fake=fake, volume=60)
            time.sleep(1.2)
            st2 = link.live_ctrl(listen=True, sing=True, fake=fake, volume=60)
            sung2 = [n for n in link.live_drain()["notes"] if n["kind"] == proto.LN_SUNG_ON]
            need((not fake or len(sung2) >= 2) and st2["spk_blocks"] > st1["spk_blocks"],
                 f"LISTEN + SING: {len(sung2)} sung notes, speaker blocks {st1['spk_blocks']} -> {st2['spk_blocks']}")
            st3 = link.live_ctrl(listen=False, sing=False, fake=True)
            need(not st3["spk_up"] and not st3["sing"], f"SING off: speaker down (flags 0x{st3['flags']:02x})")
        else:
            print("  skipped d: the speaker stays off on a board unless --speaker is given")

        # ---------------------------------------------------------------- e. off / on without a reconnect
        link.live_ctrl(listen=False, fake=fake)
        link.live_clear()
        link.live_ctrl(listen=True, fake=fake)
        time.sleep(0.25)                                  # the first note is on at 96 ms
        st = link.live_ctrl(listen=False, fake=fake)
        ev = link.live_drain()["notes"]
        need(not st["listen"] and st["heard"] == -1 and st["sung"] == -1,
             f"LISTEN off: heard {st['heard']}, sung {st['sung']}")
        if fake:
            need([(n["kind"], n["midi"]) for n in ev] == [(1, 60), (3, 57), (2, 60), (4, 57)],
                 f"the sounding note ended at LISTEN off: {[(n['kind_name'], n['midi'], n['t_ms']) for n in ev]}")
        link.live_ctrl(listen=True, fake=fake)
        time.sleep(0.25)
        ev = link.live_drain()["notes"]
        if fake:
            need(len(ev) >= 2 and ev[0]["seq"] == 0 and (ev[0]["kind"], ev[0]["midi"]) == (1, 60) and
                 abs(ev[0]["t_ms"] - EXPECTED[0][2]) <= tol,
                 f"LISTEN on again: seq {ev[0]['seq'] if ev else '-'}, first event "
                 f"{(ev[0]['kind_name'], ev[0]['midi'], ev[0]['t_ms']) if ev else '-'}")
        if not board or a.real_mic:
            st = link.live_ctrl(listen=True, fake=False)   # the real microphone (simulator: not wired = silence)
            time.sleep(0.6)
            ev = link.live_drain()["notes"]
            need(st["mic_up"] and not st["fake"] and not st["error"], f"FAKE off: microphone up (flags 0x{st['flags']:02x})")
            if not board:
                after = [n for n in ev if not n["fake"]]
                need(not after, f"simulator 'real microphone' is silent: {len(after)} notes")
            st = link.live_ctrl(listen=True, fake=True)
            time.sleep(0.25)
            ev = [n for n in link.live_drain()["notes"] if n["fake"]]
            need(not st["mic_up"] and ev and ev[0]["seq"] == 0 and (ev[0]["kind"], ev[0]["midi"]) == (1, 60),
                 "FAKE on again: microphone down, the melody restarts at seq 0")
        else:
            print("  skipped e (FAKE off): that would start the board's real microphone; give --real-mic")

        # ---------------------------------------------------------------- f. the old frames while the chain runs
        link.live_ctrl(listen=True, sing=speaker, fake=fake, volume=60)
        rtts = [link.ping()["rtt_us"] for _ in range(100)]
        nrt = []
        ref = harmonizers.make("third_below")
        ref.start({"tonic": 0, "M": 4, "K": 0, "pos": []})
        note_ok = True
        for i in range(50):
            midi = 55 + i % 20
            r = link.note(i, midi, i * 0.5, None if i == 0 else 0.5)
            nrt.append(r["rtt_us"])
            note_ok &= r["midi"] == ref.step(i, i * 0.5, midi, None if i == 0 else 0.5)
        rec["f"] = {"ping_us": {"median": runlog.pct(rtts, 50), "p95": runlog.pct(rtts, 95), "max": max(rtts)},
                    "note_us": {"median": runlog.pct(nrt, 50), "p95": runlog.pct(nrt, 95), "max": max(nrt)}}
        print(f"  while the chain runs: PING round trip median {runlog.pct(rtts, 50):.0f} us, 95 % "
              f"{runlog.pct(rtts, 95):.0f} us, max {max(rtts):.0f} us; NOTE_IN median {runlog.pct(nrt, 50):.0f} us, "
              f"max {max(nrt):.0f} us")
        need(len(rtts) == 100 and len(nrt) == 50 and note_ok, "100 PINGs and 50 NOTE_INs answered, notes = Python")
        st = link.live_ctrl(listen=False, sing=False, fake=True)
        need(not (st["listen"] or st["sing"] or st["mic_up"] or st["spk_up"]), "everything off at the end")

        s = link.stats()
        rec["stats"] = s
        need(s["bad_checksum"] == 0, "no frame from the device had a bad checksum")
    except LinkError as e:
        need(False, f"link error: {e}")
    finally:
        try:
            if link.error is None:
                link.live_ctrl(listen=False, sing=False, fake=True, wait_hw=False)
        except LinkError:
            pass
        link.close()
    return finish()


if __name__ == "__main__":
    sys.exit(main())
