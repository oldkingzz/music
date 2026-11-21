"""Check 3, audio: do microphone frames come at the real pace and arrive intact, and is what the Mac sends to the
speaker exactly what gets played?

    .venv/bin/python host/checks/check_audio.py --target sim               (the Mac stand-in, no board)
    .venv/bin/python host/checks/check_audio.py --target auto --seconds 8  (the board, found by USB id)

Until the audio hardware is wired, the audio firmware (sketch 2 built with DUET_AUDIO_STANDIN=1, INFO audio=standin)
and the simulator use a stand-in microphone (a known melody, duet_host/fake_audio.py has the same numbers) and a
stand-in speaker (counts frames, keeps a CRC-32 of every sample played). No pin outside the chip is driven.

What it checks, in order:
  a. INFO. A firmware without audio (sketch 1) -> "skipped", exit 0. audio=i2s (real hardware) -> the bit-exact
     parts are skipped and the output says so.
  b. Record N s from the board's microphone: about 100 frames per second (within 2 %), no seq gaps, and (stand-in)
     every sample equal to the Python mirror.
  c. The Mac's note finder (audio_notes.clip_to_notes) on the recording finds the melody's notes.
  d. Play N s to the board's speaker: frames played = frames sent, no underrun while sending, nothing dropped, and
     (stand-in) the board's checksum of what it played = the Mac's checksum of what it sent.
  e. Both directions at once plus one NOTE_IN every 100 ms: every note answered; the round trips are reported.
Exit code 0 = all passed (or skipped). One line is appended to runs/checks.jsonl.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duet_host import audio_notes, fake_audio, runlog  # noqa: E402
from duet_host.audio_stream import BoardAudio, record_and_play  # noqa: E402
from duet_host.link import Link, LinkError, SimTransport  # noqa: E402

ONSET_TOL_S = 0.08         # the note finder's start times are within about 50 ms of the true ones


def expected_notes(n_samples):
    """Melody notes that lie completely inside the first n_samples: [(on_s, off_s, midi)]."""
    out = []
    loop = 0
    while loop * fake_audio.LOOP < n_samples:
        for start, sound, midi in fake_audio.note_table():
            s = loop * fake_audio.LOOP + start
            if s + sound <= n_samples:
                out.append((s / fake_audio.RATE, (s + sound) / fake_audio.RATE, midi))
        loop += 1
    return out


def compare_notes(found, want):
    """found: clip_to_notes output; want: expected_notes. Returns (ok, text)."""
    if not want:
        return False, "the recording is too short to hold one whole note"
    last_end = want[-1][1]
    inside = [n for n in found if n["on"] < last_end - 0.05]
    got = [n["midi"] for n in inside]
    exp = [m for _, _, m in want]
    worst = max((abs(n["on"] - w[0]) for n, w in zip(inside, want)), default=0.0)
    ok = got == exp and worst <= ONSET_TOL_S
    text = (f"{len(inside)} notes found where the melody has {len(exp)} whole notes"
            f"{'' if got == exp else f' (found {got}, want {exp})'}; start times off by at most {1000 * worst:.0f} ms")
    return ok, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sim", help="sim, auto, or a port such as /dev/cu.usbmodem101")
    ap.add_argument("--seconds", type=float, default=8.0, help="length of each recording / playback (8 s = one loop)")
    ap.add_argument("--sim-args", default="", help=argparse.SUPPRESS)   # tests: e.g. --no-audio (sketch 1)
    a = ap.parse_args()

    fails, rec = [], {"check": "audio", "target": a.target, "seconds": a.seconds}

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
        if a.target == "sim" and a.sim_args:
            link = Link(SimTransport(args=a.sim_args.split()))
        else:
            link = Link.open(a.target)
    except LinkError as e:
        print(f"cannot open: {e}")
        return 2
    try:
        rec["kind"], rec["port"] = link.kind, link.port
        print(f"connected to {link.kind} at {link.port}")

        # ---------------------------------------------------------------- a. what is this firmware?
        info = link.hello()
        rec["info"] = {k: v for k, v in info.items() if not k.startswith("_")}
        mode = info.get("audio")
        print(f"  INFO  fw={info.get('fw')} ver={info.get('ver')} audio={mode}")
        if mode is None:
            print("  skipped: this firmware has no audio (INFO has no audio=...; sketch 1?)")
            return finish("no audio")
        if mode in ("0", "1"):
            print("  skipped: this is the older audio firmware (INFO audio=0/1, no AUDIO_STATS). Upload the current "
                  "duet_base_audio to check audio.")
            return finish("older audio firmware")
        standin = mode == "standin"
        if not standin:
            print(f"  NOTE  audio={mode}: real hardware. The bit-exact parts (samples = mirror, checksum) are skipped.")
        try:
            st0 = link.audio_stats()
        except LinkError as e:
            need(False, f"the board answers AUDIO_QUERY ({e})")
            return finish()
        need(st0["mode_name"] == mode, f"AUDIO_STATS mode {st0['mode_name']} matches INFO audio={mode}")
        ba = BoardAudio(link)
        n_want = int(round(a.seconds * 100))
        if standin and a.seconds < fake_audio.LOOP / fake_audio.RATE:
            print(f"  NOTE  {a.seconds:g} s is shorter than the 8 s loop: only the first part of the melody is checked")

        # ---------------------------------------------------------------- b. record
        r = ba.record(a.seconds)
        fps = r.frames_per_second
        gaps_ms = np.diff(r.arrival) * 1000 if len(r.arrival) > 1 else np.zeros(1)
        span = (r.arrival[-1] - r.arrival[0]) / (r.frames - 1) * 1000 if r.frames > 1 else 0.0
        rec["record"] = {"frames": r.frames, "seconds": round(r.seconds, 3), "fps": round(fps, 2),
                         "first_seq": r.first_seq, "seq_gaps": r.seq_gaps, "missing": r.missing,
                         "mean_period_ms": round(span, 3), "max_arrival_gap_ms": round(float(gaps_ms.max()), 2)}
        print(f"  recorded {r.frames} frames in {r.seconds:.2f} s = {fps:.2f} per second; mean period {span:.3f} ms, "
              f"longest gap between two arrivals {gaps_ms.max():.1f} ms")
        need(abs(r.frames - 100 * r.seconds) <= 0.02 * 100 * r.seconds,
             f"frame count {r.frames} is within 2 % of 100 per second ({100 * r.seconds:.0f})")
        need(r.seq_gaps == 0 and r.missing == 0, f"no seq gaps ({r.seq_gaps} gaps, {r.missing} frames missing)")
        if standin:
            need(r.first_seq == 0, f"the first frame has seq 0 (got {r.first_seq})")
            ref = fake_audio.samples(0, len(r.samples))
            bad = int(np.count_nonzero(ref != r.samples))
            rec["record"]["samples_differ"] = bad
            need(len(r.samples) > 0 and bad == 0,
                 f"all {len(r.samples)} samples equal the Python mirror ({bad} differ)")

        # ---------------------------------------------------------------- c. notes
        found = audio_notes.clip_to_notes(r.samples.astype(np.float32) / 32768.0)
        rec["notes_found"] = [n["midi"] for n in found]
        if standin:
            ok, text = compare_notes(found, expected_notes(len(r.samples)))
            need(ok, "note finder on the recording: " + text)
        else:
            print(f"  NOTE  note finder heard {len(found)} notes: "
                  + " ".join(audio_notes.midi_name(n['midi']) for n in found[:20]))

        # ---------------------------------------------------------------- d. play
        x = fake_audio.samples(0, int(a.seconds * fake_audio.RATE))
        p = ba.play(x)
        rec["play"] = {k: v for k, v in p.items() if k not in ("before", "after")}
        print(f"  played {p['frames_sent']} frames (1 silent + {p['frames_sent'] - 1}) in {p['seconds']:.2f} s; "
              f"latest send {p['send_late_ms_max']:.1f} ms behind schedule; buffer level at the last send "
              f"{p['level_at_last_send']} frames")
        need(p["frames_played"] == p["frames_sent"],
             f"frames played {p['frames_played']} = frames sent {p['frames_sent']}")
        need(p["underruns_while_sending"] == 0, f"no underrun while sending ({p['underruns_while_sending']})")
        print(f"  info  underruns once the stream ended: {p['underruns_after_drain']} (1 is normal: the buffer runs dry)")
        need(p["dropped"] == 0 and p["seq_gaps"] == 0, f"nothing dropped ({p['dropped']}), no seq gaps ({p['seq_gaps']})")
        if standin:
            need(p["crc_match"], f"checksum of what was played 0x{p['crc_board']:08x} = Mac's checksum of what it "
                                 f"sent 0x{p['crc_mac']:08x}")

        # ---------------------------------------------------------------- e. both at once, with notes
        link.configure("third_below", 0, 4)
        out = record_and_play(link, x, a.seconds, note_every=0.1)
        r2, p2, notes = out["recording"], out["play"], out["notes"]
        answered = [n for n in notes if "error" not in n]
        rtts = [n["rtt_us"] for n in answered]
        rec["duplex"] = {"frames": r2.frames, "seq_gaps": r2.seq_gaps, "notes": len(notes), "answered": len(answered),
                         "rtt_us": {"median": runlog.pct(rtts, 50), "p95": runlog.pct(rtts, 95),
                                    "max": max(rtts, default=0)},
                         "play": {k: v for k, v in p2.items() if k not in ("before", "after")}}
        print(f"  both ways: {r2.frames} mic frames, {p2['frames_played']}/{p2['frames_sent']} speaker frames, "
              f"{len(answered)}/{len(notes)} notes answered")
        if rtts:
            print(f"  note round trip while audio streams: median {runlog.pct(rtts, 50):.0f} us, "
                  f"95 % {runlog.pct(rtts, 95):.0f} us, max {max(rtts):.0f} us")
        need(len(notes) >= int(a.seconds * 10) - 2 and len(answered) == len(notes),
             f"every NOTE_IN answered ({len(answered)} of {len(notes)})")
        need(all(n.get("status") == 0 for n in answered), "every answer has status ok")
        need(abs(r2.frames - 100 * r2.seconds) <= 0.02 * 100 * r2.seconds and r2.seq_gaps == 0,
             f"mic while playing: {r2.frames} frames, {r2.seq_gaps} seq gaps")
        need(p2["frames_played"] == p2["frames_sent"] and p2["underruns_while_sending"] == 0 and p2["dropped"] == 0,
             f"speaker while recording: {p2['frames_played']}/{p2['frames_sent']} played, "
             f"{p2['underruns_while_sending']} underruns, {p2['dropped']} dropped")
        if standin:
            bad2 = int(np.count_nonzero(fake_audio.samples(0, len(r2.samples)) != r2.samples))
            need(r2.first_seq == 0 and bad2 == 0, f"mic while playing: samples equal the mirror ({bad2} differ)")
            need(p2["crc_match"], "speaker while recording: checksum equal")

        st = link.stats()
        rec["stats"] = st
        need(st["bad_checksum"] == 0, "no frame from the device had a bad checksum")
    except LinkError as e:
        need(False, f"link error: {e}")
    finally:
        link.close()
    return finish()


if __name__ == "__main__":
    sys.exit(main())
