"""Check 1, the link: can the Mac talk to the board, and how long does one exchange take?

    .venv/bin/python host/checks/check_link.py --target sim      (tonight: the Mac stand-in)
    .venv/bin/python host/checks/check_link.py --target auto     (the board, found by USB id)

What it checks, in order:
  1. HELLO -> INFO: firmware name, protocol version; on the board also flash = 16 MB and PSRAM about 8 MB
     (this is the first proof that the OPI PSRAM build setting is right).
  2. PING x N: none lost, the echo is right; reports the round trip (median, 95 %, max).
  3. The device clock keeps time with the Mac clock.
  4. Junk bytes and a cut-off frame do not break the link (the device drops the partial frame after 100 ms).
  5. Board only: closing and reopening the port does not reset the board (its clock keeps counting).
Exit code 0 = all passed. One line is appended to runs/checks.jsonl.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duet_host import proto, runlog  # noqa: E402
from duet_host.link import Link, LinkError  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sim", help="sim, auto, or a port such as /dev/cu.usbmodem101")
    ap.add_argument("--pings", type=int, default=300)
    ap.add_argument("--led", action="store_true", help="also blink the on-board LED green for 2 s (look at it)")
    a = ap.parse_args()

    fails, rec = [], {"check": "link", "target": a.target}

    def need(ok, what):
        print(("  ok    " if ok else "  FAIL  ") + what)
        if not ok:
            fails.append(what)

    try:
        link = Link.open(a.target)
    except LinkError as e:
        print(f"cannot open: {e}")
        return 2
    try:
        board = link.kind == "board"
        rec["kind"], rec["port"] = link.kind, link.port
        print(f"connected to {link.kind} at {link.port}")

        info = link.hello()
        rec["info"] = info
        print("  INFO  " + "; ".join(f"{k}={v}" for k, v in info.items() if not k.startswith("_")))
        need(info.get("proto") == str(proto.PROTO_VERSION), f"protocol version is {proto.PROTO_VERSION}")
        need("third_below" in info.get("plugins", ""), "plug-in third_below is present")
        if board:
            need(int(info.get("flash", 0)) == 16 * 1024 * 1024, f"flash is 16 MB (reported {info.get('flash')})")
            need(int(info.get("psram", 0)) >= 7 * 1024 * 1024, f"PSRAM is about 8 MB (reported {info.get('psram')})")

        rtts, dev, host = [], [], []
        for _ in range(a.pings):
            r = link.ping()
            rtts.append(r["rtt_us"])
            dev.append(r["dev_us"])
            host.append(time.perf_counter())
        rec["ping"] = {"n": len(rtts), "median_us": runlog.pct(rtts, 50), "p95_us": runlog.pct(rtts, 95),
                       "max_us": max(rtts)}
        need(len(rtts) == a.pings, f"{a.pings} pings answered, none lost")
        print(f"  round trip: median {rec['ping']['median_us']:.0f} us, 95 % {rec['ping']['p95_us']:.0f} us, "
              f"max {rec['ping']['max_us']:.0f} us")

        t0h, t0d = time.perf_counter(), link.ping()["dev_us"]
        time.sleep(1.0)
        t1h, t1d = time.perf_counter(), link.ping()["dev_us"]
        ratio = ((t1d - t0d) & 0xFFFFFFFF) / 1e6 / (t1h - t0h)
        rec["clock_ratio"] = ratio
        need(abs(ratio - 1.0) < 0.01, f"device clock runs at {ratio:.4f} x the Mac clock (want 1.00 +/- 0.01)")

        link.send(b"\x00\xffboot text\r\n\xa5\x00" + bytes(range(1, 60)))       # junk, no valid frame in it
        need(link.ping()["rtt_us"] > 0, "a ping right after junk bytes is answered")
        link.send(bytes([0xA5, 0x5A, proto.T.LOG, 0, 100, 0, 0x41]))              # header promises 100 bytes
        time.sleep(1.2)                                                            # 7 pending bytes x 100 ms
        need(link.ping()["rtt_us"] > 0, "a ping after a cut-off frame and 1.2 s of silence is answered")

        if a.led:
            link.led(2)
            time.sleep(2.0)
            link.led(0)
            print("  LED: it should have blinked green for 2 s (not checked by the script)")

        st = link.stats()
        rec["stats"] = st
        need(st["bad_checksum"] == 0, "no frame from the device had a bad checksum")

        if board:
            before = link.ping()["dev_us"]
            link.close()
            time.sleep(0.5)
            link = Link.open(a.target)
            after = link.ping()["dev_us"]
            gap = (after - before) & 0xFFFFFFFF
            rec["reopen_gap_us"] = gap
            need(0.3e6 < gap < 10e6, f"reopening the port did not reset the board (its clock moved on by {gap / 1e6:.2f} s)")
    except LinkError as e:
        need(False, f"link error: {e}")
    finally:
        link.close()

    rec["passed"], rec["failed"] = not fails, fails
    path = runlog.append("checks", rec)
    print(("PASS" if not fails else f"FAIL ({len(fails)})") + f"  -> {path}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
