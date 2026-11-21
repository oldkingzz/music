"""Check 2, the note path: does the plug-in on the board answer exactly what the Python reference answers?

    .venv/bin/python host/checks/check_notes.py --target sim
    .venv/bin/python host/checks/check_notes.py --target auto            (the board)
    .venv/bin/python host/checks/check_notes.py --target auto --max-songs 20

It replays testdata/note_vectors_val_v1.txt (527 bench3 songs, 46,310 notes): for every song it sends CONFIG,
then her notes one at a time, and compares each answer with the note recorded from Python and with the
Python reference run right now. It reports the time per note: the round trip seen from the Mac and the
time inside the plug-in measured on the device.
Exit code 0 = every note equal. One line is appended to runs/checks.jsonl.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duet_host import harmonizers, runlog  # noqa: E402
from duet_host.link import Link, LinkError  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def read_vectors(path):
    """Yields (plugin name, plugin id), then per song (dataset, index, meta, notes) with
    notes = [(i, midi, onset, prev_dur or None, expected)]."""
    plugin = None
    song = None
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == "H":
            plugin = (f[1], int(f[2]))
        elif f[0] == "S":
            if song:
                yield plugin, song
            song = {"ds": f[1], "idx": int(f[2]), "meta": {"tonic": int(f[3]), "M": int(f[4]), "K": int(f[5])},
                    "notes": []}
        elif f[0] == "P":
            song["meta"]["pos"] = [int(x) for x in f[1:]]
        elif f[0] == "N":
            song["notes"].append((int(f[1]), int(f[2]), float(f[4]), float(f[5]) if f[3] == "1" else None, int(f[6])))
    if song:
        yield plugin, song


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sim")
    ap.add_argument("--vectors", default=str(ROOT / "testdata" / "note_vectors_val_v1.txt"))
    ap.add_argument("--max-songs", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    try:
        link = Link.open(a.target)
    except LinkError as e:
        print(f"cannot open: {e}")
        return 2
    rec = {"check": "notes", "target": a.target, "kind": link.kind, "port": link.port, "vectors": Path(a.vectors).name}
    songs = notes = bad_file = bad_ref = 0
    rtts, comp, examples = [], [], []
    t_start = time.perf_counter()
    try:
        info = link.hello()
        rec["fw"] = f"{info.get('fw')} {info.get('ver')}"
        print(f"connected to {link.kind} at {link.port}: {rec['fw']}")
        for (name, pid), song in read_vectors(a.vectors):
            if a.max_songs and songs >= a.max_songs:
                break
            m = song["meta"]
            link.configure(pid, m["tonic"], m["M"], m["K"], m["pos"])
            ref = harmonizers.make(pid)
            ref.start(dict(m))
            for i, midi, onset, prev_dur, want in song["notes"]:
                r = link.note(i, midi, onset, prev_dur)
                py = ref.step(i, onset, midi, prev_dur)
                rtts.append(r["rtt_us"])
                comp.append(r["compute_us"])
                notes += 1
                if r["midi"] != want:
                    bad_file += 1
                    if len(examples) < 10:
                        examples.append(f"{song['ds']} song {song['idx']} note {i}: her {midi} -> {r['midi']}, file says {want}")
                if r["midi"] != py:
                    bad_ref += 1
            songs += 1
            rec["plugin"] = name
    except LinkError as e:
        print(f"  FAIL  link error after {notes} notes: {e}")
        rec["error"] = str(e)
        bad_file += 1
    finally:
        st = link.stats()
        link.close()
    secs = time.perf_counter() - t_start
    rec.update(songs=songs, notes=notes, differ_from_file=bad_file, differ_from_python=bad_ref, seconds=round(secs, 2),
               rtt_us={"median": runlog.pct(rtts, 50), "p95": runlog.pct(rtts, 95), "max": max(rtts, default=0)},
               compute_us={"median": runlog.pct(comp, 50), "p95": runlog.pct(comp, 95), "max": max(comp, default=0)},
               stats=st)
    for e in examples:
        print("  " + e)
    print(f"  plug-in {rec.get('plugin')}: {songs} songs, {notes} notes in {secs:.1f} s")
    print(f"  differ from the recorded Python output: {bad_file};  from Python run now: {bad_ref}")
    if rtts:
        print(f"  per note, seen from the Mac: median {rec['rtt_us']['median']:.0f} us, 95 % {rec['rtt_us']['p95']:.0f} us, "
              f"max {rec['rtt_us']['max']:.0f} us")
        print(f"  per note, inside the plug-in on the device: median {rec['compute_us']['median']:.0f} us, "
              f"max {rec['compute_us']['max']:.0f} us   (bench3 calls a model SLOW above 2000 us)")
    ok = notes > 0 and bad_file == 0 and bad_ref == 0 and st["bad_checksum"] == 0
    rec["passed"] = ok
    path = runlog.append("checks", rec)
    print(("PASS" if ok else "FAIL") + f"  -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
