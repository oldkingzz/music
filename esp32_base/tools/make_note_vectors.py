"""Write the note test vectors: what the Python reference answers on the bench3 songs.

For every song of the bench3 evaluation sets (bach, folk, pop, pop909; the first 300 of each split) this
feeds the notes Duet would hear (bench.heard: her melody with 10 % mishearings) through
  1. the original bench3 system in duet_v0 (futures/v3_nolatency/systems/<plugin>.py), and
  2. the copy in esp32_base/host/duet_host/harmonizers.py,
checks that the two agree on every note, and writes one text file that the C++ core
(firmware/host_tests/test_core) and the board check (host/checks/check_notes.py) replay.

Onsets and lengths are rounded to 32-bit floats first, because that is what crosses the wire, so the file
holds exactly the numbers the board sees.

    .venv/bin/python tools/make_note_vectors.py --split val
    .venv/bin/python tools/make_note_vectors.py --split val --max-songs 40 --out testdata/note_vectors_val_small_v1.txt

File format:
    H <plugin name> <plugin id>
    S <dataset> <song index> <tonic> <M> <K> <n_pos> <n_notes>
    P <pos ...>                                    bar position of each beat, cut to 1000 beats
    N <i> <midi> <has_prev> <onset> <prev_dur> <expected>
"""
import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))
from duet_host import harmonizers, proto  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duet-v0", default=str(ROOT.parent / "duet_v0"), help="the duet_v0 folder with bench3")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--plugin", default="third_below")
    ap.add_argument("--max-songs", type=int, default=300, help="songs per dataset")
    ap.add_argument("--out")
    a = ap.parse_args()

    duet = Path(a.duet_v0)
    sys.path.insert(0, str(duet / "duet_beat"))
    sys.path.insert(0, str(duet / "futures" / "v3_nolatency"))
    import bench as B2   # noqa: E402
    import bench3 as B3  # noqa: E402

    original = importlib.import_module(f"systems.{a.plugin}").build({})
    mine = harmonizers.make(a.plugin)
    pid = harmonizers.ID_BY_NAME[a.plugin]
    out = Path(a.out) if a.out else ROOT / "testdata" / f"note_vectors_{a.split}_v1.txt"
    if out.exists():
        sys.exit(f"{out} exists; choose another --out (files are never overwritten)")

    lines = [f"H {a.plugin} {pid}"]
    songs = notes = cut = 0
    for ds, items in B3.eval_sets(a.split).items():
        for idx, song in enumerate(items[:a.max_songs]):
            heard = sorted(B2.heard(song)["sop"], key=lambda n: n[0])
            pos = [int(b["pos"]) for b in song["beats"]]
            if len(pos) > proto.MAX_POS:
                pos, cut = pos[:proto.MAX_POS], cut + 1
            tonic, M, K = int(song["tonic"]), int(song["M"]), int(song["K"])
            assert 0 <= tonic < 256 and 0 < M < 256 and 0 <= K < 65536 and all(0 <= p < 256 for p in pos), (ds, idx)
            meta = {"tonic": tonic, "M": M, "K": K, "pos": pos}
            original.start(dict(meta))
            mine.start(dict(meta))
            lines.append(f"S {ds} {idx} {tonic} {M} {K} {len(pos)} {len(heard)}")
            lines.append("P " + " ".join(map(str, pos)))
            prev = None
            for i, (on, off, m) in enumerate(heard):
                onset = proto.f32(on)
                prev_dur = None if prev is None else proto.f32(prev[1] - prev[0])
                want = int(original.step(i, onset, int(m), prev_dur))
                got = int(mine.step(i, onset, int(m), prev_dur))
                assert want == got, f"{ds} song {idx} note {i}: bench3 says {want}, the copy says {got}"
                assert 0 <= int(m) < 128 and -1 <= want < 128
                lines.append(f"N {i} {int(m)} {0 if prev is None else 1} {onset!r} "
                             f"{0.0 if prev_dur is None else prev_dur!r} {want}")
                prev = (on, off)
                notes += 1
            songs += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"{out}: plug-in {a.plugin}, {songs} songs, {notes} notes, {out.stat().st_size / 1e6:.2f} MB"
          + (f"; {cut} songs had more than {proto.MAX_POS} beats (pos cut)" if cut else ""))


if __name__ == "__main__":
    main()
