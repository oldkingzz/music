# Duet ESP32-S3 base

The base system for Duet on an ESP32-S3 (Arduino only) plus a Mac over USB: a link, a note path with a
plug-in slot for harmony models, a clip path and a job queue on the Mac, and a local web panel.

The wire protocol is written out in `firmware/libraries/DuetCore/src/duet_protocol.h`. A longer set of working
notes (`ALL_INFO_v1.md`, Chinese: design, pin map, what was tested and what was not, open questions) is kept
out of the repository for now.

State on 2026-11-21: all checks pass against the Mac simulator. Nothing has run on a real board yet.

```sh
PY=../duet_v0/.venv/bin/python          # Python 3.11 with numpy, scipy, soundfile, pyserial, librosa
                                        # (the tools/ scripts use this one too; set DUET_PY to change it)

sh tools/run_all_checks.sh sim          # C++ tests, Python tests, link check, note check (no board needed)
sh tools/build_firmware.sh duet_base    # compile only
sh tools/run_panel.sh                   # local panel at http://127.0.0.1:8765 (opens nothing by itself)

# With the board, and only after the owner says so:
sh tools/upload_firmware.sh duet_base --yes-upload    # replaces the program on the board
$PY host/checks/check_link.py --target auto
$PY host/checks/check_notes.py --target auto
```

| Folder | What |
|---|---|
| `firmware/libraries/DuetCore/` | portable C++ core: protocol, engine, plug-in slot (board, simulator and tests compile the same files) |
| `firmware/duet_base/` | sketch 1: link + note path |
| `firmware/duet_base_audio/` | sketch 2: adds audio, servo, LED; audio stays off until the Mac turns it on |
| `firmware/host_tests/` | unit tests and the Mac simulator of the board |
| `host/duet_host/` | Mac side: link, blocks, routes, clips, jobs, web panel |
| `host/checks/`, `host/tests/` | checks that run against the simulator or the board; unit tests |
| `testdata/` | 527 songs, 46,310 notes with the Python reference answers |
| `tools/` | build, upload, run panel, run all checks |
