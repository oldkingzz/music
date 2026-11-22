# Duet ESP32-S3 base

The base system for Duet on an ESP32-S3 (Arduino only) plus a Mac over USB: a link, a note path with a
plug-in slot for harmony models, a clip path and a job queue on the Mac, and a local web panel.

The wire protocol is written out in `firmware/libraries/DuetCore/src/duet_protocol.h`. A longer set of working
notes (`ALL_INFO_v1.md`, Chinese: design, pin map, what was tested and what was not, open questions) is kept
out of the repository for now.

State on 2026-11-22: all checks pass against the Mac simulator. Nothing has run on a real board yet.

**Which sketch for which hardware**

- **The team's board (I2S MEMS microphone on GPIO 11/12/13, I2S amplifier on GPIO 4/5/6): use sketch 3,
  `duet_live`.** It runs the whole duet on the board: microphone → pitch tracker → her notes → harmony plug-in →
  Duet's notes → synth → speaker, and sends every note to the Mac. The pitch tracker is Christine's
  (`quantization_code_3`, branch `Christine`) and the synth is Xinrui's (`quantization_code_4`, branch `SIgrid`),
  both lifted into `DuetCore` with their constants and arithmetic unchanged. It drives no pin at boot: the
  microphone and the speaker start only when the Mac asks. It also has a built-in fake microphone ("Twinkle
  Twinkle" at 16 kHz), chosen from the Mac at run time, so a bare board with nothing wired runs the whole chain.
  It does not use the OLED, the SD card, the button or Wi-Fi; for those, flash the teammates' own sketch.
- **Sketch 2, `duet_base_audio`, is NOT for the team's board.** Its `i2s` variant uses the older pin map from
  `duet_v0/docs/PLAN.md` (PCM1808 ADC on GPIO 4/5/6/7, amplifier on 15/39/40/41, servo on 13), which collides with
  the team's wiring. Its default build is stand-in audio: a stand-in microphone sends a known melody at 24 kHz
  and a stand-in speaker counts frames; no I2S driver starts and no pin from `config.h` is driven. The firmware
  says which one it is in its INFO reply (`audio=standin` or `audio=i2s`).
- Sketch 1, `duet_base`, is the link and the note path only.

```sh
PY=../duet_v0/.venv/bin/python          # Python 3.11 with numpy, scipy, soundfile, pyserial, librosa
                                        # (the tools/ scripts use this one too; set DUET_PY to change it)

sh tools/run_all_checks.sh sim          # C++ tests, Python tests, link, note, audio and live checks (no board needed)
sh tools/build_firmware.sh duet_live            # sketch 3, the team's wiring; compile only
sh tools/build_firmware.sh duet_base            # sketch 1, compile only
sh tools/build_firmware.sh duet_base_audio      # sketch 2 with STAND-IN audio (no hardware, drives no pin)
sh tools/build_firmware.sh duet_base_audio i2s  # sketch 2 for the older PLAN.md hardware (not the team's board)
sh tools/run_panel.sh                   # local panel at http://127.0.0.1:8765 (opens nothing by itself)
$PY host/checks/check_live.py --target sim-live       # the live chain in the simulator

# With the board, and only after the owner says so:
sh tools/upload_firmware.sh duet_live --yes-upload    # replaces the program on the board
$PY host/checks/check_link.py --target auto
$PY host/checks/check_notes.py --target auto
$PY host/checks/check_live.py --target auto           # fake microphone, speaker OFF: drives no audio pin
$PY host/checks/check_live.py --target auto --real-mic --speaker   # only with the hardware wired; it makes sound
$PY host/checks/check_audio.py --target auto          # needs sketch 2; skips itself on the other sketches
```

In the panel: connect, then use the section "现场" (live). Start with the fake microphone and the speaker off.

| Folder | What |
|---|---|
| `firmware/libraries/DuetCore/` | portable C++ core: protocol, engine, plug-in slot, pitch tracker, synth, live chain (board, simulator and tests compile the same files) |
| `firmware/duet_live/` | sketch 3: the whole duet on the team's wiring (microphone 11/12/13, amplifier 4/5/6) |
| `firmware/duet_base/` | sketch 1: link + note path |
| `firmware/duet_base_audio/` | sketch 2: adds audio, servo, LED; audio stays off until the Mac turns it on. Stand-in audio by default, `i2s` variant for the real hardware |
| `firmware/host_tests/` | unit tests and the Mac simulator of the board |
| `host/duet_host/` | Mac side: link, blocks, routes, clips, jobs, web panel |
| `host/checks/`, `host/tests/` | checks that run against the simulator or the board; unit tests |
| `testdata/` | 527 songs, 46,310 notes with the Python reference answers |
| `tools/` | build, upload, run panel, run all checks |
