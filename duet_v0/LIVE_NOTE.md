# Live / studio add-on — read this before the live test

> 给 Vincent：把 `duet_live_addon.zip` 解压后，把里面的 `duet_v0/` **合并**进 Mac mini 上已有的 `duet_v0/`
> （只新增文件，不覆盖任何已有文件）。然后对 mini 上的 Claude 说：`Read LIVE_NOTE.md, then set up the live test.`

## What Vincent wants (the earlier session misread this)

A **live listening test of Duet's harmony**, not a quiz. Vincent plays a test melody from his **phone
speaker**, the Mac mini listens with a **microphone**, Duet sings its harmony note in real time, and
Vincent hears it in **headphones** to judge whether it fits. He wants to compare the new model
(`r2_mix_v2` or whatever won the overnight loop) with `m1_filter_v5lo` and the third-below rule.

## What this add-on contains (all new files; nothing in the first bundle is replaced)

- `duet_beat/live_mic.py` — real-time `LiveDuet` on a sounddevice duplex stream, beat-based (click
  track, key given, one note per beat). `--selftest` reproduces the batch Twinkle numbers offline (no
  audio device); `--listen` = mic test; `--bpm`, `--key`, `--meter`, `--beats`, `--range LO HI`.
  Its decision hook is `LiveDuet._decide(k)` (line ~160).
- `futures/panel/policy_live.py` — `PolicyDuet(lm.LiveDuet)`: the same live loop with the bench-v2
  policy plugged into the decision. **This is the template for wiring any bench3 system live**:
  subclass `LiveDuet`, feed each detected note to the system's `step(i, onset, midi, prev_dur)`,
  sing what it returns.
- `futures/panel/duet_panel.py` — local test page (http://127.0.0.1:8765). Its *Live* tab runs the D5
  effects engine (`futures/d5_live_fx/iter2/fx.py`: mic → diatonic third below → output, with gate and
  meters, every run recorded). Its *Studio* tab runs every offline effect on a clip (`studio.py`).
  Note: the panel's Live tab is the third-below effect, **not** the bench3 models.
- `futures/live_test/phone/` — the **test audio to put on the phone**:
  - `test_her_v11_countin.mp3` — a real singer (vocadito track 11, CC BY 4.0), with a count-in;
  - `twinkle_C_80bpm_countin.mp3` — Twinkle in C, 80 bpm, with a count-in (one bar of clicks);
  - `.wav` versions of the same, and `twinkle_80bpm_nocount.wav`.
- `futures/live_test/fx_live_phone2.py` — the 10-30 phone→mic test script for the D5 engine
  (picks devices by name and never uses the iPhone microphone; records mic + output every second).
- Also: `follow.py`, `singer.py`, `echo/`, `harmony/` (code), `live/` (code, cache), `futures/d4_styles`,
  `futures/d5_live_fx`, `duet_beat/bench/models` (policy weights), `duet_beat/bench_systems`.
  Big recordings and old run outputs were left out.

Checked on the laptop (merged copy, laptop venv): `live_mic.py --selftest` runs; `duet_panel`,
`studio`, `studio_effects`, `countin` import; `policy_live.load_policy()` loads the weights.
Not checked: anything with a real mic/speaker on the Mac mini.

## Suggested live test (announce each step to Vincent and wait for 开始 before opening the mic)

1. `sd.query_devices()` — pick the mic and the headphones by name. If the headphones are Bluetooth
   (AirPods), their mic switches them to low-quality call mode: prefer the Mac mini's own input or a
   wired/USB mic for input, and the headphones for output only. Duet's voice must go **only** to the
   headphones, never to a speaker the mic can hear.
2. Mic check: `live_mic.py --listen --key C --bpm 80` while the phone plays the Twinkle file.
3. Write `futures/v4_selfiter/live_bench3.py`: subclass `LiveDuet` like `PolicyDuet` does and run
   any bench3 system live (`--system r2_mix_v2 | m1_filter_v5lo | third_below`). The click / count-in
   must line up with the phone file (80 bpm, one bar count-in; the vocadito clip's tempo and key are
   in `futures/live_test/` json/scripts and the 10-30 transcript notes: start the phone on the first click).
4. Record every run (mic + Duet output) to a new file under `runs_mini/live/<HHMM>_<system>/` so
   Vincent can re-listen, and write one line per run: system, song, his verdict.
5. Same rules as PROGRAM.md §1 (no deletes/overwrites, no publishing, no git).
