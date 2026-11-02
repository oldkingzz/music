# D5 hardware test (for Vincent; run only after he says 开始)

Everything below opens the audio interface. Nothing here was run by the builder (no audio devices were
opened). Each step says what you should see. Recordings land in `futures/d5_live_fx/local_only/hw/`
(they contain a voice: keep them local).

Setup: Radxa Cubie A7S, USB audio interface (ideally 2 in / 2 out), the repo copied to the board with
`.venv` (numpy, scipy, sounddevice). Run every command from the repo root. Volume knobs at a low,
fixed position; write the knob positions down.

## 0. Board compute (no audio)

```
OMP_NUM_THREADS=1 .venv/bin/python futures/d5_live_fx/latency.py --timing-only --tag board
```
Look at `p99_us` for `sr 48000, block 128`: it must be well under 2667 us (one block). If it is over,
rerun step 2 with `--block 256` and note the result.

## 1. Cable loopback, engine bypassed (the interface's own round trip)

1. Cable from interface OUT 1 to IN 1 (line level, input gain at minimum).
2. `python futures/d5_live_fx/hw_run.py devices` and check the USB interface is the default device.
3. `python futures/d5_live_fx/hw_run.py roundtrip --sr 48000 --block 128`
4. Expected: `round trip ... ms` around 8-14 ms, peak in between -30 and -6 dBFS. Write the number down.

## 2. Cable loopback with the engine in the loop (A5.1, the real number)

Needs 2 in / 2 out. Cable OUT 1 -> IN 1 (the "mic") and OUT 2 -> IN 2 (the "speaker").
```
python futures/d5_live_fx/hw_run.py stereo --sr 48000 --block 128
```
It plays a short synthetic sung phrase on OUT 1, the engine harmonizes what arrives on IN 1 and sends it
to OUT 2, and both inputs are recorded. Expected JSON: `mic_to_speaker_ms` <= 30 (sim estimate ~20 ms),
`compute_ms_p99` < `block_ms`. Pass for A5.1 if `mic_to_speaker_ms` <= 30.
With a 1 in / 1 out interface instead: A5.1 ~= step 1 round trip + 6.3 ms (worst-case PSOLA delay at
160 Hz) + compute p99 from step 0.

## 3. Acoustic loopback

1. Mic 10 cm in front of the Duet speaker, both on the table.
2. `python futures/d5_live_fx/hw_run.py roundtrip --sr 48000 --block 128`
3. Expected: step 1 number + ~0.3 ms (10 cm of air). If it differs by more than 2 ms, note it.

## 4. Room: no howling for 3 minutes (A5.2) and silence after she stops (A5.3)

1. Mic on a stand, singer's mouth 15-30 cm from it. Speaker 1 m from the mic, pointing at the audience.
2. Set the level: singer sings a sustained "ah"; adjust the amp so Duet at 2 m from the speaker sounds as
   loud as the singer heard from 2 m (a phone SPL app at the listener spot, one at a time).
3. `python futures/d5_live_fx/hw_run.py live --sr 48000 --block 128 --tonic <key pc, C=0 ... B=11> --seconds 180`
   (2 s of soft noise first = sound check: stay silent), then sing 3 minutes with pauses of 2-3 s
   between phrases. Pass A5.2 if no ringing/howl is heard and the recording shows none.
4. Margin: repeat with `--gain-db 3`, `6`, `9`, `12` (30 s each is enough). The last value without howl is
   the margin (sim: >= 12 dB with the guard). Stop immediately (Ctrl-C) at the first sign of ringing.
5. Without guard, for comparison: `--no-guard --gain-db 0`, 30 s.
6. A5.3: in the 3-minute file, after each phrase end the speaker channel (R) must fall below -40 dBFS within
   0.2 s and stay silent during pauses (no clicks, no echo replays).
7. Send the wav files (local only) to the builder for analysis with `sim5.howl` / `sim5.gate_metrics`.
