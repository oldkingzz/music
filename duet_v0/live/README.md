# Duet live: the real-time app on the MacBook

`live/` is the program that runs during a performance. It takes the singer's voice 10 ms at a
time, works out where she is in the song, predicts when her next note will land, and makes each
harmony note sound on time: with its built-in voice, or through Logic Pro over MIDI. Before the
machine comes in it nods (servo) one beat early. Every 10 ms block goes through one function,
`Engine.process()`. Three backends feed it:

| backend | audio in | audio out | servo / LED | status |
|---|---|---|---|---|
| `file` | a wav, at real-time pace or `--fast`; optional simulated room echo (`--simulate-bleed`) | stereo wav (L = what the engine heard, R = machine) | written to the log | **tested** |
| `device` | ESP32-S3 `MIC_AUDIO` frames over USB CDC | `SPK_AUDIO` frames | `SERVO`, `LED`, `PING` frames | **tested against the firmware team's fake device** (`firmware/tools/fake_device.py`) on a pty; no board |
| `mac` | MacBook mic (sounddevice duplex stream) | MacBook speaker / headphones | (none) | callback **driven with synthetic timestamps**; a real device was never opened |

What changed in phase 2 (this README is the current state):

* **Protocol as the firmware implements it.** live/ now uses `firmware/tools/proto.py` as its
  only codec. Sequence numbers are **per message type** in both directions. The speaker jitter
  buffer is pre-filled with **4 frames**, and `--out-latency` defaults to **0.052 s** on the
  device. DTR/RTS are never touched. Stale frames are discarded on open, and nothing is
  drained on close. A **level keeper** repairs the speaker buffer after an underrun.
* **Echo guard** (`echo/guard.py`): canceller + pitch mask, with a 2 s pink-noise **sound
  check** first. It is **on by default for `mac` and `device`**. File mode can simulate the
  room (`--simulate-bleed DB`).
* **Learned harmony** (`harmony/model.py`, a CRF trained on Bach chorales) is the default
  (`--harmony learned|rule|cache`). If the model cannot be loaded, the rule plan is used.
* **MIDI out** (`--midi-out [NAME]`): a virtual CoreMIDI source "Duet" for **Logic Pro** (see
  [LOGIC.md](LOGIC.md)), with latency compensation and the nod as CC 20.
* **Demo assets** in `demo/`: MIDI files (melody + harmony tracks, real seconds) and wav renders.
* `rehearse --notes FILE`: use annotated notes of the reference (for real singing).

## Run it

The Python is `project/duet_v0/.venv/bin/python` (3.11). Phase 2 added **mido 1.3.3** and
**python-rtmidi 1.5.8** (wheels 55 kB + 147 kB; the venv grew from 623 to 624 MB). Everything
else was already there: numpy, scipy, soundfile, sounddevice, pyserial, and matplotlib (for
figures only). Run all commands from `project/duet_v0`.

```bash
PY=.venv/bin/python

# 0. test inputs (synthetic singer + 5 re-timed performances + a rehearsal take, ~9 MB)
$PY live/tests/make_inputs.py

# 1. Rehearsal: analyse the reference once, cache it (<prefix>.npz + <prefix>.json)
$PY live/duet_live.py rehearse live/tests/data/singer.wav -o live/cache/singer
#    real singing: give the reference's notes (rows "onset offset midi"); see "Real recordings"
$PY live/duet_live.py rehearse take1.wav -o live/cache/song --notes take1_notes.csv
#    optional: learn her timing habits from a rehearsal take
$PY live/duet_live.py rehearse song.wav -o live/cache/song --take take1.wav
#    optional: the evaluation's front end (both default OFF; see "Input gate and detector bias")
$PY live/duet_live.py rehearse take1.wav -o live/cache/song --notes take1_notes.csv --gate-db 15 --bias dcs

# 2. Perform, file backend (fast, or real time with the live view at http://127.0.0.1:8765/)
$PY live/duet_live.py run --ref live/cache/singer --input live/tests/data/perf_natural.wav --fast
$PY live/duet_live.py run --ref live/cache/singer --input live/tests/data/perf_natural.wav --view
#    exercise the echo guard offline: the machine's voice leaks back at -6 dB through a simulated room
$PY live/duet_live.py run --ref live/cache/singer --input live/tests/data/perf_pause.wav --fast \
     --simulate-bleed -6 --out-latency 0.052            # add --no-echo-guard to watch it collapse

# 2b. ESP32-S3 over USB (needs the board + firmware). Sound check, then sing after the noise stops.
$PY live/duet_live.py run --ref live/cache/song --backend device --port /dev/cu.usbmodem1101 --view
#     no board? the firmware team's fake device on a pseudo-terminal:
$PY firmware/tools/fake_device.py --wav take.wav     # prints FAKE_DEVICE_PORT=/dev/ttysN; use it as --port

# 2c. MacBook mic + speaker (echo guard on; headphones still recommended)
$PY live/duet_live.py run --ref live/cache/song --backend mac --view --duration 120

# 2d. Logic Pro as the voice (LOGIC.md)
$PY live/duet_live.py run --ref live/cache/song --backend mac --midi-out --midi-latency 0.008

# 3. Tests, figures, demo assets
$PY live/tests/run_tests.py            # ~8 min (3 real-time runs, 3 fake-device runs, 1 MIDI run)
$PY live/tests/run_tests.py --quick    # ~3 min, skips the real-time / pty / MIDI runs
$PY live/tests/run_tests.py --only device midi    # selected sections; updates report.json
$PY live/tests/figures.py              # live/figures/*.png
$PY live/demo/make_demo.py             # live/demo/*.mid, *.wav, manifest.json
```

`$PY live/duet_live.py run -h` lists every option. The most important ones:

* `--out-latency L`: the time from the end of a mic block to when the output block made from it
  is heard. Defaults: 5 ms (file); measured from PortAudio timestamps (mac); **52 ms (device)**,
  which is a 4-frame jitter buffer + 10.65 ms of I2S DMA and filter (firmware/README.md) + USB.
  The device value is still an estimate. The sound check *measures* the speaker -> mic delay
  (printed and stored in `summary.json` under `backend.soundcheck.delay_ms`). On the board,
  L ≈ that delay − ~2 ms (acoustic path + ADC).
* `--spk-prefill N` (device, default 4): silent frames sent before the first answer = the speaker
  jitter-buffer depth. `--no-level-keeper` turns off the post-underrun correction.
* `--harmony learned|rule|cache`: harmony line (default learned; `cache` = the plan stored at
  rehearsal, which is what you get by default when `rehearse --plan` was used).
* `--echo-guard/--no-echo-guard`, `--echo-mode full|aec|mask`, `--soundcheck SEC` (default 2;
  0 = skip, which turns the canceller off), `--aec-update frozen|adaptive` (default frozen, see below).
* `--simulate-bleed DB` (file only), `--bleed-acoustic-ms`, `--bleed-rt60`.
* `--midi-out [NAME]`, `--midi-connect DEST`, `--midi-latency S`, `--midi-channel`, `--nod-cc`,
  `--synth auto|on|off`.
* `--nod first|phrases|off`, `--no-profile`, `--no-guard` (overlap guard), `--log-every N`.

Each run writes `live/runs/<name>/`: `duet.wav` (file mode) or `session.wav` (mac/device: mic as
received, voice as sent), `log.jsonl` (one record per 10 ms frame, with per-stage compute
`ft`/`echo`/`fol`/`acc`/`syn`/`tot` in ms, position, confidence, tempo, pitch, deadline slack;
plus one record per event: `note`, `held`, `missed`, `take_back`, `suppressed`, `servo`, `led`)
and `summary.json` (timing statistics, every scheduled note, decisions, nods, the plan used,
backend counters, sound-check result, echo-guard statistics, MIDI log).

## How it works

```
mic block k (240 samples, 10 ms)                     now = (k+1)·10 ms
  -> EchoGuard.cancel(mic, ref) echo canceller; ref = block sent after k-1   (echo/guard.py, optional)
  -> StreamGate                 input level gate, eval/duet_eval/features.py   (optional, --gate-db; default off)
  -> FeatureTracker.step        YIN pitch, voicing, loudness, novelty        (follow.py)
  -> EchoGuard.mask(ft, ...)    drop/re-estimate frames whose pitch is the machine's (optional)
  -> Follower.step              position on the reference: follower v2, a note-level HMM (follow.py)
  -> Accompanist.step(now, compute = 0)   anchors, tempo fit, predictive commits  (follow.py, v2)
  -> Engine._collect            each new decision 'play note j at time at' is scheduled
                                at output sample round(at·SR) in StreamingVoice
  -> NodPlanner / LED           servo and LED commands, re-timed every frame
  -> StreamingVoice.render      output block heard at [now + L, now + L + 10 ms)
  -> MidiOut (optional)         the same notes as note_on/off at wall(at) - midi latency
```

* **One clock.** Input block k ends at `now = (k+1)·HOP/SR`. The engine always renders the output
  block that will be heard at `[now + L, now + L + 10 ms)`. The Accompanist gets the same `L` as its
  `out_latency`, so when it commits a note at `at >= now + L`, the note can be written to the exact
  sample. Each note's **scheduling slack** is `at − (now + L)`. Processing time is hidden inside
  `L` as long as the block finishes before its deadline, and the backends measure that deadline.
  So `compute` is passed as a constant 0, which also makes runs deterministic: the real-time run
  makes exactly the same decisions as `--fast`.
* **Streaming synthesis** (`voice.py`) produces the same waveform as `dsp.render_voice`, sample for
  sample. Phase is carried across blocks. A note can be shortened after it has started (the next
  note arrived, or a false entry was taken back); its release then starts at the current render
  position, so there is no click.
* **The nod** (`engine.NodPlanner`) has three steps:
  1. **Lift** to 102° one beat before the predicted entry. The beat length is estimated from the
     rehearsal and scaled by her current tempo.
  2. **Drop** to 72° at 5°/10 ms, sent 100 ms before the entry. The 30° move takes 60 ms. The other
     ≈40 ms covers up to one 20 ms PWM frame plus servo response. SG90: 0.10–0.12 s/60° at 4.8 V,
     50 Hz PWM, per the datasheet below.
  3. **Return** to 90° 250 ms after the entry.

  The entry time is predicted again every frame until the drop is sent. There is no nod unless
  the follower is confident.
* **Rehearsal cache** (`rehearsal.py`) reproduces `run_v0.py` exactly:
  * detector delays are calibrated on the synthetic singer and subtracted
  * plan via `plan_harmony` (or `--plan`)
  * breaths via `find_gaps`
  * join point at the first attack after 25% of the song
  * beat length from the inter-onset intervals: the longest grid that fits ≥ 70% as well as the
    best one, with dotted multiples allowed. It recovered 60/72/84/100/120/140 bpm exactly on the
    synthetic melody. At 50 bpm it returned the half beat. `--beat` overrides it.
  * optional timing profile learned from a rehearsal take
* **Overlap guard.** After a pause, the Accompanist sometimes commits a stale predicted note at
  the same instant as the re-entry note. In the `pause` scenario this happened twice: j=8 and j=9
  at 19.455 s, then j=9 and j=10 at 20.185 s. The offline renderer plays both, as a two-note blip.
  The engine instead plays only the lower-index (re-entry) note when two decisions are less than
  60 ms apart, and logs a `suppressed` event. The Accompanist's decisions are not changed;
  `--no-guard` turns this off.
* **Threads.** The engine runs in the audio thread: the PortAudio callback for mac, or the main
  loop for file/device, set to `QOS_CLASS_USER_INTERACTIVE`. The JSONL writer and the HTTP view run
  in their own threads and only read engine state. `sys.setswitchinterval(1 ms)` limits how long a
  helper thread can hold the GIL.
* **Live view** (`viewserver.py` + `view/index.html`) uses only the standard library: one
  `ThreadingHTTPServer` on 127.0.0.1. The page polls `/state.json` at 15 Hz. It shows:
  * the score as a piano roll in reference time, with her pitch drawn where the follower places her
  * the harmony notes (filled once they have been sent)
  * confidence, tempo vs rehearsal, the next harmony note with a countdown, nod state, and compute per block

* **Echo guard placement.** `cancel(mic, ref)` runs before `FeatureTracker.step` and
  `mask(ft, expected)` right after it; `expected` = MIDI of the reference notes `acc.k-1 .. acc.k+1`.
  `ref` is the block returned by the previous `process()` call, which is exactly what every
  backend sends right after that mic block (file: the speaker timeline; mac: the PortAudio output
  buffer; device: the SPK_AUDIO frame). With MIDI out and the built-in voice muted, the
  canceller has no exact reference (Logic's timbre is different), so only the mask runs. It uses the
  muted built-in voice as its pitch reference, with delay L + 2 ms.
* **Sound check** (`echo_link.SoundCheck`, the same code in every backend). The backend sends
  2 s of pink noise (RMS 0.05 FS) one block per mic block and records both. It then sends
  silence while `guard.calibrate` runs (in a worker thread for mac/device; 36-63 ms measured),
  waits ≥ 0.3 s, and flushes the canceller's reference history. Only then does the engine start.
  The guard's GCC-PHAT delay estimate is the measured speaker -> mic delay.
* **Canceller frozen after the sound check** (`--aec-update frozen`, default). The canceller in
  `echo/guard.py` keeps adapting a background filter during the song and copies it over the
  output filter when it looks better. In the `pause` scenario at ≥ 0 dB bleed, that copy
  happened 3 times while the machine's tonal false entry was the only sound in the room, and
  the duet lost track (9/26 notes). With the filter frozen: 25-26/26 in every synthetic scenario
  up to +12 dB, and no regression elsewhere (table below). The trade-off: a mid-song change of
  the echo path is not tracked, so **repeat the sound check if the mic or speaker moves**
  (echo/README.md says the same). Implemented as `copy_ratio = 0` on the canceller instance;
  echo/ is unchanged.
* **USB link (device).** Codec = `firmware/tools/proto.py`. The host keeps one outgoing seq counter
  per frame type (the firmware counts gaps in the SPK_AUDIO seq) and counts incoming gaps per
  type: a MIC_AUDIO gap = mic frames lost, since the seq is assigned at capture. PONG is matched to
  its PING by seq. STATUS counters are wrapping u16, so underruns and overruns are reported as
  deltas over the song. Start-up order:
  1. Open the port without driving DTR/RTS. pyserial's two line updates are disabled for this
     port and HUPCL is cleared, so `close()` does not drop DTR either. On the S3's
     USB-Serial/JTAG these lines can reset the chip.
  2. Discard 0.2 s of frames (stale frames queued before the port was opened).
  3. Send 4 silent SPK frames (pre-fill), then answer every MIC frame with exactly one SPK frame.
     The device's mic and speaker clocks are locked, so the buffer level stays constant.

  Writes time out after 0.1 s and are counted, not retried. On exit the host sends servo-rest
  and LED-off, waits 50 ms, and **discards** unsent output (`tcflush`). It never calls
  `tcdrain`/`flush()`, which blocks forever if the device has stopped reading.
* **Level keeper (device).** After an underrun the device plays silence for U ticks and
  re-primes. When the host catches up, the buffer is U frames deeper: +U×10 ms of latency, and
  the echo path no longer matches what the canceller learned. The firmware trims 1 frame per
  0.5 s while the level is above 6 (`SPK_TRIM_ABOVE` = `SPK_PRIME_FRAMES` 4 + 2,
  `firmware/duet_io/config.h`); the firmware's fake device does not trim at all. The host reads
  `STATUS.spk_buffer_frames`. Once 3 reports in a row agree on a level that differs from the
  baseline, it drops (or inserts silent) SPK frames to restore the level. Transients are ignored.
* **MIDI clock.** `wall(at) = at + W0`, where W0 = sliding 2 s minimum of (arrival time of block
  k − now_k) − input latency. A scheduler thread sends each message `--midi-latency` before
  `wall(at)`: it sleeps until 2 ms before, then spins. The engine's L is raised to at least
  input latency + MIDI latency + 8 ms, so every note is decided before it must be sent.

## What was verified (live/tests/report.json, M4 MacBook, 2026-10-15)

The full suite was run once (`tests/last_run_full.txt`: 80 checks passed, 1 failed). Its
real-time section was then run a second time (`tests/last_run.txt`). Both are reported below.
Other agents kept the machine busy throughout (load average **8-12** on 10 cores). Every
real-time number here was measured under that load.

Inputs for the timing numbers: the synthetic singer and five `perform.make()` re-timings, whose
onsets are exactly known. **These are not real singers** (see "Real recordings" below).

| check | result |
|---|---|
| Protocol: live/ uses `firmware/tools/proto.py`; 300 random frames with garbage and corruption | 286/286 good frames recovered, 0 corrupt frames delivered. Per-type seq counters and per-type gap counting checked, including the wrap 255 -> 0 and interleaved STATUS/PONG |
| Serial open/close (pty, every `ioctl` recorded) | 0 DTR/RTS ioctls (stock pyserial: 1 on the same pty, the control case); HUPCL cleared |
| `StreamingVoice` vs `dsp.render_voice` | max difference 1.5e-8 |
| Live engine decisions == the offline `run_v0` loop (learned plan, compute 0) | identical on all 6 inputs |
| Output wav: notes where decided | each onset within **5 samples** (0.2 ms) of the decided time; pitch within 0.04 st; machine channel = offline re-render to ≤ 1.35 LSB |
| Timing vs ground truth (file mode, **learned harmony**) | median \|async\|: reference 3.7 ms · natural **10.8** · pause 10.9 · skip 5.5 · noisy 9.3 · expressive 35.5 (**14.2** with the rehearsal profile). Identical to phase 1 with the rule plan: the plan changes pitches, not timing |
| Learned vs rule harmony | same 37 melody notes; 35/37 harmony pitches differ; A minor; 2.6-14 ms to plan; harmony 3-12 semitones below; falls back to the rule plan if the model cannot load |
| Output latency sweep L = 5…120 ms | predictive stays at 10.8 ms (natural); reactive grows 1:1 (49 -> 164 ms). `figures/latency_sweep.png` |
| Speed, `--fast` | real-time factor **0.06** (~16× faster than real time; phase 1 measured 0.03 on a quieter machine); per-block p50 **0.55 ms**, p99 1.0-1.5 ms |
| Echo guard cost | **0.31-0.39 ms** mean per block (canceller + mask); p99.9 1.7 ms |
| Cost vs song length | 34 s song: p50 0.46 ms · 137 s: 0.90 ms · 274 s: 1.51 ms per block |

**Echo guard vs simulated bleed** (file mode, `echo/sim.py` room: 2 ms acoustic delay, RT60 0.3 s,
250 Hz speaker roll-off; L = 52 ms like the device; 2 s sound check; learned harmony;
harmony notes sung of 26, median |async| vs truth). `figures/bleed_guard.png`:

| bleed at the mic (re her voice) | none | −18 dB | −12 dB | −6 dB | 0 dB | +6 dB |
|---|---|---|---|---|---|---|
| natural, no guard | 26 · 10.8 ms | 24 · 15.4 | 7 · 74.7 | 1 | 1 | 5 |
| natural, guard | — | 26 · 10.8 | 26 · 10.8 | 26 · 10.7 | 26 · 10.1 | 26 · 10.8 |
| pause, no guard | 26 · 10.9 ms | 9 | 7 | 1 | 1 | 5 |
| pause, guard (frozen canceller, default) | — | 26 · 10.9 | 26 · 10.9 | 26 · 10.9 | **25** · 10.8 | 26 · 11.2 |
| pause, guard (`--aec-update adaptive`, phase-1 echo/ behaviour) | — | | | | **9** | **9** |

No wrong notes in any run. A diagnostic sweep (scratch script, not part of the suite) with the
frozen canceller covered all five scenarios at −12…+12 dB: ≥ 25/26 notes everywhere (skip: 12
of the 12 notes available with bleed, vs 13 without). With the **rule** harmony at L = 52 ms,
`pause` at 0 dB still dropped to 16/26. So the tolerance depends on the harmony line and
on L. One synthetic song, 26 notes per run: each threshold is only good to about ±3 dB.

**Real time** (file mode paced at 10 ms, 35 s, 3502 blocks, QoS user-interactive; "done" =
block finished after the moment it was complete; a 4-frame device buffer glitches only above 40 ms):

| run | compute p50 / p99.9 / max (ms) | done p50 / p99.9 / max (ms) | blocks > 20 ms / > 40 ms | RTF |
|---|---|---|---|---|
| run 1, no view | 0.80 / 2.0 / 4.7 | 1.1 / 4.4 / 6.1 | 0 / 0 | 1.00003 |
| run 1, view polled 20 Hz (563 polls, 0 failed) | 0.81 / 2.5 / 4.5 | 1.2 / 7.4 / 20.7 | 1 / 0 | 1.00003 |
| run 1, echo guard + −6 dB bleed, L = 52 ms | 0.81 / **10.2** / 22.7 | 1.6 / 15.0 / 27.9 | 1 / 0 | 1.00002 |
| run 2, no view | 0.71 / 5.6 / 29.5 | 1.1 / 15.7 / 34.9 | 3 / 0 | 1.00003 |
| run 2, view polled | 0.82 / 3.3 / 10.7 | 1.2 / 8.1 / 18.3 | 0 / 0 | 1.00003 |
| run 2, echo guard + −6 dB bleed | 1.01 / 8.4 / 26.0 | 1.8 / **53.9 / 61.9** | 17 / **12** | 1.00003 |

**Worst cases.** The worst single block took 29.5 ms of compute (run 2, no view). The worst
block finished 61.9 ms after it was complete (run 2, guard). That run's log shows the process
**woke up 52 ms late** (`wake_late_ms`): a stall in which macOS did not schedule the process at
all, not slow compute. The outliers are spread over every stage (the follower, the
FeatureTracker, the guard), which is what preemption looks like. With guard, p50 compute is
0.8-1.0 ms, i.e. ~10 % of the 10 ms budget. Two checks failed for this reason: run 1
"guard p99.9 < 10 ms" (10.15 ms) and run 2 "guard: no block > 40 ms" (12 blocks). On the
device, such a stall causes underruns; the level keeper then repairs the buffer (next table).
The decisions were identical to `--fast` in the no-guard real-time runs, and all 26 notes were
sung in the guard runs. **Re-measure on an idle machine before the demo.**

**Device backend vs the firmware team's fake ESP32** (`firmware/tools/fake_device.py` in its own
process, pty, real-time pace, 4-frame pre-fill, L = 52 ms):

| check | result |
|---|---|
| Same engine run offline on the mic frames the host received | output **bit-identical** (3830 frames) |
| What the fake played vs what the host sent | **918 480 / 918 480 samples equal**; the last 3 frames were still in the buffer when the fake's wav ended |
| Pacing and link | 1:1 (3830 answered); 21 stale frames discarded at start; 0 mic frames lost after start-up; fake: 0 SPK seq gaps, 0 underruns, 0 overflows; host: 0 checksum/header errors |
| Speaker buffer level (STATUS) | 3 frames after each pop in all 383 reports (= 4 before the pop; the firmware trims only above 6) |
| Nod, PING | SERVO 102 / 72 / 90 reached the device; PONG matched by seq, RTT median 0.06 ms (pty, not USB) |
| Echo through the fake (`--echo-gain 1.0`, 2 ms) + a forced 80 ms host stall mid-song, **guard on** | sound check: delay **32.0 ms** (expected 3 frames + 2 ms on the fake), ERLE 34.4 dB. The stall caused 1 underrun and the level went 3 -> 7; the keeper dropped 4 frames 17.7 s in. **26/26 notes** |
| Same, **guard off** | 3/26 notes (the duet collapses) |

Before the keeper existed, the same kind of event (a real load stall, gain 1 echo, guard on)
left the level at 6 for the rest of the song. The canceller was then 30 ms off, the mask had to
work on 1930 frames, and 24/26 notes were sung. An earlier run under the same conditions sang
3/26; its cause was not recorded (that summary was overwritten).

**Mac backend, callback driven with synthetic PortAudio timestamps:** L measured as 12.50 ms
(the value fed in); output identical to file mode. With the default guard and the speaker
leaking into the mic at −6 dB, the callback ran the sound check (noise out, calibration in a
worker thread, measured delay 14.5 ms = 12.5 ms + 2 ms acoustic, ERLE 30.3 dB): 26/26 notes.
Without the guard: 1/26.

**MIDI out**, checked by a **second process** that owns a virtual MIDI input port
(`tests/midi_sink.py`). Both processes use `time.perf_counter()`, which is mach_absolute_time
and system-wide. Real-time file run, 35 s, `--midi-latency 0.010`:

* 26/26 note_ons received, in order, with the right pitches;
* receive time − scheduled time: median **0.05-0.08 ms**, max **1.08 / 1.78 ms** (two runs),
  within ±2 ms;
* scheduled time = the built-in note's stream time + one constant clock offset − MIDI latency, to
  within 0.000 ms for every note;
* note_offs balance the note_ons; the nod arrives as CC 20 = 72 / 51 / 64.

The virtual source "Duet" was created and then seen and opened by another process. **No
message was ever sent through it**, because Logic Pro was running during the tests and listens
to every MIDI source. The timing test therefore sends to a private destination instead.

**MIDI files:** start times survive a write/read round trip to within half a tick (0.37 ms at
84 bpm), and the files render with the built-in voice.

## Input gate and detector bias (the evaluation's front end, optional)

Both are **off by default**; nothing changes for a cache rehearsed without them.

* `rehearse --gate-db X` fixes the evaluation's level gate at rehearsal time: threshold = 90th
  percentile of the reference's 10 ms frame levels − X dB, 6 dB soft knee, causal per frame. The
  reference and `--take` are analysed after the gate, and the threshold is stored in the cache.
  `run` gates the live input with it (after the echo canceller, before the pitch tracker). The code
  is `eval/duet_eval/features.py` (`gate_threshold`, `apply_gate`, `StreamGate`), imported, not copied.
  `run --gate-db X` sets X for one run (threshold from the stored reference level); `--gate-db off`
  disables a stored gate.
* `rehearse --bias A,G,R` (seconds; attack, glide, repeat) or `--bias dcs` / `--bias ecd` (the
  evaluation's per-dataset values in `eval/results/v2_frozen.json`) replaces the synthetic-singer
  detector calibration, also for the `--take` profile. `run --bias` overrides it for one run.
* **Why.** On a choir close mic the other singers' bleed is all the mic hears in her rests. On DCS
  Tebe Poem S1 (rehearse Take01 with its annotated notes, perform Take02, `--fast --out-latency 0.052`)
  the live app gives 2.4 % of plan notes on time (±60 ms) with the defaults, 14.3 % with
  `--gate-db 15`, and 23.8 % with `--gate-db 15 --bias dcs` (the frozen evaluation: 24.3 % for this
  take). These are the `media/demo` numbers, reproduced with the CLI alone (live tests section `gate`).
  Without the gate, voiced bleed an octave below her next note (≈ G3 against her G4) at 27.4 s moves
  the follower from inside her rest to that note, 5 s early. The Accompanist marks three plan notes
  missed, and at 32.1 s (her re-entry is at 32.55 s) confirms an anchor on a reference note she sings
  3.4 s later. It confirms no other anchor until 62.9 s, so it sings ~26 notes about 3 s early. The
  follower itself is back on position by ~36 s; the stale anchor, not the follower, keeps Duet early.
* **Solo singer, quiet room: leave the gate off.** On the synthetic solo singer the gate changes
  nothing that matters: off / 30 / 25 / 20 / 15 dB give natural 20.9 / 22.3 / 21.1 / 20.9 / 21.3 ms
  median |async|, all five scenarios within 5 ms of no gate, the same notes played, 0 wrong. The threshold is absolute, taken from the rehearsal recording,
  so if the live mic chain is quieter than the rehearsal recording a narrow gate fades her out. If you
  want one against room noise, record the rehearsal through the same mic and gain and use a wide gate
  (25–30 dB). Use 15 dB (the evaluated value) when other voices reach the mic. No bias has been
  calibrated for a solo singer yet (GAPS.md item 4).

## Real recordings (important)

**Update (2026-10-16).** The table below is phase 2 (follower v1). With follower v2 and the
evaluation's front end, see the section above and `media/demo/` (the v2 demos); the `live/demo/`
assets below are v1.

Phase 1 tested only the synthetic singer. In phase 2, the demo assets made me run the engine
(follow.py, **unchanged**) on real choir takes: rehearse on take 1, perform take 2, file mode.
It mostly does **not** work yet:

| take (rehearsal -> performance) | harmony notes sung |
|---|---|
| ECD Der Greis S1, `offline_notes` reference | 0 / 243 (follower jumped to the end of the song after 20 s) |
| ECD Seele Christi S1 / S2, `offline_notes` reference | 0 / 96, 1 / 102 |
| DCS Locus Iste full choir S1, `offline_notes` reference | 1 / 99 |
| same five takes with **annotated reference notes** (`rehearse --notes`) | DCS QuartetA S2 9/85, QuartetB S1 4/85, FullChoir S1 16/85, ECD DG S2 4/111, SC1 S2 2/32 |

Two causes are visible:

1. **The reference notes.** `offline_notes` over-segments real singing (236 detected vs 54
   annotated notes on an ECD soprano take, from vibrato and consonants). `find_gaps` turns
   every consonant into a "breath" (49 in a 75 s take), so every note becomes a phrase entry.
   `rehearse --notes` fixes this part: annotated notes, and breaths = rests ≥ 0.25 s between them.
2. **The Follower** (online DTW in follow.py) loses real takes. It resets 3-20 times per take
   and jumps to wrong places. `Accompanist.max_r` only ever grows, so a single jump ahead marks
   every earlier plan note "missed" for the rest of the song.

In the DCS demo, the follower locked the opening of Take02 onto the **recapitulation** of the
rehearsal (the same music, 84 s later). The machine harmonised the first 30 s with the plan of
the recapitulation: 15/16 melody pitches agree, timing ±111 ms median to the nearest onset. It
then ran out of reference and stopped. This belongs to the follow.py owner and the evaluation
phase, not live/. I report it because it is the biggest gap between the demo and a real singer.

## Demo assets (`live/demo/`, 33 MB, regenerate with `demo/make_demo.py`)

| file | content |
|---|---|
| `greensleeves_natural.mid` | synthetic Greensleeves, natural performance: "Singer (melody)" = exact notes re-timed to the performance; "Duet (harmony)" = the 26 notes Duet sang (learned harmony, median 10.8 ms from her true onsets), nod as CC 20; 83.3 bpm; positions in real seconds |
| `greensleeves_natural_duet.wav` | the run: L = singer, R = Duet's built-in voice (35 s) |
| `greensleeves_natural_midi_render.wav` | the .mid rendered back with the built-in voice (L melody, R harmony) |
| `dcs_locus_iste_S1_take02.mid` / `_duet.wav` / `_midi_render.wav` | the real DCS take above: melody = DCS score-aligned notes of Take02 (113), harmony = the 16 notes Duet sang (the recapitulation lock), 85.3 bpm. **Dagstuhl ChoirSet, CC BY-NC 4.0**: non-commercial use, cite the dataset |
| `manifest.json` | sources, counts, tempo, timing, MIDI round-trip error, what happened |

Nobody has listened to any of these wav files (no audio was played).

## What is NOT verified

* **No real microphone, speaker, ESP32, servo or USB link was used.** Not measured: the
  MacBook round-trip latency; whether PortAudio accepts 24 kHz / 240-frame blocks on the
  built-in devices; real USB-CDC timing; whether opening the port resets the S3 (the host now
  never drives DTR/RTS, but what the macOS driver does on `open()` is not under our control);
  the real speaker -> mic delay (the 52 ms default is still the firmware's estimate plus USB).
* **Device mode was tested only against fake devices.** The firmware team's fake has no trim
  logic and samples STATUS after the pop. The real firmware's STATUS level is sampled
  asynchronously, so it may read 3 or 4. The level keeper uses the minimum over its baseline
  window and needs 3 equal reports before it acts, but this has not run on silicon.
* **Echo guard:** simulated rooms and the fake's pure-delay echo only. No real room, no real
  speaker distortion, no real singer. The frozen-canceller default does not track a mid-song
  path change.
* **Logic Pro was never opened.** LOGIC.md follows Logic 10.7-11 menus and has not been checked
  in this version. The MIDI timing was measured into a private test port, not into Logic's
  audio output. Logic's own output latency and the patch's attack must be measured by ear (LOGIC.md).
* **Real-time numbers come from a loaded shared machine** (see above).
* **Accuracy on real singers is poor** (see "Real recordings"). All good timing numbers come
  from the synthetic singer.
* The live view page was never opened in a browser. The beat estimator is a heuristic.

## Proposed changes / notes for other owners

1. **echo/** (guard.py): after the sound check, do not copy the background filter over the
   foreground during the song (live/ sets `copy_ratio = 0`; data above). Better still: allow
   a copy only when the reference is broadband, never on a tonal machine note.
2. **firmware/** (no edits made there):
   * `tools/fake_device.py` does not implement the firmware's trim (`SPK_TRIM_ABOVE`/`AFTER`).
     After an underrun its latency stays high forever, unlike the firmware's.
   * Please confirm the semantics of `STATUS.spk_buffer_frames` on the board (sampled before
     or after a pop). The level keeper only needs it to be consistent.
   * The host pre-fills **4 frames**, the same as the firmware's prime (`SPK_PRIME_FRAMES 4`,
     `firmware/duet_io/config.h`): playback starts at 4 and the level settles at 4, below the trim
     threshold (the trim acts only above 6 = prime + 2).
   * No firmware bug was found: the codec, per-type seq, PONG seq and STATUS layout all matched.
3. **follow.py**: (a) the phase-1 proposal still stands (a stale prediction committed together
   with the re-entry note; live/'s overlap guard covers it). (b) Real takes: `max_r` should not
   jump on a follower move that is not confirmed by anchors. Repeated sections (recapitulations)
   need a position prior. `offline_notes`/`find_gaps` over-segment real voices. Use
   annotated or score notes (`rehearse --notes`).
4. **PLAN.md**: device→host seq is per type, the host pre-fills 4 frames, and the device-mode
   L is 52 ms.

## Files

| file | what |
|---|---|
| `duet_live.py` | CLI: `rehearse`, `run`, `devices` |
| `engine.py` | `Engine` (per-block pipeline, echo guard hook, MIDI hook, instrumentation), `NodPlanner`, `JsonlLogger` |
| `voice.py` | `StreamingVoice` (block-by-block `render_voice`) |
| `rehearsal.py` | reference analysis + cache, detector calibration, beat estimate, `--notes`, `choose_plan` (learned / rule harmony) |
| `backends.py` | `FileBackend` (+ simulated bleed), `MacBackend`, `DeviceBackend` (+ `LevelKeeper`, `open_serial_no_toggle`) |
| `proto.py` | thin layer over `firmware/tools/proto.py`: payload helpers, per-type `SeqCounters` / `SeqTracker` |
| `echo_link.py` | echo/ glue: `make_guard`, `SoundCheck`, `BleedSim` (echo/sim.py room) |
| `midi.py` | `MidiOut` (virtual port, scheduler, clock), `export_midi`, `read_midi_notes`, `render_midi` |
| `LOGIC.md` | Logic Pro as the voice; rendering the demo MIDI files |
| `viewserver.py`, `view/index.html` | live view |
| `tests/run_tests.py`, `tests/midi_sink.py`, `tests/make_inputs.py`, `tests/figures.py`, `tests/fake_device.py` | tests (`fake_device.py` = live's own minimal fake, kept for lockstep experiments; the suite uses the firmware's) |
| `tests/report.json`, `tests/last_run_full.txt`, `tests/last_run.txt` | measured numbers and console logs |
| `figures/` | `latency_sweep.png`, `timeline_natural.png`, `frame_time.png`, `bleed_guard.png` |
| `demo/` | demo MIDI + wav files, `make_demo.py`, `manifest.json` |
| `cache/`, `runs/`, `tests/data/` | generated (regenerable) |

## Sources (opened)

* python-sounddevice 0.5.1 Streams API: https://python-sounddevice.readthedocs.io/en/0.5.1/api/streams.html
* Apple Energy Efficiency Guide, QoS classes: https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheTaskLevel.html
* SG90 datasheet compilation: https://www.friendlywire.com/projects/ne555-servo-safe/SG90-datasheet.pdf
* pyserial 3.5 source (`serial/serialposix.py`: `open()` drives DTR then RTS unless dsrdtr/rtscts), read in the venv
* python-rtmidi 1.5.8 / mido 1.3.3 (virtual ports, `MidiIn` callbacks), used through their installed APIs
* firmware/README.md (protocol semantics, jitter buffer, latency budget), echo/README.md, harmony/README.md, data/README.md
