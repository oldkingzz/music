# Duet

Duet is a small robot that sings a harmony line with a live singer. It does not wait to hear each of her
notes, which is always late. It follows where she is in a known song, predicts when her next note will
start, and sends its own note early enough to sound on time. Before it joins in, it nods.

MEAM/IPD 5160 Advanced Mechatronics (Prof. Mark Yim), Penn, Fall 2026. Final project, team of 4.

## How it works, in one picture

```text
singer -> mic + preamp + filter -> PCM1808 ADC -> ESP32-S3 ==USB==> MacBook: listen -> follow -> predict -> harmony
                                                   ESP32-S3 <==USB== MacBook
          speaker <- MAX98357A amp <-------------- ESP32-S3 --------> servo (nod)
```

- **ESP32-S3** (`firmware/`): moves audio both ways in 10 ms frames with a small, constant latency, and
  drives the servo and LED.
- **MacBook** (`live/`, `follow.py`): finds her position in the song with a hidden Markov model, predicts
  her next onset with phase correction on a slowly adapting tempo, plans the harmony at rehearsal, and
  schedules every note to the exact output sample, early by the output latency.
- Course requirements: sensing (mic), actuation (speaker and nod servo; 2 actuators), computation
  (ESP32-S3 and Mac), and something new (predictive score following on real singing).

The full story, with the numbers and the weak points: [docs/guide/explanation.md](docs/guide/explanation.md).

## Status (2026-10-16)

| Part | State | Evidence |
|---|---|---|
| Software (follow, predict, harmony, live app) | works on recordings; never run with a live mic | `live/README.md`, `docs/VERIFICATION.md` |
| Real-recording evaluation | done, pre-registered, held-out | `eval/RESULTS.md` |
| Firmware | compiles (0 warnings), host tests pass; never run on a board | `firmware/README.md` |
| Schematic | done; ERC clean; pin map checked 53/53 | `hardware/README.md`, `hardware/REVIEW.md` |
| PCB layout | work in progress: DRC fails, no Gerbers | `hardware/LAYOUT.md` |
| Mechanical design | CAD, STL and DXF done; some dimensions still to measure | `mech/README.md` |
| Built, measured, heard | **not yet**: no hardware built, latency not measured, nobody has listened | `docs/GAPS.md` |

What is missing for the final demo, ranked: [docs/GAPS.md](docs/GAPS.md).

Key dates (course page): pitch week of 10/19, proposal week of 10/26, integration review week of 11/30,
last class 12/7.

## Key numbers

| | Value | Source |
|---|---|---|
| On time (within 60 ms), fresh held-out choir piece, real system | **25.9 %** [19.3, 33.4] (v1: 2.1 %) | `eval/RESULTS.md` v2.3 |
| Same, reused piece with the human baseline | **33.4 %** [29.8, 37.4] (v1: 7.3 %) | `eval/RESULTS.md` v2.3 |
| Predicting vs reacting, same follower | +6 and +14 points on time | `eval/RESULTS.md` v2.3 |
| Distance from a human partner (median \|a\|) | 92 vs 52 ms: **+40 ms** [26, 53]; not yet human-level | `eval/RESULTS.md` v2.4 |
| Largest remaining loss | hearing her onsets late: about 26 points | `eval/RESULTS.md` v2.0 |
| Synthetic singer (smoke test, not evaluation) | median 21 ms, 96 % within 60 ms | `run_v0.py` |
| Output latency on the board | about 48–52 ms, **estimated, not measured** | `firmware/README.md` §2 |
| Parts cost | PCB parts $38.20 per board, $63.92 cash with minimum orders; prototype $53.32 + ADC module; PCB fab not yet quoted (2026-10-16, after the speaker/mic change) | `hardware/README.md` |

## Start here

New to the project? Do the tutorial first. It needs no hardware, and takes about 15 minutes.

```bash
git clone <repository> duet && cd duet
python3.11 -m venv .venv && .venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python live/tests/make_inputs.py
.venv/bin/python live/duet_live.py rehearse live/tests/data/singer.wav -o live/cache/singer
.venv/bin/python live/duet_live.py run --ref live/cache/singer --input live/tests/data/perf_natural.wav --fast --name first
.venv/bin/python docs/guide/tools/run_report.py live/runs/first --truth natural
```

The last command writes `live/runs/first/duet.mid` and `timeline.png` next to `duet.wav` (her voice left,
Duet right) and prints Duet's timing: median 20.9 ms from her true onsets.

| I want to | Read |
|---|---|
| learn the project by doing | [docs/guide/tutorial.md](docs/guide/tutorial.md) |
| do one specific job (record the singer, flash the board, measure latency, Logic Pro, regenerate the schematic or figures, run the tests, get the datasets) | [docs/guide/how-to/](docs/guide/how-to/) |
| look up a command, a flag, the pin map, the USB frames or a folder | [docs/guide/reference.md](docs/guide/reference.md) |
| understand why it works and where it is weak, to present it | [docs/guide/explanation.md](docs/guide/explanation.md) |
| see the interfaces every part must respect | [docs/PLAN.md](docs/PLAN.md) |
| see why each design choice was made | [docs/decisions/](docs/decisions/) |
| see the last full test run | [docs/VERIFICATION.md](docs/VERIFICATION.md) |

## Repository

| Folder | What |
|---|---|
| `follow.py`, `dsp.py` | the core: listening, following (HMM), predicting; signal processing and the built-in voice |
| `live/` | the real-time app on the MacBook (file, Mac and board backends, MIDI out, live view) |
| `firmware/` | ESP32-S3 I/O firmware, its host tests, the USB codec and a fake board |
| `hardware/` | KiCad schematic (generated by script), BOMs, prototype wiring, PCB (in progress) |
| `mech/` | mechanical design from one parameter file: STL, DXF, renders |
| `harmony/` | learned harmony line (a CRF trained on Bach chorales) |
| `echo/` | the robot hearing itself: simulation and the echo guard |
| `eval/` | evaluation on real recordings: protocols, harness, results |
| `data/` | dataset loaders, download scripts, sanity checks (the datasets themselves are not in git) |
| `media/` | deck figures, visuals and demo audio/MIDI |
| `docs/` | plan, gaps, verification, methods, decisions, and this guide |
| `singer.py`, `perform.py`, `run_v0.py` | synthetic singer and the first prototype's test harness |

Every top-level entry, one line each: [reference §8](docs/guide/reference.md#8-repository-map).

## Rules for working in this repository

- **Python:** `.venv/bin/python` (3.11), installed from `requirements-lock.txt`.
- **Datasets are licensed and not in git** (Dagstuhl ChoirSet CC BY-NC 4.0; ESMUC Choir Dataset and
  vocadito CC BY 4.0). Each person downloads them ([how-to](docs/guide/how-to/get-the-datasets.md)); never
  commit or re-share them. `*.wav` is git-ignored for this reason.
- **Recordings of our singer** need her consent and stay on our machines.
- **Never delete** project files; move them to the Trash instead.
- **Hardware:** do not power the speaker, open the mic, move the servo or flash a board without the
  board owner's go-ahead.
- Check free disk (`df -h .`) before anything that writes a lot (datasets, evaluation runs, the ESP32
  toolchain).
