# Duet futures: shared brief for every agent (read fully before starting)

Five deliverables ("玩法"), built in parallel by five builder agents, then checked by independent
verifier agents. Vincent will read only the results. The acceptance lines below are the contract;
do not loosen them, and report a FAIL as a FAIL.

Working directory for everything: `/Users/wangzesen/work/claude_sandbox/MEAM5160/project/duet_v0`
Python: `.venv/bin/python` (3.11; numpy 2.4, scipy 1.17, librosa 0.11, soundfile, numba, music21 10.5,
mido, torch 2.15 with MPS, sounddevice). Run every command from the working directory.

## Hard clock (America/New_York, today 2026-10-30; check with `date`)

| | builders | verifiers |
|---|---|---|
| first complete results.json written | 11:25 | |
| final results.json + REPORT.md | 11:50 | 12:10 |
| absolute stop (no more commands) | 11:55 | 12:12 |

- Write a usable, honest result EARLY (by 11:25), then improve it. A late perfect result is worth nothing.
- No single shell command may run longer than 8 minutes. Never leave background processes running
  when you finish. Kill anything you started that is still running.
- Heartbeat: every time you finish a step (at least every 8 minutes), overwrite
  `futures/status/<id>.json` with `{"id": "<id>", "updated": "<date -Iseconds>", "stage": "...",
  "done": false, "note": "one line"}`. Set `"done": true` as your very last action. A watchdog flags
  any agent whose heartbeat is older than 10 minutes, and the lead will stop it.

## Rules (all hard)

1. Write ONLY inside your own folder `futures/<your folder>/` (plus your status and results files,
   and `futures/common/` if you own it). NEVER edit, move or delete any existing file in the repo.
   To change existing code, copy it into your folder and edit the copy. Do not delete anything you
   did not create in this session; to discard your own scratch files, just leave them.
2. No microphone, no speaker: never open an audio device (no sounddevice streams, no afplay/say
   playback). Everything is file-based and simulated. If a deliverable needs a real hardware
   measurement, write the script and an exact step-by-step test sequence for Vincent instead
   (`HARDWARE_TEST.md` in your folder); he runs it after saying 开始.
3. Licences:
   - Dagstuhl ChoirSet (`data/external/dcs`, CC BY-NC) and ESMUC (`data/external/ecd`): allowed for
     local evaluation and local listening only. Any audio derived from them goes in
     `<your folder>/local_only/`. Never in a folder meant for pages.
   - vocadito (`data/external/vocadito`, CC BY 4.0, real solo singers): allowed in demos with
     attribution "vocadito (Bittner et al.), CC BY 4.0".
   - Hooktheory (`duet_beat/bench/data/pop.json`): training and scoring only. Never render its
     melodies to audio, never write its titles or artists anywhere.
   - Bach chorales (public domain), Nottingham folk (GPL), POP909 (MIT), synthetic singer: free to use.
   - Page-safe demo audio goes in `<your folder>/demo/` as mp3 (lame is at /opt/homebrew/bin/lame;
     ffmpeg at /opt/homebrew/bin/ffmpeg), each <= 1.5 MB, with a `demo/SOURCES.md` line per file
     (source, licence).
4. Disk: 11 GB free. Keep your folder under 500 MB. No checkpoint-every-step loops. Downloads are
   allowed only if `df -h /` shows more than 1 GB free; prefer what is already installed.
5. pip: never change the version of an already-installed package. If a new package would pull an
   upgrade/downgrade, do not install it (use `pip install --dry-run` first). Small pure additions
   (e.g. pyworld) are fine if they install cleanly.
6. Tune on validation data only. Keep a test split untouched until one final run, and say so.
7. Compute must run on the target board (below). Use numpy/scipy (torch only for training; export
   weights to numpy for inference if you train a net). No GPU at inference.

## Target hardware: Radxa Cubie A7S

Allwinner A733: 2x Cortex-A76 + 6x Cortex-A55, up to 2.0 GHz; PowerVR BXM-4-64 GPU; NPU 3 TOPS INT8;
6 GB LPDDR5; Linux aarch64; 5 V USB-C power. Audio: assume a USB audio interface for mic and speaker
(on-board audio is not documented). One microphone and one speaker.

Board speed estimate (use exactly this so all five agree): run the timed code single-threaded on an
M4 efficiency core:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 taskpolicy -c background .venv/bin/python ...`
then report `board_estimate = 2.0 x E-core time` and label it as an estimate. Also report the plain
M4 time.

## Shared definitions (use these exactly; they match the page Vincent approved)

- Interval class between two notes: `abs(a - b) % 12`.
- good: interval class in {0,3,4,7,8,9}. harsh: {1,2,6,10,11}. fourth: {5}.
- "好听" (good share): time-weighted share of HER singing time where Duet's note is good against her
  TRUE note; Duet silent counts as not good. This is the 0.79 metric of the realtime leaderboard
  (`duet_beat/bench.py`, `interval_shares`).
- "不冲突" (no-clash share) for a multi-voice harmony: time-weighted share of her singing time where
  NONE of Duet's sounding voices is harsh against her true note (fourths allowed). Report it per voice
  too, and report Bach's own value on the same material as a reference.
- Parallel 5ths / 8ves: between any two voices including hers, consecutive slices where both voices
  move and the interval class stays 7 (or 0) in the same direction of motion.
- For chord accompaniment with 7ths/tensions (deliverable 4 pop/jazz): "clash" = her note's pitch
  class is a semitone ABOVE any sounding chord tone (the jazz avoid-note rule, a minor 9th). Report
  the harsh-interval version as well.

## Existing assets (read before writing new code; reuse instead of re-inventing)

- `README.md`, `docs/guide/explanation.md`: what Duet already is (a score-following duet robot).
- `dsp.py`: SR = 24 kHz, HOP 10 ms, YIN pitch, `wsola`, `render_voice` (built-in machine voice).
- `singer.py`: synthetic singer (vowel 'ah', glides, vibrato, breaths); `score_notes`, `render`.
- `follow.py`: `analyse` (features), `offline_notes` (audio -> notes), `detect_key`, `plan_harmony`
  (rule: diatonic 3rd/6th below), `align_score`, `Follower` (HMM score follower), `Accompanist`
  (predicts her next onset and schedules Duet's note early by the output latency).
- `live/`: the real-time app. `live/duet_live.py rehearse ...` then `run --ref ... --input ... --fast`;
  `live/rehearsal.py` (Reference, estimate_beat), `live/voice.py` (StreamingVoice), `live/engine.py`.
  Tutorial commands are in `README.md` "Start here".
- `harmony/`: a CRF trained on 243 Bach chorales that writes ONE harmony line (alto);
  `harmony/data/bach_satb.json` has soprano-onset SATB slices:
  `[onset_beat, dur_beats, S, flag, A, T, B, chord_pcs]` per slice, with tonic/mode per chorale;
  `harmony/data/split.json` is a tune-grouped train/val/test split. `harmony/eval.md` has its numbers.
- `echo/`: closed-loop simulation of the speaker leaking into the mic; `echo/guard.py` has an
  `EchoCanceller` (MDF frequency-domain NLMS) and a `PitchMask`; `echo/physical.py` mic-placement
  acoustics.
- `eval/`: pre-registered evaluation on real choir takes (take 1 = rehearsal, later takes = test);
  `eval/RESULTS.md`; `data/loaders.py` (`split_takes`), `data/MANIFEST.md`.
- `duet_beat/`: the realtime chord benchmark (bench.py, leaderboard 0.793 for `policy`);
  `duet_beat/bench/data/{folk,pop,pop909}.json` (songs with per-beat chord labels `z`, 7 diatonic
  triads, train/val/test), `duet_beat/model.py` ChordHMM, `duet_beat/theory.py`.
- Real singing: `data/external/vocadito` (40 solo excerpts, note annotations in
  `Annotations/Notes/*_notesA1.csv` as onset_s, f0_hz, dur_s), ECD, DCS (see MANIFEST).

## Deliverables and their acceptance lines (verbatim from the approved page)

D1 录一遍，配一遍 (record then harmonize): she sings 4-8 bars; a few seconds later the speaker plays her
original voice plus Duet's synthesized alto, tenor and bass: a four-part choir.
  - A1.1 From the end of her singing to playback start <= 5 s, all computed on the board.
  - A1.2 No-clash share >= 95%, and zero parallel 5ths / 8ves.
  - A1.3 Blind listening: Duet's harmonization mixed with Bach's real one, listeners cannot tell them
    apart close to half the time. (Needs humans: build the blind-test kit; give automatic proxies.)
  - Deliverable: one-command demo + 3 demo recordings.

D2 她自己的合唱团 (her own choir): the harmony is her own recording pitch-shifted into three voices.
  - A2.1 At shifts of up to 7 semitones listeners hear "the same person", not a machine. (Human; give
    proxies such as formant preservation, and A/B files.)
  - A2.2 The three voices use the same harmony as D1 and pass the same no-clash standard.
  - A2.3 A 30 s recording is processed within 30 s on the board.
  - Deliverable: demo + original / choir comparison recordings.

D3 第二遍就会了 (rehearsal mode): first take Duet only listens; on the second take Duet already knows
what she sings next and sings with her in real time.
  - A3.1 Good share on the second take >= 95% (the realtime guessing system is at 79%).
  - A3.2 She sings 20% faster or 20% slower: Duet keeps up, no lost beat.
  - A3.3 She stops for 2 s and resumes: Duet is back in sync within one beat.
  - Deliverable: live-demo fallback + a two-take comparison recording.

D4 换个风格 (style switch): the same melody harmonized as Bach, pop and jazz.
  - A4.1 Blind test: listeners name the style correctly >= 80%. (Human; build the kit; give proxies.)
  - A4.2 Every style: no-clash share with her >= 90%.
  - A4.3 Chord-prediction leaderboard: predicting the original chords from the melody, clearly above
    the current HMM.
  - Deliverable: three-style comparison recordings + leaderboard results.

D5 实时和声效果器 (live harmonizer effect): her voice is pitch-shifted in real time to a diatonic third
below and played with her, like a stage harmonizer.
  - A5.1 Latency mic-in to speaker-out <= 30 ms end to end, measured on the board.
  - A5.2 No howling: speaker at normal singing volume, mic and speaker 1 m apart in one room, 3 minutes
    of singing without howling or feedback.
  - A5.3 When she stops, the speaker is silent within 0.2 s and does not replay room noise or its own echo.
  - A5.4 Clash time after her note changes <= 10% of her singing time by the existing scoring.
  - Deliverable: real-time demo + latency measurement record + 3-minute no-howling recording.

## What every builder must produce

1. Code in your folder; one command that regenerates your results (write it at the top of REPORT.md).
2. `futures/results/<id>.json`:
   ```json
   {"id": "d1", "title_zh": "录一遍，配一遍", "updated": "...",
    "summary_zh": "two short Chinese sentences: what works, what does not",
    "acceptance": [{"line": "A1.1", "text_zh": "...", "measured": "4.2 s board estimate (2.1 s M4)",
                    "threshold": "<= 5 s", "status": "pass|fail|needs_human|needs_hardware",
                    "evidence": "relative path to the log/json that shows it"}],
    "demos": [{"file": "futures/d1_record_harmonize/demo/x.mp3", "label_zh": "...", "source": "...",
               "page_safe": true}],
    "local_only": ["paths of local-only listening files"],
    "how_to_run": "one command",
    "caveats_zh": ["short honest caveats"]}
   ```
   `status` is `pass` only when you measured it under these exact definitions; a number from
   simulation where the line says "on the board" or "in a room" is `needs_hardware` with the
   simulated number in `measured`.
3. `<your folder>/REPORT.md`: method in a few lines, numbers, what failed, what to do next.
