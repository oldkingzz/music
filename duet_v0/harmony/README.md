# harmony/: learned harmony line for Duet

This folder holds a small CRF that writes Duet's harmony voice. It was trained on 243 Bach chorales
from the music21 corpus and is a drop-in replacement for `follow.plan_harmony`, which always sings
a diatonic third or sixth below the singer.

```python
import sys; sys.path.insert(0, 'harmony')
from model import plan_harmony_learned
plan, key_name = plan_harmony_learned(notes)   # notes from follow.offline_notes(...)
# plan: [{'note', 'r', 'dur', 'midi', 'h'}], the same notes as plan_harmony keeps
acc = Accompanist(notes, plan, join_r)         # unchanged
```

Runtime needs numpy only. A song takes 1–2 ms, and the parameters are `params/crf_alto.npz`
(92 KB).

## Result in one table

Measured on 53 held-out chorales whose tunes never appear in training. Full detail, confidence
intervals and figures are in **[eval.md](eval.md)**.

| | Rule (`plan_harmony`) | **Learned (shipped)** | Bach |
|---|---|---|---|
| Harmony note is in Bach's chord | 51.6 % | **65.1 %** (+13.5, CI +10.5…+16.6) | 100 % |
| Same note as Bach's alto | 25.1 % | **31.4 %** (+6.4, CI +2.3…+10.1) | 100 % |
| Two-voice consonance with the melody | 100 % (3rds/6ths only) | 94.3 % | 64.7 % |
| Parallel 5ths / 8ves per 100 moves | 0 / 0 | 0 / 0 | 0.04 / 0.04 |
| Held (repeated) harmony notes | 17 % | 34 % | 38 % |
| Key detected from the melody alone | 64 % | **83 %** | n/a |

## Files

| file | what |
|---|---|
| `model.py` | `plan_harmony_learned`, `HarmonyModel`: key detection, feature templates, Viterbi, duet constraints |
| `train.py` | CRF training (L-BFGS, partial labels) and the count-based baselines |
| `chorales.py` | music21 corpus → `data/bach_satb.json` (soprano-onset SATB slices) |
| `dataset.py` | tune-grouped train/val/test split → `data/split.json` |
| `ablation.py` | feature-set × σ selection on val → `params/ablation_alto.json` |
| `evaluate.py` | λ/β sweep on val, all generators on test, bootstrap CIs → `results/eval_results.json` |
| `metrics.py` | music-theory metrics: consonance, parallels, crossing, spacing, range, leaps, chord tones |
| `pipeline_eval.py` | test sopranos through synthetic singer → YIN → `offline_notes` → planners |
| `integration_check.py` | runs `run_v0.run_scenario` with the learned plan |
| `make_examples.py` | Greensleeves wavs + MusicXML + piano roll → `examples/` |
| `report.py` | figures → `figures/` |
| `tests/test_harmony.py` | 14 unit tests |
| `params/` | `crf_alto.npz` (shipped model and its decoding defaults λ=1.0, β=0.25), baselines, logs |
| `examples/` | `greensleeves_{rule,learned}.wav`, `harmony_{rule,learned}.wav`, `greensleeves_compare.musicxml`, `plans.json` |
| `figures/` | `test_metrics.png`, `duet_tradeoff_val.png`, `interval_motion_hist.png`, `ablation_val.png`, `greensleeves_pianoroll.png` |

The folder takes about 41 MB on disk, most of it wavs from the integration check under
`results/integration/`.

## How to run

```bash
cd project/duet_v0
.venv/bin/python -m unittest discover -s harmony/tests -v    # 14 tests, < 1 s
.venv/bin/python harmony/make_examples.py                    # regenerate the listening example
```

The full retraining and evaluation sequence is in eval.md §9 and takes about 10 minutes on CPU.

## What was verified

- **Leakage control.** The split is tune-disjoint. Repeated settings of the same hymn tune were
  grouped by melodic 4-gram Jaccard similarity. 21 duplicate corpus files were removed, after a
  unit test caught them.
- **Training math.** The CRF gradient matches finite differences, and forward-backward log Z and
  the marginals match brute-force enumeration (unit test). Viterbi matches brute force on random
  problems.
- **Drop-in compatibility.**
  - The plan keeps exactly the same notes and keys as `plan_harmony`, including for unpitched and
    too-short segments.
  - `Accompanist` accepts it.
  - `run_v0.run_scenario` produces identical timing with either plan.
- **Robustness.**
  - The harmony stays 0–16 semitones below the melody.
  - There are no parallel 5ths or 8ves under default decoding, on Greensleeves and on random
    melodies.
  - Results are transposition-equivariant (±1, ±2 semitones) and octave-equivariant (a singer an
    octave lower gets the same line an octave lower).
- **Through the real rehearsal front end, with a synthetic singer.** Test results are unchanged
  (31.4 % exact, 65.0 % in chord, 94.4 % consonant).

## What is NOT verified

- **Nobody has listened to it.** I cannot play audio here, so the wavs are unheard and no listener
  study exists. All quality claims rest on symbolic metrics.
- **Real voices** have not been tested. The front-end test uses the project's synthetic singer.
- **Musical genre.** Behaviour on pop or folk songs with modulation or chromaticism is untested.
  The model is trained only on Bach chorales and the key detector assumes the song ends on its
  tonic.
- **The λ/β thresholds** (at least 95 % consonance, held-note rate at most Bach's) are a design
  judgement made on validation, not a measured listener preference.
- **Tenor and bass** models are supported by the code (`--voice`) but were not trained or
  evaluated.

## Proposed (optional) change outside this folder

None is required. If the team wants the learned line by default, `run_v0.py` (and later `live/`)
can swap `plan_harmony(det)` for `plan_harmony_learned(det)`; the call signature is the same.
