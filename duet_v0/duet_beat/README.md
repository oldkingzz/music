# duet_beat: Duet on a click, guessing chords while it listens

The simplified Duet from the 10/25 spec (claude.ai/artifact/R74zwgC3EBAwQ8bniSh3bZ): the singer follows a
click track, the key is given, there is no rehearsal. Before every beat Duet guesses the chord behind the next
beat with a hidden Markov model, plans its alto note for the next 4 beats together with the chords (Viterbi),
sings only the first, and replans every beat. It never uses audio it could not have heard yet.

It is separate from the rest of the repo and borrows only: the YIN front end (`follow.FeatureTracker`), the
synthetic singer (`singer.py`), Duet's voice (`dsp.render_voice`) and the tune-grouped chorale split
(`harmony/data/split.json`).

| File | What it holds |
|---|---|
| `theory.py` | major scale, the 7 diatonic triads, alto range, consonances |
| `data.py` | Bach chorales on a beat grid with chord labels → `data/bach_beats.json` |
| `model.py` | the chord HMM: counts with Laplace smoothing, forward filter, prediction |
| `planner.py` | the online loop for every system (HMM and baselines B0–B4), joint (chord, note) planning, join and off-click rules |
| `audio.py` | clean and vibrato singers, click, audio observations, Duet's voice, stereo mix |
| `songs.py` | Twinkle and the user's arpeggios with hand-written chords; the hand-set "slow harmony" setting |
| `metrics.py` | M1–M5, U, song-level bootstrap |
| `run_eval.py` | train / tune on validation / score the test set → `out/results.json` |
| `run_audio.py` | front-end check, slow harmony on Bach, demos with audio → `out/audio_results.json`, `out/media/` |
| `build_page.py`, `page_template.html` | the results page |
| `tests/test_core.py` | causality (with a power check), normalisation, legal notes, oracle |

Run from the repo root (about 40 s in total):

```bash
.venv/bin/python duet_beat/data.py
.venv/bin/python duet_beat/run_eval.py
.venv/bin/python duet_beat/run_audio.py
.venv/bin/python duet_beat/tests/test_core.py
.venv/bin/python duet_beat/build_page.py <target_dir>
```

Results (10/25, 27 held-out major chorales, 1,460 beats): chord guessed right 35.0 % vs 30.9 % for always-I;
Duet's note in Bach's chord 63.6 % vs 61.5 % for always-I and 42.9 % for the repo rule run without rehearsal;
peeking at the current beat (not causal) would give 53.9 % chords. The audio front end matches the score on
96.8–98.3 % of beats and changes the scores by under 1 point. On the user's arpeggios with the slow-harmony
setting and the 12-vector observation: 87.5 % chords, 85.7 % notes in chord (one song, small sample).
