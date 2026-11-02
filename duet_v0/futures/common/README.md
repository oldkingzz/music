# futures/common: shared SATB harmonizer (owner: D1)

```python
import sys; sys.path.insert(0, 'futures/common'); from satb import harmonize
out = harmonize(notes, tonic=None, mode=None)   # notes: [(onset_s, dur_s, midi), ...] monophonic, sorted
# out = {'tonic': pc, 'mode': 'major'|'minor', 'key_name': 'G major',
#        'slices': [{'on', 'dur', 'S', 'A', 'T', 'B', 'chord_pcs', 'label'}, ...]}   one slice per melody note
```

- numpy only, ~5 ms per 4-8 bar phrase after the first call (first call loads `satb_tables.json`, built
  from the TRAIN split of `harmony/data/bach_satb.json`).
- Key: `harmony/model.detect_key_melody` when tonic/mode are None.
- Chord Viterbi (whole phrase known) over (chord pc set, bass pc) states learned from Bach, then a
  voicing DP over (A, T, B) with a hard ban on parallel 5ths/8ves (all voice pairs, including hers) and a
  near-hard ban on harsh intervals (ic 1, 2, 6, 10, 11) against her note.
- A voice may hold across slices (same midi). Her register is shifted by octaves to Bach's soprano range
  internally and shifted back, so A/T/B follow her octave.
- A phrase ends where the gap to the next note is >= 0.10 s (cadence statistics are used there).
- Notes with midi <= 0 are dropped (the output then has fewer slices than the input). Empty input returns
  `slices: []`. Optional `params={...}` overrides weights in `satb.P` for one call.
- v2 (11:10, frozen 11:13 for the D1 test run): soprano-context term (previous/current/next note) in the chord
  Viterbi, a boundary cost so a moving voice is not harsh against her neighbouring note (robust to onset
  errors in transcribed melodies), and a repair loop that re-decodes until there are no parallels (0 on 300
  random wild melodies; worst case ~0.4 s for 60 notes, typically ~10 ms).
- Bach test split (53 chorales, key detected): no-clash 100 %, parallels 0/0, chord = Bach's 33.5 %.
  The clash gate that mutes voices against her measured pitch lives in `futures/d1_record_harmonize/pipeline.py`
  (`gate_masks`), not here.
- The API stays stable; D1 improves the internals in place. Evaluation: `futures/d1_record_harmonize/`.
