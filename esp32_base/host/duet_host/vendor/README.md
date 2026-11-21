# vendor

Unchanged copies from the first Duet repository (`project/duet_v0`, copied 2026-11-21):

| File | What the base uses from it |
|---|---|
| `dsp.py` | `SR` = 24 kHz, `HOP` = 10 ms, the YIN pitch estimator, `render_voice` (the soft machine voice) |
| `follow.py` | `analyse` (pitch and loudness per 10 ms) and `offline_notes` (splits a recording into notes, robust to vibrato) |

Do not edit them here. If a change is needed, change the caller in `duet_host/` or copy the function out.
`follow.py` imports `dsp` as a top-level module, so `duet_host/audio_notes.py` puts this folder on `sys.path`.
