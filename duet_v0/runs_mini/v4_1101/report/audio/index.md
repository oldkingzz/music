# Duet audio clips: champion vs old model vs baseline (Phase D, run v4_1101)

How to listen: use headphones; her voice (the "ah" singer) leans left and Duet's clarinet leans right.
Play the four files of one song back to back and follow the clarinet: it is Duet's real-time harmony line under her melody.

| file | song | system | role | beats | seconds | clip music_nd | clip double | song music_nd | clip notes unlike m1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| twinkle_r2_mix_v2.mp3 | twinkle | r2_mix_v2 | presentable champion | 40 | 31.0 | 0.583 | 0.111 | 0.614 | 18 / 35 |
| twinkle_r3_style_mix_v3.mp3 | twinkle | r3_style_mix_v3 | best score, not presentable | 40 | 31.0 | 0.611 | 0.111 | 0.636 | 15 / 35 |
| twinkle_m1_filter_v5lo.mp3 | twinkle | m1_filter_v5lo | old model m1 | 40 | 31.0 | 0.694 | 0.056 | 0.705 | 0 / 35 |
| twinkle_third_below.mp3 | twinkle | third_below | simple baseline | 40 | 31.0 | 0.694 | 0.000 | 0.659 | 23 / 35 |
| bwv140.7_r2_mix_v2.mp3 | bwv140.7 | r2_mix_v2 | presentable champion | 40 | 31.0 | 0.591 | 0.167 | 0.675 | 6 / 35 |
| bwv140.7_r3_style_mix_v3.mp3 | bwv140.7 | r3_style_mix_v3 | best score, not presentable | 40 | 31.0 | 0.561 | 0.167 | 0.658 | 8 / 35 |
| bwv140.7_m1_filter_v5lo.mp3 | bwv140.7 | m1_filter_v5lo | old model m1 | 40 | 31.0 | 0.652 | 0.091 | 0.675 | 0 / 35 |
| bwv140.7_third_below.mp3 | bwv140.7 | third_below | simple baseline | 40 | 31.0 | 0.576 | 0.061 | 0.592 | 27 / 35 |
| bwv245.26_r2_mix_v2.mp3 | bwv245.26 | r2_mix_v2 | presentable champion | 40 | 31.0 | 0.653 | 0.083 | 0.693 | 21 / 39 |
| bwv245.26_r3_style_mix_v3.mp3 | bwv245.26 | r3_style_mix_v3 | best score, not presentable | 40 | 31.0 | 0.625 | 0.028 | 0.670 | 16 / 39 |
| bwv245.26_m1_filter_v5lo.mp3 | bwv245.26 | m1_filter_v5lo | old model m1 | 40 | 31.0 | 0.667 | 0.000 | 0.705 | 0 / 39 |
| bwv245.26_third_below.mp3 | bwv245.26 | third_below | simple baseline | 40 | 31.0 | 0.472 | 0.000 | 0.455 | 32 / 39 |

## Songs

- twinkle: Twinkle Twinkle Little Star (traditional); 4 beats per bar, 48 beats in all; the clip is beats 0-39 (40 beats, 35 of her notes).
- bwv140.7: Bach chorale bach/bwv140.7 (val split, public domain); 4 beats per bar, 68 beats in all; the clip is beats 0-39 (40 beats, 35 of her notes).
- bwv245.26: Bach chorale bach/bwv245.26 (val split, public domain); 4 beats per bar, 48 beats in all; the clip is beats 0-39 (40 beats, 39 of her notes).

## What the numbers mean

- Every system ran on the FULL song exactly as in bench3: Duet hears her notes one at a time with the benchmark's 10 % mishearing, never sees the chord labels, and holds each note until her next onset. The clip was cut afterwards: the first min(song beats, 40) beats at 80 bpm (0.75 s per beat, about 30 s), re-attacked notes (Phase B chose no tied renders).
- clip music_nd: share of her singing time in the clip, from beat 4 on (Duet may listen first), where Duet's note is consonant with her TRUE note AND a tone of the TRUE chord AND not her own note or its octave (rescore_nd's definition). clip double: share of the same time where Duet sings her note or its octave. song music_nd: the same score over the whole song (this one song, not the dataset).
- clip notes unlike m1: how many of Duet's notes in the clip differ from m1_filter_v5lo's, out of her notes in the clip.
- For reference, val headline music_nd over all four datasets: r2_mix_v2 0.6444, r3_style_mix_v3 0.6532, m1_filter_v5lo 0.6206, third_below 0.5054; told the chord (m4_oracle) 0.8388.
- All numbers are val (this bundle has no test split) and every system was tuned on val.
- Licences: audio of Bach chorales (public domain) and Twinkle only; never folk, pop or pop909.
- 12 mp3 files, 5.96 MB in all (128 kbit/s stereo).
- Made by $ABS/runs_mini/v4_1101/phaseD/A/render_audio.py on 2026-11-01 16:18 EST; per-clip details and Duet's notes in $ABS/runs_mini/v4_1101/phaseD/A/render_log.json.
