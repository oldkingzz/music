<!-- Phase D draft (report writer). src path conventions: RUN/ = /Users/san/claude_sandbox/music/duet_v0/runs_mini/v4_1101/ ; V3/ = /Users/san/claude_sandbox/music/duet_v0/futures/v3_nolatency/ ; SI/ = /Users/san/claude_sandbox/music/duet_v0/futures/v4_selfiter/ ; DB/ = /Users/san/claude_sandbox/music/duet_v0/duet_beat/ ; DER1, DER2 = RUN/report_drafts/tools/derived_numbers_v1.json, derived_numbers_v2.json (differences and means of the result files only; scripts derive_numbers*.py in the same folder). -->
# Duet v4: a better real-time chord guess (Mac mini run v4_1101, 2026-11-01)

**What changed and why.** Duet picks each harmony note the instant her note starts, so it must guess the chord in real time. Vincent's listening session put all the audible problems there, and the diagnosis agrees: with the same note chooser, telling Duet the true chord at her onset lifts music_nd from 0.621 to 0.840. The old HMM names the right triad only 43 % of the time, because her melody under one chord usually fits 2-3 triads equally well. The new model keeps m1_filter_v5lo's chooser and all six weights and only changes the chord belief, to a normalized log-linear mix b ∝ b_HMM^0.3 · b_GRU^0.7 of m1's HMM and m3_seq_v5's fixed three-GRU average. The GRUs remember more of her melody (top-1 accuracy 0.48 vs 0.43); the HMM share keeps the belief steadier, so Duet leaps less than with the GRUs alone (folk large leaps 0.0185 vs 0.0212).
<!-- src: SI/listen/result.json; RUN/diag/diag_rt_v1.json headline_mean_of_4 (music_nd rt 0.6206, told 0.8403; chord_acc rt 0.4335, gru 0.4841); RUN/diag/why/why_d_v1.json all_spans.mean_tied 2.02-3.12; V3/systems/r2_mix_v2.py (ALPHA = 0.7); big = last val rows of V3/leaderboard_v3.jsonl for r2_mix_v2 (line 175) and r1_gru_chooser_v1 (line 172) = DER2 ideas.E2 / ideas.E1 big_folk -->

**Numbers** (music_nd: share of her singing time where Duet is consonant with her, on a true chord tone, and not on her own note or octave; headline = mean of the 4 datasets):

<!-- src (table): music_nd = RUN/nd/<name>/results/nd_<name>.json; seed 2 = RUN/seed2/<name>.json; top-1 chord accuracy = RUN/diag/diag_champs_v1.json table_4dp headline.acc (0.4335 / 0.4913 / 0.4916); song counts = RUN/report/board.txt -->
| val split | m1_filter_v5lo (old) | r2_mix_v2 (presentable) | r3_style_mix_v3 (best score) |
|---|---:|---:|---:|
| bach (17 songs) | 0.6655 | 0.6853 | 0.7028 |
| folk (170) | 0.6766 | 0.7126 | 0.7178 |
| pop (300) | 0.5601 | 0.5908 | 0.5963 |
| pop909 (40) | 0.5804 | 0.5891 | 0.5959 |
| **headline** | **0.6206** | **0.6444** | **0.6532** |
| seed-2 rescore | 0.6208 | 0.6470 | 0.6527 |
| top-1 chord accuracy | 0.434 | 0.491 | 0.492 |

Baselines: a third below her 0.5054; m3_seq_v5 0.6104 (its own chooser copies her note up to 18 % of the time); told the true chord (m4_oracle) 0.8388.<!-- src: RUN/nd/third_below/results/nd_third_below.json; RUN/nd/m3_seq_v5/results/nd_m3_seq_v5.json (bach double 0.1844); V3/results/nd_m4_oracle.json -->

**Tuesday.** r2_mix_v2 is the Tuesday model if Vincent passes its quiz, otherwise m1_filter_v5lo. r3_style_mix_v3 scores higher but is not presentable under the strict rule: its HMM side is a style-aware HMM (pooled, pop and pop909 copies of m1's HMM, weighted by a running style guess from her rhythm), which is more than "the HMM".<!-- src: RUN/state_R4.json champion, presentable_champion; SI/PROGRAM.md §4.3, §5; V3/systems/r2_style_hmm_v1.py -->

**Files** (in duet_v0/): models in `futures/v3_nolatency/systems/`; board, history, audio and cards in `runs_mini/v4_1101/report/`; diagnosis in `runs_mini/v4_1101/diag/`.

**Reproduce one score** (about 10 s on CPU; no `M1_*`, `M3_*` or `DUET_HIST` variables set; pick a new `<run>`, since rescore_nd writes `results/nd_<name>.json` into the current folder):

```
mkdir -p $ABS/runs_mini/<run>/nd/<name>/results && cd $ABS/runs_mini/<run>/nd/<name> && \
$ABS/.venv/bin/python $ABS/futures/v3_nolatency/tl.py 600 $ABS/.venv/bin/python \
$ABS/futures/v3_nolatency/rescore_nd.py <name>
```

With `<name>` = `r2_mix_v2` it prints 0.644 (bach 0.685, folk 0.713, pop 0.591, pop909 0.589).<!-- src: SI/PROGRAM.md §2.3; RUN/hb/ log of the r2_mix_v2 nd run (END rc=0 secs=10); RUN/nd/r2_mix_v2/results/nd_r2_mix_v2.json -->

**Still weak.**
- Pop: 0.59 vs 0.71 on folk; chord accuracy only 0.37 on pop.<!-- src: RUN/nd/r2_mix_v2/results/nd_r2_mix_v2.json; RUN/diag/diag_champs_v1.json table_4dp r2_mix_v2 pop acc 0.3697 -->
- Mishearing: the benchmark mishears 10 % of her notes, costing 0.04-0.06 per dataset; about 80 % of that is in the note choice (Duet can only fit what it heard), so fixing it needs to know which notes were misheard.<!-- src: DB/bench.py NOISE = 0.10; RUN/r4/B/check_why_20261101_160400.json cost_total 0.0412-0.0565, belief share 13-24 % (DER1 r4B_mishearing); RUN/diag/recheck/recheck_rt_extra_result_20261101_145317.json (m1 clean melody 0.668 vs heard 0.6206) -->
- Labels: the 7 triads of the major key, so 7th chords and the raised 7th (leading tone) in minor-key songs count as wrong; a limit of the labels, not tuned.<!-- src: DB/theory.py CHORDS; SI/PROGRAM.md §4.4 -->
- The gain is an average: over the 17 Bach val songs it beats m1 on 9 whole songs but on only 6 of the 30-second clips.<!-- src: RUN/phaseD/A/clip_context.json summary -->

All numbers are val, and everything was tuned on val (this bundle has no test split).
