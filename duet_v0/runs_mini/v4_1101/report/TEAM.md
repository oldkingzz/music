<!-- Phase D final (finalizer, 2026-11-01): report_drafts/TEAM.md with the fact-check and spec-check corrections. src path conventions: RUN/ = /Users/san/claude_sandbox/music/duet_v0/runs_mini/v4_1101/ ; V3/ = /Users/san/claude_sandbox/music/duet_v0/futures/v3_nolatency/ ; SI/ = /Users/san/claude_sandbox/music/duet_v0/futures/v4_selfiter/ ; DB/ = /Users/san/claude_sandbox/music/duet_v0/duet_beat/ ; DER1, DER2 = RUN/report_drafts/tools/derived_numbers_v1.json, derived_numbers_v2.json (differences and means of the result files only; scripts derive_numbers*.py in the same folder). -->
# Duet v4: a better real-time chord guess (Mac mini run v4_1101, 2026-11-01)

**What changed and why.** The weak spot is the real-time chord guess: told the true onset chord, the same chooser scores 0.840 instead of 0.621. r2_mix_v2 changes only the chord belief, to b ∝ b_HMM^0.3 · b_GRU^0.7 (m1's HMM times m3_seq_v5's three-GRU average); m1_filter_v5lo's chooser and six weights stay. The GRUs remember more melody; the HMM damps leaps.
<!-- src: SI/listen/result.json (Vincent: the problems are in the real-time chord guess); RUN/diag/diag_rt_v1.json headline_mean_of_4 (music_nd rt 0.6206, told 0.8403); V3/systems/r2_mix_v2.py (ALPHA = 0.7); leaps: last val rows of V3/leaderboard_v3.jsonl for r2_mix_v2 (line 175, folk big 0.0185) and r1_gru_chooser_v1 (line 172, folk big 0.0212) = DER2 ideas.E2 / ideas.E1 big_folk -->

**Numbers** (music_nd = share of her singing time with Duet consonant with her, on a true chord tone, not on her note or octave; headline = mean of 4 datasets; val only, tuned on val):

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

**Tuesday.** r2_mix_v2 if Vincent passes its quiz, else m1_filter_v5lo. r3_style_mix_v3 scores higher but is not presentable (PROGRAM §4.3): its HMM side is a style-aware HMM (pooled, pop-only and pop909-only versions of m1's HMM, evidence weight 1.0 instead of 1.5, weighted by a running style guess from the meter and her rhythm), more than "the HMM".<!-- src: RUN/state_R4.json champion, presentable_champion; SI/PROGRAM.md §4.3, §5; V3/systems/r2_style_hmm_v1.py (STYLES lam 1.0; SubHMM corpora pop / pop909; StyleFilter.start logw += mprior[M], _finish_beats logw += rh[:, rword]); V3/systems/m1_filter_v5lo.py (P lam 1.5) -->

**Files** (all in `report/`): `code/` (the new systems' .py files), `board_v2.txt`, `history_v2.txt`, `audio/`, `CARD_*_zh.md`, `DIAGNOSIS_zh.md`. **Reproduce one score** (~10 s on CPU): where `systems/` lacks them, first copy `code/*.py` into `$ABS/futures/v3_nolatency/systems/`; set no `M1_*`, `M3_*` or `DUET_HIST` variable; use a new `<run>` (rescore_nd writes `results/nd_<name>.json` into the current folder):

```
mkdir -p $ABS/runs_mini/<run>/nd/<name>/results && cd $ABS/runs_mini/<run>/nd/<name> && \
$ABS/.venv/bin/python $ABS/futures/v3_nolatency/tl.py 600 $ABS/.venv/bin/python \
$ABS/futures/v3_nolatency/rescore_nd.py <name>
```
With `<name>` = `r2_mix_v2` it prints 0.644 (bach 0.685, folk 0.713, pop 0.591, pop909 0.589).<!-- src: SI/PROGRAM.md §2.3; RUN/hb/r2B_r2_mix_v2_nd.log (END rc=0 secs=10); RUN/nd/r2_mix_v2/results/nd_r2_mix_v2.json -->

**Still weak.**
- Pop: 0.59 vs 0.71 on folk; chord accuracy only 0.37 on pop.<!-- src: RUN/nd/r2_mix_v2/results/nd_r2_mix_v2.json; RUN/diag/diag_champs_v1.json table_4dp r2_mix_v2 pop acc 0.3697 -->
- Mishearing (10 % of her notes) costs 0.04-0.06 per dataset (m1 0.043-0.052, r3_style_mix_v3 0.041-0.056); on r3_style_mix_v3 about 80 % of it is in the note choice (Duet fits what it heard).<!-- src: DB/bench.py NOISE = 0.10; RUN/r4/B/check_why_20261101_160400.json (rows.*.heard = r3_style_mix_v3's music_nd; cost_total 0.0412-0.0565; cost_belief / cost_total 0.13-0.24 = DER1 r4B_mishearing); RUN/diag/recheck/recheck_rt_extra_result_20261101_145317.json per_ds music_nd_clean_melody vs RUN/nd/m1_filter_v5lo (0.043-0.052) -->
- Labels are the 7 major-key triads: 7th chords and minor-key raised 7ths count as wrong.<!-- src: DB/theory.py CHORDS; SI/PROGRAM.md §4.4 (known limit, not tuned) -->
- Average gain: of 17 Bach val songs, r2_mix_v2 wins 9 whole songs but only 6 opening 30-s clips.<!-- src: RUN/phaseD/A/clip_context.json summary (CHAMP = r2_mix_v2: song level 9 / 1 / 7, clip level 6 / 1 / 10) -->
