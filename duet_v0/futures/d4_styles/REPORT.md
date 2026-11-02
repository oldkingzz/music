# D4 换个风格 (style switch): report

Regenerate everything: `bash futures/d4_styles/run_all.sh` (the one-time test run was `.venv/bin/python futures/d4_styles/chord_board.py --test`; it refuses to run twice).

## A4.3 chord-prediction leaderboard (test)

Task: the whole clean melody of a song -> the diatonic triad (7 classes) of every beat; scored on every labelled beat. Songs: bach 27, folk 171, pop 973, pop909 40.

| system | bach | folk | pop | pop909 | mean |
|---|---|---|---|---|---|
| hmm_causal | 0.362 | 0.350 | 0.265 | 0.254 | 0.308 |
| hmm_causal_benchproto | 0.348 | 0.349 | 0.246 | 0.248 | 0.298 |
| hmm_fb | 0.576 | 0.522 | 0.344 | 0.325 | 0.442 |
| hmm_vit | 0.569 | 0.525 | 0.340 | 0.323 | 0.439 |
| style_hmm | 0.572 | 0.713 | 0.428 | 0.427 | 0.535 |
| bigru | 0.624 | 0.754 | 0.455 | 0.519 | 0.588 |
| bigru_vit | 0.624 | 0.754 | 0.455 | 0.519 | 0.588 |
| ens (chosen on val) | 0.630 | 0.758 | 0.454 | 0.518 | 0.590 |

- hmm_causal: baseline 1, the PDR ChordHMM (Bach train) run through duet_beat/planner.run exactly as the realtime system (decides 72 ms before each beat), on the clean melody. hmm_causal_benchproto: the same under the bench protocol (10 % misheard notes, from beat 8); its val Bach value reproduces the leaderboard_v2 "chord" number (0.319) exactly.
- hmm_fb / hmm_vit: baseline 2, the same ChordHMM decoded offline (forward-backward / Viterbi); hmm_fb was better on val.
- style_hmm: one HMM per style, duration-weighted pitch-class emissions + onset weight, separate downbeat emissions, positional transitions; kappa / onset weight / lambda / decoder tuned per style on val.
- bigru: 2-layer bidirectional GRU (96 units/direction) on 39 per-beat features (pc durations, note onsets, rest, bar position, meter, style id), trained on MPS on the union of the 4 train splits (Bach x12, folk x3, pop909 x2 oversampled), epoch chosen on val; inference in numpy (max |logit| difference to torch ~1e-6, asserted).
- bigru_vit / ens: BiGRU log-posteriors + style-HMM transitions (and, for ens, w x style-HMM emissions), Viterbi; w, beta on val.

Gains of the chosen system (95 % bootstrap CI over songs, paired, 2000 resamples; mean = mean of the 4 dataset gains):

- ens_vs_hmm_causal: bach +26.8 [+24.3, +29.2]; folk +40.8 [+38.9, +42.8]; pop +18.9 [+17.6, +20.2]; pop909 +26.4 [+23.2, +29.5]; mean +28.2 [+27.0, +29.5]
- ens_vs_hmm_fb: bach +5.4 [+3.3, +7.4]; folk +23.5 [+21.4, +25.6]; pop +11.0 [+10.0, +12.1]; pop909 +19.3 [+15.9, +22.6]; mean +14.8 [+13.7, +16.0]

Compute: 79 us/beat on M4; board estimate 192 us/beat (2 x M4 E-core, estimate) -> a 255-beat song in 49 ms.

## A4.2 no-clash per style (rendered note events vs her true notes)

| melody | Bach harsh | pop avoid | pop harsh | jazz avoid | jazz harsh |
|---|---|---|---|---|---|
| bach_test_bwv267 | 1.000 | 0.986 | 0.746 | 0.974 | 0.783 |
| folk_ashover_35 | 1.000 | 1.000 | 0.911 | 1.000 | 0.868 |
| pop909_063 | 1.000 | 0.950 | 0.818 | 0.955 | 0.800 |
| vocadito_12 | 1.000 | 1.000 | 0.766 | 1.000 | 0.744 |

Bach's own A/T/B against the soprano on the same chorale excerpt (bach/bwv267, harsh rule): 0.766 (per voice A 0.938, T 0.891, B 0.891). Per-voice values for every style are in out/styles.json.

Robustness on 30 more test melodies (Bach/folk/POP909, 32 beats): min Bach harsh 1.000, pop avoid 0.955, jazz avoid 0.971; mean pop harsh 0.837, jazz harsh 0.766.

- Bach: D1 futures/common/satb.py (chord Viterbi + voicing DP with a near-hard ban on harsh intervals against her), A/T/B sung by the synthetic singer.
- pop: per half bar the triad maximizing the pop-style BiGRU log-posterior minus avoid-note and harsh time against her; split into two one-beat chords when clearly better. Block chord on the half-bar + broken-chord eighths (piano), bass root on beats 1 and 3, soft straight-eighth shaker.
- jazz: the same base harmony as diatonic 7ths; before each downbeat chord change a tritone substitution, ii-V, or secondary dominant is tried (rotating) and kept only if its avoid-note time is no worse than the plain chord and <= 10 % of her time there. Rootless 4-note voicings (3-7 with 9/13/5, the variant with least clash), Charleston-type swung comping, walking bass (root, chord tones, chromatic approach chosen to avoid clashes), swung ride.

## A4.1 style identity (needs listeners)

Blind kit: futures/d4_styles/blind/ (12 clips = 4 melodies x 3 styles, random ids, KEY.json, INSTRUCTIONS.md in Chinese). Automatic proxy (mean over 4 melodies):

| style | 7th-chord share | substitution share | swung off-beat onsets | straight 8th onsets | onsets/s | max voices | chord changes/beat | mean note (beats) |
|---|---|---|---|---|---|---|---|---|
| bach | 0.01 | 0.00 | 0.03 | 0.29 | 2.07 | 3.0 | 1.33 | 0.88 |
| pop | 0.00 | 0.00 | 0.00 | 0.32 | 2.38 | 4.0 | 0.53 | 0.88 |
| jazz | 0.70 | 0.29 | 0.24 | 0.00 | 2.13 | 5.5 | 0.59 | 0.72 |

Leave-one-melody-out nearest-centroid classifier on these 8 statistics names the style of 83% of the 12 clips (a weak proxy: it shows the arrangements differ in the intended ways, not that listeners hear it).

## Demos (page-safe, futures/d4_styles/demo/, sources in demo/SOURCES.md)

- futures/d4_styles/demo/folk_ashover_35_bach_pop_jazz.mp3 (0.83 MB): Nottingham folk nottingham/ashover.abc#35 (test split), synthetic singer; GPL (Nottingham dataset)
- futures/d4_styles/demo/vocadito_12_bach_pop_jazz.mp3 (0.67 MB): vocadito track 12 (Bittner et al.), notes from its A1 annotation; beat 0.480s by estimate_beat; pulse x key score 0.65; CC BY 4.0
- futures/d4_styles/demo/bach_test_bwv267_bach_pop_jazz.mp3 (1.05 MB): Bach chorale bach/bwv267 soprano (music21 corpus, test split), synthetic singer; public domain

## What failed / caveats / next

- 和弦标签只有 7 个调内三和弦，离调和弦被折算成最接近的三和弦；流行数据（Hooktheory）只用于训练和打分，从未渲染成音频。
- 基线 1 是实时系统的因果判断（提前 L+delta 决定），本任务是离线看整首旋律，所以增益里有一大块来自“能看后文”；对离线 HMM 的增益更能说明模型本身。
- 流行/爵士的“刺耳版”不冲突比例低：三和弦/七和弦下任何经过音都会和某个和弦音成二度，这是定义使然，验收用避免音版。
- 巴赫风格用 D1 的 futures/common/satb.py（硬性禁止刺耳音程），所以刺耳版是 100%，比巴赫本人在同一段上的值还高；这说明它很保守，不代表更像巴赫。
- vocadito 片段的拍子是从音符起点估出来的（estimate_beat），真人唱得不完全按拍，流行/爵士伴奏和她可能有轻微错拍；音符用的是人工标注而不是自动转写。
- A4.2 的和声是从她的真实音符（乐谱或人工标注）算出来的，没有经过音高识别这一步；换成麦克风实录要再测。巴赫本人的参考值用的是女高音起音切片，内声部在一个旋律音内部的移动被忽略。
- 风格可辨认度只有统计代理（七和弦比例、替代和弦、摇摆八分音符），合成音色很简单，真人盲听还没做。
- Next: real listeners for A4.1; a key/beat tracker for real singing instead of annotation notes; richer timbres; chords beyond the 7 diatonic triads in the leaderboard labels.
