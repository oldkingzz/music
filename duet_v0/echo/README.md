# echo/: the machine hearing itself

In a duet the speaker plays Duet's own voice into the room, and some of it reaches the microphone. If the
follower takes that sound for the singer, it ends up following itself. This folder covers three things:

1. It measures how bad the problem is, using a closed-loop simulation.
2. It provides a guard (`guard.py`) that sits between the mic and `FeatureTracker`.
3. It turns the numbers into physical design rules: where the mic goes, which polar pattern, how far from the speaker.

Everything here is simulated with the synthetic Greensleeves singer from `singer.py`. Nothing has been
measured on hardware or with a real singer. See "What is NOT verified" below.

## Files

| file | what |
|---|---|
| `sim.py` | Closed-loop simulator. Each 10 ms the mic gets `singer + gain · echo(machine voice so far)`. The machine voice is streamed from what `Accompanist` has committed so far. `follow.py` and `dsp.py` are imported unmodified. |
| `guard.py` | `EchoGuard`, the mitigation stage. `EchoCanceller` is a two-path partitioned-block frequency-domain NLMS (MDF) filter with 10 ms blocks. `PitchMask` is the reference-aware pitch mask. |
| `sweep.py` | 780 main runs (5 methods × 13 bleed levels × 4 scenarios × 3 echo paths) plus 208 stress runs. Uses 7 processes and takes about 17 min on an M4. |
| `analyze.py` | Tolerance tables, writes `results/summary.json`, and draws `figures/bleed_sweep.png`, `tolerance.png`, `mic_placement.png` and `stress.png`. |
| `figures.py` | Mechanism figures: `figures/why_it_fails.png` and `figures/aec_erle.png`. |
| `physical.py` | Mic-placement acoustics, writes `results/physical.json`. |
| `tests/test_echo.py` | 11 checks. Plain asserts, so no pytest is needed. All pass. |

```bash
cd project/duet_v0
.venv/bin/python echo/tests/test_echo.py        # ~40 s
.venv/bin/python echo/physical.py
.venv/bin/python echo/sweep.py --procs 7        # ~17 min, writes echo/results/*.jsonl (~3 MB)
.venv/bin/python echo/analyze.py
.venv/bin/python echo/figures.py
```

No new packages were installed. Total output is about 2 MB.

## How the simulation works

- **Singer and engine.** The scenarios are the ones from `run_v0.py`: natural, expressive, pause and skip. The engine is the one in `follow.py`.
  - With no bleed, the closed loop reproduces `run_v0.py` exactly (test 1). With the shipped follower v2 (2026-10-16) steady gives a median |async| of 6.59 ms and natural 20.84 ms; with the phase-1 v1 follower they were 5.63 and 10.84 ms. The echo guard numbers below were measured with v1.
  - Engine compute time is fixed at 2 ms so that runs are deterministic. `run_v0` measures it instead.
- **Machine voice.** The voice is `StreamVoice`, the causal version of `dsp.render_voice`. Its correlation with the offline render is above 0.995 (test 2).
  - A note the Accompanist takes back, such as a false phrase entry, stops when it is retracted, just as it would on the device.
- **Echo path (`room_ir`).** The path from the reference the host sends to the mic samples it receives has these parts:
  - 20 ms system delay: the 2-frame jitter buffer from PLAN plus converter latency.
  - 1, 2 or 3 ms acoustic delay, i.e. 0.34, 0.69 or 1.03 m.
  - A 250 Hz second-order roll-off for a small speaker.
  - An exponential room tail with RT60 = 0.3 s. Test 4 checks the decay at −200 dB/s.
  - A direct-to-reverberant ratio taken from the critical distance r_c = 0.057·√(QV/RT60), with V = 40 m³ and Q = 2, giving r_c = 0.93 m. The DRR is 2.7 dB at 0.69 m.
- **Bleed level.** This is the RMS of a sustained machine note's echo at the mic, relative to the singer's RMS over her sung 10 ms frames. The measured realised bleed matches the target within about 2 dB (test 3).
- **Stress cases.**
  - Speaker distortion at 3 % THD, from a 2nd- and 3rd-order polynomial. A linear canceller cannot remove this part.
  - The echo path changes 40 % into the song: the room tail is partly replaced, as when somebody walks in front of the speaker.

## Result 1: the risk without a guard

**Main metric.** Over 4 scenarios × 3 echo paths there are 312 planned harmony notes per bleed level, and 273 are played when there is no bleed. The skip scenario misses some notes by design.

**Tolerance criterion.** A level counts as tolerated only if all three hold:
- the pooled median |asynchrony| stays within 10 ms of no-bleed (13.2 ms), as the brief asks;
- at least 90 % of the no-bleed notes are still played;
- no extra wrong notes are sung.

Timing on its own is not enough. A system that loses track simply stops playing, and the few notes it still plays are on time. In every failure the played-notes check was the one that failed first. Both versions are reported below.

| bleed (dB re her voice) | −36 | −30 | −24 | −18 | −12 | −6 | 0 | +6 |
|---|---|---|---|---|---|---|---|---|
| no guard: median ms / notes played | 15 / 100 % | 15 / 93 % | 16 / 79 % | 16 / 78 % | 17 / 76 % | 34 / 47 % | 45 / 11 % | 46 / 8 % |

**Why it fails** (`figures/why_it_fails.png`). Pitch errors are rare: at −6 dB, 66 of 322 frames where both voices sound are wrong. But every wrong frame is a sub-harmonic.
- Her A4 and the machine's F4 form a 5:4 major third, so the two voices share one period.
- YIN reports that common period, F2, two octaves down.
- A few of those frames fall on an onset. Pitch confirmation then fails, the follower resets, it jumps to the wrong bar, and the rest of the song is lost.

**The pause scenario is worst.** It fails already at −30 dB. During her pause, the echo of the machine's false phrase entry is the only sound in the room, and it passes the voicing gate.

## Result 2: the mitigations

| method | tolerated bleed (all three criteria) | timing-only criterion | no-bleed cost |
|---|---|---|---|
| no guard | **−30 dB** | −9 dB | n/a |
| canceller, no sound check (cold) | −36 dB, *worse than nothing* | −12 dB | plays only 219 of 273 notes even with zero bleed |
| pitch mask only | **−24 dB** (pause limits it; natural/expressive +3, skip −3) | 0 dB | median 15.0 ms instead of 13.2 ms |
| canceller with 2 s sound check | **−6 dB** | ≥ +6 dB | none |
| **canceller + mask** | **≥ +6 dB** (highest level tested; 96 % of notes played, median 15.6 ms at +6) | ≥ +6 dB | median 14.9 ms instead of 13.2 ms |

Under stress, tested on natural and pause with the 2 ms path:

| stress case | none | canceller | mask | canceller + mask |
|---|---|---|---|---|
| 3 % speaker distortion | −30 | −6 | −9 | **−3 dB** |
| echo path changes mid-song | −30 | −18 | −12 | **≥ +6 dB** |

CPU cost of the guard per 10 ms frame: median 0.30 ms, worst p99 4.2 ms with 7 parallel simulations sharing the CPU. Test 11 checks that a quiet run stays under 3 ms.

### Design decisions, with the evidence for each

1. **A cold adaptive canceller is harmful in a duet.** I learned this the hard way during development.
   - The machine voice is narrow-band: a few harmonics. Each new note excites frequencies the filter has never seen.
   - The singer's voice is correlated with the reference, because it is the same song, in consonant harmony and in the same rhythm. So an NLMS filter that adapts during the duet partly learns to cancel her.
   - Measured result: a cold start gives a median ERLE of 2.4 dB, and on the no-bleed runs the duet drops from 273 to 219 played notes.
   - Valin's variable-step rule (the paper behind Speex) is implemented as an option (`step='valin'`). On these tonal references it was not robust enough without a sound check.
2. **Remedy: a 2 s sound check.** Before the song, the speaker plays soft pink noise at about −10 dB below the machine voice. This teaches the filter every frequency at once, giving 21–29 dB ERLE at the end of the check depending on bleed SNR.
   - After the check, the filter adapts only when the mic is explained by the echo estimate (double-talk gate |e|² < 0.25·|ŷ|²).
   - It uses Speex's foreground/background two-path scheme. The output filter changes only when the background filter is clearly better, and the background is reset if it diverges.
   - Measured result (`figures/aec_erle.png`): median ERLE 26 dB for the whole song while she sings throughout. After the path change it drops to 0–5 dB and climbs back to about 14 dB by the end of the song.
3. **The pitch mask.** The machine's pitch comes from YIN on the reference itself, aligned by the delay that the sound check measures with GCC-PHAT.
   - If the detected pitch is at or below the machine's note, it is the machine or a mixture sub-harmonic, because the harmony line is always below her. In that case the mask applies, unless the pitch is one of her expected notes (previous, current or next note of the score).
   - When the mask applies, the window is passed through a comb filter `x[n] − x[n − τ_machine]` (fractional delay), which nulls every machine harmonic, and YIN runs again. Comb filtering does not change her periodicity, so her pitch survives without bias. Test 7: an A4+F4 mix goes from F2 to 69.0.
   - My first version excluded YIN lags instead. That was biased by up to 0.6 semitone, and it produced early glide onsets.
   - Two additions were needed:
     - **Voicing recovery:** a loud but aperiodic frame becomes voiced if the comb leaves a clean period.
     - **A 200 ms hold of the machine pitch after the reference stops,** to cover the room tail.
4. **Canceller + mask together** is the only configuration that keeps the duet intact at 0 dB and above, and it is the one I recommend.
   - The canceller removes 25–30 dB of the linear echo.
   - The mask catches what the canceller cannot: non-linear residue, path changes, and the first notes after a change.

**Proposed change for `live/`.** Insert `guard.cancel(mic, ref)` before `FeatureTracker.step` and `guard.mask(ft, expected)` after it, where `expected` is `[notes[acc.k−1], notes[acc.k], notes[acc.k+1]]['midi']`. Then run the 2 s sound check at start-up through SPK_AUDIO; it needs no protocol change. `ref` must be exactly the samples sent in SPK_AUDIO, aligned frame by frame with MIC_AUDIO.

**Proposed change for `dsp.py`.** `dsp.yin` could return its normalised difference curve; `guard.yin_curve` currently duplicates the maths.

## Result 3: physical design (`physical.py`, `figures/mic_placement.png`)

### Assumptions

- She sings at 80 dB SPL at 1 m. Hunter, Švec & Titze measured about 82–85 dB for loud /a/ at 1 m.
- The machine is 3 dB under her at 1 m.
- The room is 40 m³ with RT60 0.3 s and speaker Q = 2, giving r_c = 0.93 m.
- Direct sound follows the 1/r law: −6 dB per doubling of distance.
- The reverberant field is uniform.
- A cardioid follows 0.5 + 0.5 cos θ. It rejects diffuse sound by its random energy efficiency, −4.8 dB. Its rear rejection is assumed to be 15 dB, because the ideal null is −∞ and real capsules are worse.

| mic placement | machine voice at the mic | verdict |
|---|---|---|
| A: omni mic on the Duet board, 5 cm from its own speaker | **+20 dB** | Fails even with canceller + mask (limit about +6 dB, or −3 dB with distortion). Do not do this. |
| B: cardioid on the far face of the box, 15 cm, speaker behind it | −2 dB | Works only with canceller + mask; marginal under speaker distortion. |
| C: omni on a stand 20 cm from her mouth, speaker 1 m away | −14 dB | Needs the guard. |
| **D: cardioid on a stand 20 cm from her mouth, speaker 1 m behind it (180°)** | **−21 dB** | Recommended. 27 dB of margin with canceller + mask. Still not enough with no guard in the pause case (limit −30 dB). |
| E: cardioid headset 5 cm from her mouth | −33 dB | Best acoustically; safe even without the guard. |

### Rules for the hardware and mechanical teams

- Give the mic its own cable, and do not mount it on the Duet enclosure.
- Use a cardioid electret on a stand about 20 cm from her mouth, aimed at her, with the speaker directly behind the mic at 1 m or more.
- Aim the speaker at the listener, away from the mic.
- Every doubling of the speaker distance gains 6 dB on the direct part only. The reverberant part, about 2.7 dB below direct at 0.7 m in this room, does not fall with distance, so beyond r_c ≈ 0.9 m extra distance helps little.
- Treat the room and the cardioid's −4.8 dB diffuse rejection as the floor.
- Keep the speaker's distortion low: drive it well below rated power. At 3 % THD the guarded limit drops from ≥ +6 dB to −3 dB.
- Repeat the sound check whenever the mic or speaker is moved.
- Set the preamp gain for singer plus echo. At 0 dB bleed the combined peak is about 1.4× her peak, so leave at least 6 dB of headroom in the PCM1808 input range.

## What was verified

- Closed loop with no bleed equals `run_v0.py`: natural and steady medians and p90s match `outputs/synthetic/*/metrics.json`.
- The streamed machine voice equals the offline render (correlation > 0.995). The realised bleed matches the target within 2–3 dB. The room IR decays at −200 dB/s for RT60 = 0.3 s.
- The canceller converges on white noise (> 25 dB, and the delay estimate lands within 2 samples of 528). After a sound check it gives > 20 dB ERLE in 0 dB double talk.
- The comb recovers her pitch under the harmony. The guard is transparent when the reference is silent. The guard does not change the no-bleed result on natural.
- GCC-PHAT delay estimation is within 2 samples.
- All sweep numbers come from `results/main.jsonl` and `results/stress.jsonl` through `analyze.py` into `results/summary.json`.

## What is NOT verified

- **Real audio.** No real singer, real speaker, real room or real mic was used. The singer is synthetic, and the machine voice is the simple `render_voice` 'oo'. Real voices have breath noise, consonants and formant changes that affect both YIN and the canceller.
- **Echo path model.** The direct path, a noise tail and a 250 Hz roll-off are a simplification. There are no early reflections from a table, no structure-borne path through the enclosure, and no real speaker frequency response. The 20 ms system delay is assumed, and USB jitter is not modelled; the canceller assumes the reference-to-mic delay stays constant within ±1 block.
- **Mic and speaker figures.** The 15 dB cardioid rear rejection and the 80 dB / 77 dB singer and machine levels are assumptions. They must be checked against the chosen capsule's polar plot and with an SPL meter.
- **Sample size.** There are 26 harmony notes per run and 12 runs per level, so a tolerance threshold has roughly ±3 dB of granularity. Only one song was tested.
- **Distortion and path-change models.** The 3 % THD model is memoryless; real speaker non-linearity has memory. The path change is one synthetic perturbation.
- **Real-time timing.** The guard's CPU cost was measured inside the simulation on the MacBook, not inside the live app with USB I/O.

## Sources (opened)

- J.-M. Valin, "On Adjusting the Learning Rate in Frequency Domain Echo Cancellation With Double-Talk", IEEE TASLP 15(3), 2007: https://ar5iv.arxiv.org/html/1602.08044 (μ_opt ≈ residual-echo / error, leakage η̂ by regression, μ_max = 0.5)
- Speex `mdf.c` (MDF of Soo & Pang 1990; two-path foreground/background filter, reset when the background is worse): https://nest-open-source.googlesource.com/nest-mobile-client-iphone/5.2.2/Speex/+/a55be66ee69c9e58ffd207397017cc929cc95a8a/libspeex/mdf.c
- Critical distance D_c = 0.057·√(QV/RT): https://strutt.arup.com/help/Electroacoustics/DCritical.htm
- 1/r law, −6 dB per doubling of distance: https://sengpielaudio.com/calculator-distancelaw.htm
- First-order polar patterns (cardioid ½(1+cos θ), null at 180°, REE 0.333 = −4.8 dB): https://fdn.montana.edu/rmaher/ee417_fl08/MIC%20HANDOUT.pdf
- Shure, null angles (cardioid 180°, supercardioid 125°, hypercardioid 110°): https://www.shure.com/en-US/performance-production/louder/specialist-polar-patterns-hypercardioid-and-subcardioid
- Hunter, Švec & Titze, "Comparison of the Produced and Perceived Voice Range Profiles in Untrained and Trained Classical Singers", J. Voice (SPL at 1 m): https://pmc.ncbi.nlm.nih.gov/articles/PMC4782147
