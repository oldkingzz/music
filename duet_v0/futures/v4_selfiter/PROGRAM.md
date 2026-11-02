# Duet self-iteration on the Mac mini — PROGRAM (v2)

> 给 Vincent：AirDrop `duet_mini.zip` 到 Mac mini，解压，`cd duet_v0`，运行 `claude`，选 Opus 5.5 + max effort，
> 粘贴：`ultracode. Read futures/v4_selfiter/PROGRAM.md and follow it exactly, phase by phase.`
> 你在场的部分：装环境 + 权限设置（~20 分钟）、盲听（~10 分钟）。之后切到 auto mode 就可以离开。

You are Claude Code on Vincent's Mac mini (Apple M4, 16 GB RAM). Ultracode is on: use the Workflow tool
for experiment rounds. This file is your whole brief. The laptop's CLAUDE.md rules are restated in §1
because they are not on this machine.

## 0. Mission, deadline, and the one thing that matters

- **Project.** Duet, a robot that sings a harmony line *under* a human melody singer (MEAM 5160,
  Group 6: Akshaya Venugopal, Christine Meng, Vincent Wang, Xinrui Gao).
- **Goal.** Latency is solved. Make Duet's harmony sound as good as possible under the zero-latency
  benchmark `futures/v3_nolatency/bench3.py`, judged by the fair score `music_nd` (consonant with her
  AND a tone of the true chord AND not her own note / octave) and by Vincent's ears.
- **Deadline.** Vincent meets the professor and team on **Tuesday 2026-11-03**. Stop experimenting at
  **Monday 2026-11-02 10:00 local time** or after 10 h of loop time, whichever is first; then Phase D.
- **What matters most.** On 10-30 Vincent was grilled on model details and could not answer. A result
  he cannot explain is worth nothing on Tuesday.
  - The current Tuesday model is **`m1_filter_v5lo`** (HMM chord filter + no-copy chooser, music_nd
    0.6206). It stays the Tuesday model unless a *presentable* champion (§4.3) wins AND Vincent passes
    its quiz (§5).
  - Prefer the simplest change that wins. Every result you report says in one plain Chinese sentence
    what changed and why it helps.
- **Talk to Vincent** in concise Chinese; each technical term as 中文（English） the first time; math in
  LaTeX. Files for teammates are English.

## 1. Hard rules (also for every subagent — paste §1 into every agent prompt)

1. **Never delete or overwrite.** No `rm`, `os.remove`, `shutil.rmtree`, `unlink`, `find -delete`,
   `> existing_file`, `open(existing, 'w')`, `mv` onto an existing path. New outputs go to new files
   under `$ABS/runs_mini/<run>/`. If something must be removed, list it and ask Vincent.
   *Allowed exceptions:* appending (`>>`, mode `'a'`) to logs, heartbeat logs, and
   `futures/v3_nolatency/leaderboard_v3.jsonl` (bench3 appends by design). State is versioned:
   `state_R<R>.json` per round, never rewritten; the latest is the highest R.
2. **Frozen files.** Never edit anything in `futures/v4_selfiter/CHECKSUMS.sha256`. Before every round run
   `cd $ABS && shasum -a 256 -c futures/v4_selfiter/CHECKSUMS.sha256`; any mismatch → stop, tell Vincent.
3. **No test split.** This bundle has none on purpose (`eval_sets('test')` raises KeyError). Never try
   to rebuild it (no music21 re-extraction, no downloads). Tune on `val` only, and say so in reports.
4. **No publishing, no git, no network, no installs beyond §2.** Everything stays local. Licences:
   DCS/ESMUC are CC BY-NC; Hooktheory (`pop`) is CC BY-NC-SA — **never render audio of pop/pop909/folk
   songs, never write pop titles or artists**. Audio: Bach chorales (public domain) and Twinkle only.
5. **Resources.** ≤ 3 Python jobs at once; `torch.set_num_threads(3)` (1 for GRU inference); CPU only.
   Wrap every Python run in `futures/v3_nolatency/tl.py <seconds>` (macOS has no `timeout`): 600 s per
   eval, 3600 s per training run. Run everything in the foreground; no background jobs inside agents.
6. **Disk.** Each round: `diskutil info / | grep -i "free space"` (df under-reports on macOS). Stop if
   < 20 GB free or `runs_mini/` > 2 GB. ≤ 2 checkpoints per system, each < 50 MB. Logs < 5 MB each.
7. **One name = one config.** Every config lives as constants in its own file
   `systems/r<R>_<idea>_v<k>.py`; never reuse a name. Never set `M1_*`, `M3_*`, `DUET_HIST` in the
   environment for a scored run (check `env | grep -E '^(M1_|M3_|DUET_HIST)'` is empty); new systems
   may not read `os.environ`.
8. **Paths.** In Phase A record `ABS` = the absolute path of `duet_v0` in `state_R0.json` and paste it
   literally into every agent prompt. Every Bash call is one line starting with `cd $ABS/... && `.

## 2. Phase A — setup (Vincent present, ~20 min)

0. **Bundle check.** `futures/v4_selfiter/{CHECKSUMS.sha256,EXPECTED.json}` exist; there is no `.venv/`.
1. **Environment** (tell Vincent first; he approves installs):
   ```
   brew install python@3.11 lame          # lame must end up at /opt/homebrew/bin/lame (make_demos3 uses it)
   python3.11 -m venv .venv
   .venv/bin/pip install -r requirements-lock.txt
   .venv/bin/pip install torch==2.15.0    # not in the lock file; needed by every GRU system
   .venv/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"   # 2.15.0 2.4.6
   ```
   If a pinned package fails, install it unpinned and log it.
2. **Integrity:** `shasum -a 256 -c futures/v4_selfiter/CHECKSUMS.sha256`.
3. **How to score `music_nd` (use this everywhere):** `rescore_nd.py` writes `results/nd_<name>.json`
   relative to the current directory with mode `'w'`, so always run it from a fresh directory:
   ```
   mkdir -p $ABS/runs_mini/<run>/nd/<name>/results && cd $ABS/runs_mini/<run>/nd/<name> && \
   $ABS/.venv/bin/python $ABS/futures/v3_nolatency/tl.py 600 $ABS/.venv/bin/python \
   $ABS/futures/v3_nolatency/rescore_nd.py <name>
   ```
   (~10 s, < 1 GB per call.) Raw `music`, `big`, `par`, flags: `cd $ABS/futures/v3_nolatency &&
   $ABS/.venv/bin/python tl.py 600 $ABS/.venv/bin/python bench3.py --system <name> --note "<one line>"`.
4. **Reproduce** `third_below`, `m1_filter_v5lo`, `m3_seq_v5`: headline_nd must match `EXPECTED.json`
   (0.5054 / 0.6206 / 0.6104, ±0.003). If not, stop and report.
5. **Seed-2 scorer, written once now:** `futures/v4_selfiter/seed2.py <name>` — copy `bench.heard`'s
   code with the seed `^ 5160` changed to `^ 5161`, assign it to `bench3.B2.heard` before building the
   system, compute `music_nd` exactly like `rescore_nd.py`, write `runs_mini/<run>/seed2/<name>.json`
   (refuse if it exists). Record its sha256 in `state_R0.json`. Score `m1_filter_v5lo` with it now.
6. **No permission prompts overnight.** Show Vincent, then write `$ABS/.claude/settings.json` with
   `permissions.allow` for `Bash(cd:*)`, `Bash(mkdir:*)`, `Bash(ls:*)`, `Bash(cat:*)`, `Bash(date:*)`,
   `Bash(df:*)`, `Bash(du:*)`, `Bash(diskutil info:*)`, `Bash(shasum:*)`, `Bash(env:*)`, `Bash(grep:*)`,
   the `.venv/bin/python` paths, `Write`, `Edit`; and `permissions.deny` for `Bash(rm:*)`,
   `Bash(git:*)`. Then run a 1-agent test Workflow that scores `third_below` with §2.3 and appends a
   heartbeat line. Repeat until it finishes with zero prompts. Ask Vincent to switch to **auto mode**
   before he leaves.

## 3. Phase B — blind listening (Vincent present, ~10 min)

Why: nobody has checked that higher `music_nd` sounds better. If the ears disagree, an overnight run on
the score is wasted.

1. Build `futures/v4_selfiter/listen/` with `make_listen.py` and a local page `ab.html` (open with
   `open listen/ab.html`; no server, no publishing).
   - **Render safely:** `sys.path.insert(0, '$ABS/futures/v3_nolatency'); import make_demos3 as D`.
     Importing is safe; never call `D.main()` or `D.songs()` (they write `audio/demos.json`). Use
     `D.cut`, `D.render(song, hs, T_b, path)` with paths under `listen/audio/`, `D.reed`, and
     `data.load()['split']['val']` for Bach val songs.
   - **Run every system on the FULL song** (`hs = bench3.run_song(system, song)`), then render the cut
     `D.cut(song, K)` with `hs[:len(cut['sop'])]`, `K = min(song['K'], int(12 / T_b))` at 80 bpm. Never
     run a system on a cut song: `m4_oracle`'s lookup keys on `K` and silently stops being an oracle.
     `m4_oracle` (val-only lookup in this bundle) works on Bach val songs only, not Twinkle.
   - Don't use bwv308 / bwv415 (earlier demos). Tied variant: copy `D.render` into your wrapper and
     merge consecutive equal Duet notes into one held note before `D.reed`.
   - **16 forced-choice pairs**, left/right randomized, key in `listen/key.json` (not in the page):
     4 × (`m1_filter_v5lo` vs `third_below`) on Twinkle + 3 Bach val;
     4 × (`m4_oracle` vs `m1_filter_v5lo`) on 4 Bach val;
     4 × (`m1_filter_v5lo` vs `m3_seq_v5`) on Twinkle + 3 Bach val — does copying her note sound bad?;
     3 × (`m1_filter_v5lo` tied vs re-attacked); 1 repeat of an earlier pair.
     Optional tags per pair: clashing / boring / jumpy / robotic. At the end the page shows a short
     result code; Vincent pastes it into the chat.
2. **Decide:**
   - **Agree** = Vincent prefers the higher-`music_nd` side in ≥ 3/4 of m1-vs-third_below AND ≥ 3/4 of
     oracle-vs-m1 → use `music_nd` as in §4. If the repeat pair flips, note "low confidence", proceed.
   - m1-vs-m3 is informative: if m3 wins ≥ 3/4, tell Vincent; lower the `double` ceiling to 0.05 only
     if he says yes.
   - **Disagree** → tell Vincent which comparison failed and what the tags say, propose one objective
     change, wait for his yes.
   - Tied wins ≥ 2/3 → all later renders use tied notes.
3. Save `listen/result.json`. Tell Vincent to switch to auto mode and that he can leave.

## 4. Phase C — the self-iteration loop (unattended)

### 4.1 Rounds

- **Between rounds, you (the orchestrator, not a workflow)** run `date`, check the 10 h / Monday 10:00
  limits, checksum (§1.2), disk (§1.6), read `hb/*.log`, and write `state_R<R>.json` (round,
  champion, every idea tried with a short hash and verdict, dead ideas).
- **Each round = one Workflow:** first 3 implementer agents with `parallel()` (one idea each from §4.4,
  never one already in state), then reviewer agents for candidates that pass §4.2 — so at most 3
  Python jobs run at once. No supervisor agent: stalls are bounded by `tl.py` and by caps below.
- **Implementer:** write `systems/r<R>_<idea>_v<k>.py` → score with §2.3 (nd + bench3) → append one
  line to `runs_mini/<run>/hb/<label>.log` before and after every command → ≤ 4 variants and ≤ 45 min
  (90 min if one training run is needed) → return name, numbers, one-sentence explanation.
- **Reviewer** (per candidate):
  (a) causality probe: on 24 val songs × 3 cut points, randomize every later pitch, later durations and
  all chord labels; outputs up to the cut must be identical;
  (b) cheat grep in the file and everything it imports from `systems/`: fail on `eval_sets`, `load()`,
  `_extra`, `bench/data`, `bach_beats`, `'val'`, `'test'`, `split`, `os.environ`, or any `open(` /
  `json.load` / `torch.load` other than its own `.pt` files in `systems/`. `build(train)` may use only
  `train`; `step` only its arguments and `start(meta)`;
  (c) seed 2 via `seed2.py` (champion scored once per champion change);
  (d) range: every Duet note in `theory.ALTO` (53–77) and above her heard note on ≤ 5 % of notes
  (measure with a wrapper; do not edit bench3).
- After a Workflow returns, log any agent whose last `hb` line is a command start older than that
  command's tl limit + 5 min in `hb/ALERTS_<R>.md`. An idea that fails twice (crash, timeout, no gain)
  is dead; never retry it.

### 4.2 Keep rule (the only one; never change it during the run)

A candidate becomes champion only if all hold:
- headline `music_nd` ≥ champion + **0.005**, AND **folk and pop** (170 and 300 val songs) each improve
  by ≥ **+0.003** — a gain carried by bach (17 songs) or pop909 (40) alone does not count;
- seed 2: (candidate − champion) ≥ +0.003;
- no dataset drops more than 0.01 below the champion;
- `double` (rescore_nd's per-dataset value) ≤ **0.08** on every dataset; no SLOW flag;
- `big` and `par` (bench3 rows) per dataset ≤ champion + 0.01. Starting values for `m1_filter_v5lo`
  (big/par): bach 0.003/0.001, folk 0.009/0.004, pop 0.011/0.003, pop909 0.007/0.004;
- reviewer (a)–(d) all ok.
Starting champion: `m1_filter_v5lo`, 0.6206. Report how many variants were scored in total.

### 4.3 Presentable

A champion is *presentable* only if it is **one chord guesser + `m1_filter_v5lo`'s chooser**. A chord
guesser is: the HMM; one GRU; the fixed 3-GRU average already in `m3_seq_v5` (`full` 96, `big` 128,
`s1` 96, probabilities averaged); or a log-linear mix of the HMM and one of these with one weight
$\alpha$. The chooser keeps its 6 `w_*` weights (note `w_unis` is inert because `below=1`); it may swap
one weight for another but not add one. You must be able to explain it in 3 plain sentences. Anything
else may be logged as "best score, not presentable" and never becomes the Tuesday model.

### 4.4 Ideas (in order)

The gap to the told-the-chord ceiling (0.839) is mostly *guessing the chord*. The GRU guesses chords
better (`fit` on bach 0.81 vs HMM 0.72) but its own chooser copies her up to 18 %; `m1_filter_v5lo`'s
chooser avoids copying.
- **Round 1 = E1 alone** `r1_gru_chooser`: subclass `systems.m1_filter_v5lo.Filter`. Hold
  `ens = systems.m3_seq_v4.Ens([_load('full',96), _load('big',128), _load('s1',96)], -1.0, dict(V3, step=0.01))`.
  In `start`: `super().start(meta); ens.start(meta)`. In `step`: `self._p = ens.chord_probs(onset, midi,
  prev_dur)` exactly once per note (it advances the GRU state), then `return super().step(...)`;
  `belief(k)` returns `self._p`. Build with `G(m1_filter_v5lo.build(train).m, ens)`,
  `torch.set_num_threads(1)`. A laptop check gave headline_nd **0.644** (bach 0.692, folk 0.713, pop
  0.590, pop909 0.581; double ≤ 0.05): reproduce within ±0.003. Also run the variant with the `full`
  GRU alone; if within 0.005 of the 3-GRU average, prefer the single GRU (easier to explain).
- **Wave 1** (rounds 2+): **E2** normalized $b_{HMM}^{1-\alpha} b_{GRU}^{\alpha}$, one $\alpha$;
  **E3** chord-duration prior from the meter (chords tend to change on beat 1, at a steady rate);
  **E4** choose by expected fit over the beats her note will cover; **E5** choose by expected
  consonance with her *true* note given 10 % mishearing.
- **Wave 2** only if wave 1 gave no presentable champion and ≥ 3 h remain: E6 retrain one GRU on the
  no-copy target (1 seed, ≤ 3600 s); E7 small beam search (width ≤ 8, ≤ 4 notes ahead). Log these as
  "not presentable" unless they meet §4.3 unchanged.
- **Skip:** melody retrieval, repeat-phrase memory (≈ +0.004), anything with new chord types.
- **Known limit, do not work on it:** the chord labels are the 7 basic triads of the major key, so 7th
  chords and the raised 7th note in minor-key songs count as wrong. State it in the report only.
- **Stop** when 2 consecutive rounds keep nothing, or 8 rounds, or 10 h, or Monday 10:00, or a checksum
  / disk stop.

## 5. Phase D — wrap-up for Tuesday (automatic when the loop stops)

Write into `runs_mini/<run>/report/` (self-contained, < 50 MB, Vincent AirDrops it back):
1. `MORNING.md` (Chinese, ≤ 1 page): champion vs 0.6206 per dataset; each idea, one line + verdict;
   alerts; listening result; total variants scored; "all numbers are val, tuned on val".
2. `board.txt`: every system ranked by headline `music_nd`, with `double`, `big`, `par`.
3. `audio/`: champion vs `m1_filter_v5lo` vs `third_below` on Twinkle + 2 Bach val chorales (tied
   notes if Phase B chose them).
4. **Two cards, always** — `CARD_m1_zh.md` for `m1_filter_v5lo`, and `CARD_new_zh.md` for the
   presentable champion if there is one. Each answers: input and output; data and training objective;
   who it is compared with, by how much, and why it wins; where it fails and why; next step. Each has
   "三句话讲清楚" and the 10 questions a professor is most likely to ask, with 2-sentence answers. Every
   term gets a one-line 中文（English） explanation and an everyday example. The leading tone appears
   only in the failure section: "our 7 chords are all built from the major scale, so in minor-key
   songs the raised 7th note that pulls up to the tonic (导音, leading tone) counts as wrong — a limit
   of the labels, not something we tuned."
5. **History, in `music_nd`:** score `m3_seq_v5h0`, `m3_seq_v5h4`, `m3_seq_v5h16`, `m3_seq_v5` (and the
   same windows for the champion if it uses the GRU; write new wrapper files, no env vars) and give:
   current note only → last 4 beats → last 16 beats → whole song. If you quote the old numbers
   (0.658 / 0.674 / 0.695 / 0.710 / 0.720) label them "raw music, copying counted". Repeat memory ≈
   +0.004.
6. `TEAM.md` (English, ≤ 1 page) for the teammates.
7. `QUIZ.md`: the 10 questions without answers + 5 numbers to recall (0.505, 0.621, champion, 0.839,
   what doubling means).
Then tell Vincent in Chinese: the champion, whether it is presentable, and that the quiz comes next.
Quiz him one question at a time in chat; he answers in his own words; point out only what is missing.
He passes with ≥ 8/10 and all 5 numbers within ±0.01. Only then does the new champion become the
Tuesday model; otherwise he presents `m1_filter_v5lo` with `CARD_m1_zh.md`.
