# Logic Pro as Duet's voice

Duet's built-in voice is a soft synthetic "oo". For a nicer machine voice, Duet can send its
harmony line as MIDI to Logic Pro (Creator Studio), and a choir or vocal instrument in Logic
sings it. Duet still does all the listening, following and timing. Logic only turns the notes it
receives into sound.

> **Not verified in Logic.** Logic was never opened while this was built. The menu names below
> follow Logic Pro 10.7–11 and may differ slightly in your version. The Duet side *was* tested:
> a second process listening on a virtual MIDI port received every note_on within ±2 ms of its
> scheduled time (`live/tests/run_tests.py` section 11, numbers in `live/README.md`).

## How it works

`--midi-out` creates a **virtual MIDI source called "Duet"** (CoreMIDI). Any app can see it
while Duet runs. Each machine note becomes a `note_on` at its scheduled time and a `note_off` at
its end, on channel 1 by default. The servo nod is also sent as **CC 20** (lift ≈ 72, dip ≈ 51,
rest ≈ 64), in case you want to drive visuals from it.

Duet sends each note **early by `--midi-latency`** (default 10 ms), so Logic's own delay (audio
buffer, plus the instrument's attack) is absorbed and the note *sounds* on time. When MIDI drives
an external synth, Duet mutes its built-in voice in the mac and device backends.

## One-time setup in Logic

1. **Start Duet first** (the "Duet" source has to exist), for example:
   ```bash
   .venv/bin/python live/duet_live.py run --ref live/cache/song --backend mac --midi-out --view
   ```
   Logic picks up new MIDI sources while it is running, so you can also open Logic first.
2. **Audio settings** (Logic Pro › Settings › Audio › Devices):
   * set **Output Device** to the headphones or speakers you will use;
   * set **I/O Buffer Size to 64 or 128 samples**. Smaller is lower latency. If you hear
     crackles, use 128.
   * Note the **Resulting Output Latency** shown under the buffer setting. You will need it in step 6.
3. **New project › Software Instrument** track.
4. **Choose a vocal patch.** In the Library (press `Y`), pick a choir or vocal sound, for
   example something under *Orchestral › Choir*, or an Alchemy "Vox"/"Choir" patch. Names depend
   on which sound packs are installed. Prefer a patch with a **fast attack**. Pad-like choirs
   swell in over 100 ms or more, so a note would be *heard* late even if it starts on time. If the
   patch has an Attack control, turn it down.
5. **Route the "Duet" input to this track.** Open the track inspector (`I`) and set **MIDI In
   Port** to **Duet**; keep the channel on All (or 1). Older Logic versions have no per-track
   setting. They feed *all* MIDI inputs to the selected instrument track, which also works.
6. **Arm the track.** Select it, then **record-enable it (`R`) or turn on Input Monitoring**.
   Logic plays the incoming notes live either way. Record-enable it if you also want the
   performance captured as MIDI.
7. **Tell Duet Logic's latency:** `--midi-latency` = Resulting Output Latency from step 2, plus
   about 1 ms for MIDI, plus your patch's attack time (the time until the note is clearly heard).
   For example, a 128-sample buffer at 48 kHz gives about 6 ms, so try `--midi-latency 0.008`.

## Performing

```bash
# MacBook mic + Logic through headphones
.venv/bin/python live/duet_live.py run --ref live/cache/song --backend mac --midi-out --midi-latency 0.008 --view

# ESP32 board (mic + servo nod on the board, voice from Logic on the Mac)
.venv/bin/python live/duet_live.py run --ref live/cache/song --backend device --port /dev/cu.usbmodem1101 \
    --midi-out --midi-latency 0.008
```

Things to know:

* **Use headphones** with Logic as the voice. The echo guard's canceller needs the exact signal
  that reaches the speaker, and Logic's choir is not that signal. With MIDI out, Duet therefore
  runs only the pitch mask (it reads the machine's pitch from its own muted voice) and skips the
  sound check.
* The board's speaker stays silent (its built-in voice is muted). Add `--synth on` if you want
  both voices.
* Duet raises its output latency to at least `in-latency + midi-latency + 8 ms`, so every note
  is decided before it has to be sent. Prediction makes this free (README: latency sweep).
* `--midi-channel N` sets the channel. `--nod-cc -1` turns off the nod CC. `--midi-connect IAC`
  sends to an existing destination (e.g. an IAC bus) instead of creating "Duet".
* **Logic hears every MIDI source by default.** While Duet runs with `--midi-out`, the selected
  instrument track of any open Logic project will play it. This is intended, but keep it in mind.
* Calibrating `--midi-latency` by ear: record the session with Logic recording the instrument
  track's audio (or with `--backend mac`'s `session.wav`), and compare a machine note's audible
  start with the moment the matching built-in note would start (`summary.json`, `notes_out`).
  Then change `--midi-latency` by the difference.

## Rendering the exported demo MIDI files

The current demos (follower v2) are in **`media/demo/`**. `live/demo/` holds the older phase-2 (follower
v1) demos and is kept only for the record. `media/demo/` has, for each demo performance:

| file | what |
|---|---|
| `<name>.mid` | type-1 MIDI: a tempo track, **"Singer (melody)"** (ch 1), **"Duet (harmony)"** (ch 2, plus the nod as CC 20) |
| `<name>_duet.wav` | the run itself: left = the singer as recorded, right = Duet's built-in voice |
| `<name>_midi_render.wav` | the `.mid` rendered back with the built-in voice (left melody, right harmony): a quick listen without Logic |
| `<name>_excerpt20s.mid` / `.wav` | a 20 s excerpt of the same (the window rule is in `media/INDEX.md`) |

Only the `.mid` files are in git. The `.wav` files are git-ignored (the real take contains dataset audio);
regenerate them with `.venv/bin/python media/tools/make_demo.py <scratch dir>`, which needs the datasets.
For any run of your own, `docs/guide/tools/run_report.py live/runs/<name>` writes the same kind of
`duet.mid` into the run folder.

All note positions in the `.mid` are **real seconds from the start of the recording**. The file
has one tempo (the rehearsal beat, scaled to this performance), so bars line up roughly. Seconds
are exact everywhere: 960 ticks per beat is about 0.7 ms per tick.

1. **File › Open** the `.mid`. Logic creates a project with the file's tempo and one track per MIDI track.
   (Dragging the file into an existing project also works. Accept Logic's offer to import the
   tempo, or the notes will not land at the right seconds.)
2. Give each track an instrument from the Library. For example: a solo voice or soft lead for
   "Singer (melody)", and the choir patch for "Duet (harmony)". Or **mute the melody track** and import
   the real singer instead: **File › Import › Audio File…** `<name>_duet.wav`, placed at the
   project start (1 1 1 1), then pan/mute the right channel (Duet's built-in voice) or keep
   only the left channel. The harmony track then lines up with her real voice.
3. Check that the project starts at 1 1 1 1 with no count-in offset, so time 0 of the wav equals
   time 0 of the MIDI.
4. **File › Bounce › Project or Section…** (⌘B) to write a wav or m4a.

The demos in `media/demo/` are:

* `greensleeves_natural`: synthetic singer, exact ground truth; 26/26 harmony notes, median 20.9 ms.
* `tebe_poem_S1_take02`: **the primary real demo**, a Dagstuhl ChoirSet soprano take (*Tebe Poem*, full
  choir, S1, rehearsal Take01, performance Take02) run with the evaluation's front end
  (`--gate-db 15 --bias dcs`): 23.8 % of plan notes within 60 ms, median +130 ms late.
* `tebe_poem_S1_take02_gate_only`: the same take with the input gate only (14.3 %).
* `tebe_poem_S1_take02_live_defaults`: the same take with the live app's defaults, a failure case: Duet
  sings the second section about 3 s early (2.4 %). `live/README.md` ("Input gate and detector bias")
  explains why.

Dagstuhl ChoirSet is CC BY-NC 4.0: non-commercial use only, with attribution. Credit it as in
`media/demo/CREDITS.md` ("Audio: Dagstuhl ChoirSet (Rosenzweig et al., TISMIR 2020), CC BY-NC 4.0").
The old `live/demo/dcs_locus_iste_S1_take02` files are a phase-2 (v1) failure case; do not use them in
the deck.
