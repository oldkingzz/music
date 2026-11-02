"""v4: trace any v11 phone runs (run choir_1102; argv = run names, default the 13:24 run) with the current engine
(one voice per part). Per melody note: true onset (offline pyin), live - true, prediction - true, bpm after; per chord on
a melody onset: sounds - true (target -35 ms); the one-voice stats; and a check that the replay matches what the page
really played: onsets of the run's own output channel (channel 1) against the replay's part-note starts.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py v11trace4 600 .venv/bin/python runs_mini/choir_1102/v11_trace_v4.py [run ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402  (tripwire, page v13, NOTES hook, voice_stats)

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

E = V.E
RUNS = sys.argv[1:] or ['132458_r2_mix_v2']
E.P3.MODELS.load()
lib = E.P13.P12.P5.CHOIR
score = [(float(a), int(m)) for a, b, m in lib['v11']['melody']]
_acc = E.CV6.ChoirStart._accept
TRACE = []


def acc(self, note, c):
    b = self.mel[c][0]
    pred = self.t_of(b)
    _acc(self, note, c)
    TRACE.append((b, note['t'], pred, 60 / self.Tb, note['pitch']))


E.CV6.ChoirStart._accept = acc
for run in RUNS:
    w = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0]
    x = w[:, 0]
    sung = E.true_notes(x)
    truth = E.match(score, sung, lib['v11']['bpm'])
    TRACE.clear()
    V.NOTES.clear()
    eng, _ = E.replay(x, {})
    print(f'== {run}: song {eng.song_id}, shift {eng.S}, low {eng.low}, start {eng.diag.get("start_t")}, '
          f'id {eng.diag.get("id_time")}, tempo end {60 / eng.Tb:.1f}, stopped {eng.stopped}')
    print('   sung (offline, engine time):', [(round(a, 2), round(p, 1)) for a, e, p in sung[:30]])
    print(f"{'beat':>6} {'pitch':>6} {'true':>7} {'live-tr':>8} {'pred-tr':>8} {'bpm':>6}")
    for b, t, pred, bpm, p in TRACE:
        tr = truth.get(b)
        f = lambda v: '-' if tr is None else str(round((v - tr) * 1000))  # noqa: E731
        print(f"{b:6.2f} {p:6.1f} {'-' if tr is None else round(tr, 3):>7} {f(t):>8} {f(pred):>8} {bpm:6.1f}")
    seen = {}
    for idx, song, nt in V.NOTES:
        if song != eng.song_id or nt is None or idx is None:
            continue
        on, off, m, part = eng.parts[idx]
        seen.setdefault(on, []).append((part, round(nt.midi), round(nt.i0 / E.SR + eng.Lo, 3), round(nt.dur, 3)))
    print('chords (beat: sounds - true onset ms | parts):')
    for on in sorted(seen):
        tr = truth.get(on)
        s0 = min(v[2] for v in seen[on])
        print(f'  {on:6.2f}: {"-" if tr is None or on == 0 else round((s0 - tr) * 1000):>6} | '
              + ' '.join(f'{p}{m}' for p, m, s, d in sorted(seen[on])))
    print('   voices:', V.voice_stats(eng))
    # what the page really played: onsets in the output channel vs the replay's part-note starts (stream time)
    if w.shape[1] > 1:
        y = w[:, 1].astype(float)
        env = librosa.onset.onset_strength(y=y, sr=E.SR, hop_length=240)
        pk = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=float(np.max(env)) * 0.08,
                                    wait=8)
        out_on = librosa.frames_to_time(pk, sr=E.SR, hop_length=240)
        starts = sorted({round(nt.i0 / E.SR, 3) for idx, song, nt in V.NOTES if nt is not None and nt.done > 0})
        d = [min(out_on - s, key=abs) for s in starts] if len(out_on) else []
        print('   output channel onsets vs replay starts (ms, stream time):',
              [round(v * 1000) for v in d][:40], ' rms out', round(float(np.sqrt(np.mean(y ** 2))), 4))
