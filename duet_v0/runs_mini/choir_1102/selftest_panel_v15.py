"""Offline selftest of page v15 (run choir_1102): with Ode to Joy as a third library song,
1. the Ode phone file (3 s of silence, then the synthetic singer) and two re-sung versions of it (5 semitones down at
   84 bpm; an octave down at 104 bpm, like a man singing) are recognised as 'ode' and followed: chord timing against
   the true onsets (target -35 ms) and the share of sung time with a sounded choir note a semitone away;
2. every earlier Twinkle / v11 run is still recognised as before, with its chord timing (choir_timing_eval_v14 'S now':
   v11 medians 57 78 48 49 10, Twinkle 24 22 30 ms from target).
Tripwire (no sound); each run with the latency it was sung with.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py selftest_v15 1500 .venv/bin/python runs_mini/choir_1102/selftest_panel_v15.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402  (tripwire, page v13, truth, replay, NOTES)
import live_panel_v15 as P15  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from dsp import render_voice  # noqa: E402

E = V.E
LO = 0.0421
OUT = os.path.join(E.ABS, 'runs_mini/choir_1102/selftest_panel_v15.json')
RUNS = {'203926_r2_mix_v2': ('v11', 0.174), '195940_r2_mix_v2': ('v11', 0.174), '113335_r2_mix_v2': ('v11', 0.1305),
        '114908_r2_mix_v2': ('v11', 0.1305), '124132_r2_mix_v2': ('v11', 0.1305),
        '132458_r2_mix_v2': ('v11', 0.1305), '195914_r2_mix_v2': ('twinkle', 0.174),
        '113217_r2_mix_v2': ('twinkle', 0.1305), '114844_r2_mix_v2': ('twinkle', 0.1305),
        '110255_r2_mix_v2': ('twinkle', 0.1305), '095417_r2_mix_v2': ('twinkle', 0.1305)}


def semitone_share(eng, x):
    """Share of voiced frames where a sounded choir note is a semitone (or major 7th) from the sung pitch."""
    import librosa
    f0, vf, _ = librosa.pyin(x, fmin=70, fmax=800, sr=E.SR, frame_length=2048, hop_length=240)
    t = librosa.times_like(f0, sr=E.SR, hop_length=240) - E.LI
    midi = librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))
    tune = float(np.median((midi[vf] - np.round(midi[vf]) + 0.5) % 1.0 - 0.5))
    snd = [(nt.i0 / E.SR + eng.Lo, nt.i0 / E.SR + eng.Lo + nt.dur, nt.midi)
           for _, _, nt in V.NOTES if nt is not None and nt.done > 0 and nt.dur > 0]
    bad = tot = 0
    for i in np.flatnonzero(vf):
        live = [c for a, e, c in snd if a <= t[i] < e]
        if not live:
            continue
        m = round(midi[i] - tune)
        tot += 1
        bad += any((m - c) % 12 in (1, 11) for c in live)
    return round(bad / max(tot, 1), 3), tot


def run(x, sid, rt):
    E.LI, E.LO = rt - LO, LO
    V.NOTES.clear()
    eng, _ = E.replay(x, {})
    r = {'song': eng.song_id, 'want': sid, 'shift': eng.S}
    if eng.song_id == sid:
        lib = P15.P5.CHOIR
        score = [(float(a), int(m) + (eng.S or 0)) for a, b, m in lib[sid]['melody']]
        truth = E.match(score, E.true_notes(x), lib[sid]['bpm'] if sid == 'v11' else 60.0 / eng.Tb)
        log = [(nt.i0, idx, song) for idx, song, nt in V.NOTES if nt is not None]
        s = E.score_run(eng, log, truth)
        r.update({k: s.get(k) for k in ('chords', 'chord_med_abs_from_target', 'chord_p90_abs_from_target')})
        r['semitone_share'], r['frames'] = semitone_share(eng, x)
    return r


if __name__ == '__main__':
    E.P3.MODELS.load()
    assert sorted(P15.P5.CHOIR) == ['ode', 'twinkle', 'v11'] and 'ode' in P15.P3.SONGS
    res = {}
    ode = P15.P5.CHOIR['ode']
    x0 = sf.read(os.path.join(E.ABS, P15.ODE['wav']), dtype='float32', always_2d=True)[0][:, 0]
    variants = {'phone file (C, 96 bpm)': x0}
    for name, st, bpm in (('5 down, 84 bpm', -5, 84.0), ('octave down, 104 bpm', -12, 104.0)):
        sb = 60.0 / bpm
        notes = [(3.0 + a * sb, (b - a) * sb - 0.03, m + st) for a, b, m in ode['melody']]
        y = render_voice(notes, int((5.0 + ode['K'] * sb) * E.SR))
        variants[name] = (y * 0.6 / np.abs(y).max()).astype(np.float32)
    for name, x in variants.items():
        res['ode | ' + name] = r = run(x, 'ode', 0.174)
        print('ode |', name, json.dumps(r), flush=True)
    for name, (sid, rt) in RUNS.items():
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', name, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        res[name] = r = run(x, sid, rt)
        print(name, json.dumps(r), flush=True)
    wrong = [k for k, r in res.items() if r['song'] != r['want']]
    print('recognised wrong:', wrong or 'none', flush=True)
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1)
