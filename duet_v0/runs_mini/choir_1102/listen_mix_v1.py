"""Listening files (run choir_1102): the singer's voice from a run's mic channel with Duet's choir placed where it is
heard in the headphones (LI + LO = 130.5 ms after the mic, the calibrated round trip the timing eval assumes). Two
files per run: 'before' takes the run's own output channel (the engine it was sung with), 'now' re-renders the notes
today's engine schedules when the run's mic is replayed through page v13 (choir_v6 defaults: one voice per part, start
notes traced). The choir gets the same gain in both files. Offline (tripwire). Writes
runs_mini/choir_1102/listen_v1/<k>_<song>_<run>_{before,now}.wav and listen_v1.json (today's chord timing per run).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py listen_v1 900 .venv/bin/python runs_mini/choir_1102/listen_mix_v1.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402  (tripwire, page v13, truth, replay, NOTES)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

E = V.E
OUT = os.path.join(E.ABS, 'runs_mini/choir_1102/listen_v1')
RUNS = [('132458_r2_mix_v2', 'v11'), ('124132_r2_mix_v2', 'v11'), ('114844_r2_mix_v2', 'twinkle')]
DELAY = int(round((E.LI + E.LO) * E.SR))
VOICE_RMS, CHOIR_RMS = 0.10, 0.08


def gain(y, rms):
    """Gain that gives y's loud part (above -40 dB of its peak) this RMS."""
    a = np.abs(y)
    loud = y[a > a.max() * 0.01]
    return rms / (float(np.sqrt(np.mean(loud ** 2))) + 1e-9)


def mix(voice, choir):
    y = np.zeros(max(len(voice), DELAY + len(choir)), np.float32)
    y[:len(voice)] += voice
    y[DELAY:DELAY + len(choir)] += choir
    return y / max(1.0, float(np.abs(y).max()) / 0.95)


def render(eng, n):
    """The choir notes the engine actually sounded (rendered at least in part), at their final lengths."""
    v = type(eng.choir)(level=eng.choir.level)
    for idx, song, nt in V.NOTES:
        if nt is not None and nt.done > 0:
            v.schedule(nt.i0, nt.dur, nt.midi)
    return v.render(n)


if __name__ == '__main__':
    os.makedirs(OUT)
    E.P3.MODELS.load()
    lib = E.P13.P12.P5.CHOIR
    res = {}
    for k, (run_name, sid) in enumerate(RUNS, 1):
        y = sf.read(os.path.join(E.ABS, 'runs_mini/live', run_name, 'run.wav'), dtype='float32', always_2d=True)[0]
        x, before = y[:, 0], y[:, 1]
        V.NOTES.clear()
        eng, _ = E.replay(x, {})
        now = render(eng, len(x) + E.SR)
        gv, gc = gain(x, VOICE_RMS), gain(before, CHOIR_RMS)
        for tag, choir in (('before', before), ('now', now)):
            sf.write(os.path.join(OUT, f'{k}_{sid}_{run_name[:6]}_{tag}.wav'), mix(x * gv, choir * gc), E.SR,
                     subtype='PCM_16')
        sung = E.true_notes(x)
        score = [(float(a), int(m) + (eng.S or 0)) for a, b, m in lib[sid]['melody']]
        truth = E.match(score, sung, lib[sid]['bpm'] if sid == 'v11' else 60.0 / eng.Tb)
        log = [(nt.i0, idx, song) for idx, song, nt in V.NOTES if nt is not None]
        r = E.score_run(eng, log, truth)
        res[run_name] = {key: r[key] for key in ('song', 'chords', 'chord_med_abs_from_target',
                                                 'chord_p90_abs_from_target', 'per_chord')}
        res[run_name]['sounded_notes'] = sum(1 for _, _, nt in V.NOTES if nt is not None and nt.done > 0)
        print(run_name, json.dumps({key: v for key, v in res[run_name].items() if key != 'per_chord'}), flush=True)
    with open(os.path.join(OUT, 'listen_v1.json'), 'x') as fh:
        json.dump(res, fh, indent=1)
