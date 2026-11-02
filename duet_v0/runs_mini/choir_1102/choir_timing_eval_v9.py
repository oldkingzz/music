"""v9 (middle settings between steady and fast, all adaptive, no phase jump). v8 (adds the phase jump: a miss beyond 1/4 beat realigns at once). v7 (adaptive tempo: steady by default, fast after 3 same-side misses; the a cappella rows report the follower's
own prediction errors, the offline truth match does not work on his narrow-interval singing). v6 (adds Vincent's a cappella Twinkle runs, scored in the key the follower found, and robust Theil-Sen tempo
variants). v5 (compares steadier tempo estimates: longer window, less recency, slower gain). v4 (compares the duration weighting of beat-map updates). v3 of choir_timing_eval_v2.py: the first melody note must match exactly (pitch class within 0.6 semitone, held
0.5 s; Twinkle runs start with the phone file's count-in clicks a semitone below C), per-chord errors are listed, and
more settings are compared. Where does the choir's timing error on v11 come from, and which GENERAL follower settings reduce it? (run choir_1102)
No song-specific timing: the v11 score stays the snapped transcription (choir_songs_v5.json); only follower settings vary.

For Vincent's phone runs (v11: 113335, 114908; Twinkle: 113217, 114844) the TRUE onsets of the melody in the run are
measured offline: librosa pyin on the run's microphone channel, steady voiced runs, a scoop shorter than 80 ms merged into
the note it reaches (onset = scoop start); matched in order to the score's melody notes. Each run is replayed through
page v13's choir (offline, tripwire, the run's calibrated latency) with a set of follower settings; per run:
- detect: live onset of each matched note - its true onset (how well the live segmentation hears her onsets);
- chord: when each part chord on a melody onset sounds - that note's true onset (target -35 ms = the engine's lead);
  reported as median, and median / p90 of the distance to the target.

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/choir_timing_eval_v2.py
"""
import sys
import types

ABS = '/Users/san/claude_sandbox/music/duet_v0'


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('offline (sounddevice.' + name + ')')


for _m in ('sounddevice', '_sounddevice'):
    sys.modules[_m] = _NoSound(_m)

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import live_panel_v13 as P13  # noqa: E402
import choir_v6 as CV6  # noqa: E402
P3 = P13.P3
from dsp import SR  # noqa: E402

ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_choir', time.strftime('%H%M%S') + '_teval9')
OUT = os.path.join(ABS, 'runs_mini/choir_1102/choir_timing_eval_v9.json')
RUNS = {'113335_r2_mix_v2': 'v11', '114908_r2_mix_v2': 'v11', '113217_r2_mix_v2': 'twinkle',
        '114844_r2_mix_v2': 'twinkle', '095417_r2_mix_v2': 'twinkle', '110255_r2_mix_v2': 'twinkle'}
LI, LO = 0.1305 - 0.0421, 0.0421
LOG = []
_init = CV6.ChoirStart.__init__


def _init_log(self, *a, **k):
    _init(self, *a, **k)
    sch = self.choir.schedule

    def schedule(start, dur, midi, tag=None):
        LOG.append((start, tag[1] if tag else None, self.song_id))
        return sch(start, dur, midi, tag=tag)
    self.choir.schedule = schedule


CV6.ChoirStart.__init__ = _init_log


def true_notes(x):
    f0, vf, _ = librosa.pyin(x, fmin=70, fmax=800, sr=SR, frame_length=2048, hop_length=240)
    t = librosa.times_like(f0, sr=SR, hop_length=240)
    midi = np.where(vf, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    segs, cur = [], []
    for i, m in enumerate(midi):
        if np.isnan(m) or (cur and abs(m - np.median(midi[cur[-6:]])) > 0.8):
            if len(cur) >= 3:
                segs.append(cur)
            cur = [] if np.isnan(m) else [i]
            continue
        cur.append(i)
    if len(cur) >= 3:
        segs.append(cur)
    notes = [[float(t[s[0]]) - LI, float(t[s[-1]]) - LI, float(np.median(midi[s[2:] if len(s) > 4 else s]))]
             for s in segs]
    merged = []
    for n in notes:                                       # a short scoop into the next note: that note starts there
        if merged and merged[-1][1] - merged[-1][0] < 0.08 and n[0] - merged[-1][1] < 0.05 \
                and abs(n[2] - merged[-1][2]) < 2.5:
            n = [merged[-1][0], n[1], n[2]]
            merged.pop()
        merged.append(n)
    return [n for n in merged if n[1] - n[0] >= 0.06]


def match(score, sung, bpm):
    """Score melody notes (beat, midi) -> true onset, in order (pitch class +-1, within 0.6 s of a line through the
    last match at the song's tempo)."""
    out, j, anchor = {}, 0, None
    tb = 60.0 / bpm
    for b, m in score:
        if anchor is None:
            k0 = next((k for k, (a, e, p) in enumerate(sung) if min((p - m) % 12, (m - p) % 12) <= 0.6
                       and e - a > 0.5), None)
            if k0 is None:
                return out
            out[b], j, anchor = sung[k0][0], k0 + 1, (sung[k0][0], b)
            continue
        want = anchor[0] + (b - anchor[1]) * tb
        for k in range(j, len(sung)):
            a, e, p = sung[k]
            if a > want + 0.6:
                break
            if abs(a - want) <= 0.6 and min((p - m) % 12, (m - p) % 12) <= 1.0:
                out[b], j, anchor = a, k + 1, (a, b)
                break
    return out


def replay(x, settings):
    old = {k: getattr(CV6.ChoirStart, k) for k in settings}
    for k, v in settings.items():
        setattr(CV6.ChoirStart, k, v)
    try:
        LOG.clear()
        s = P3.Session(ROOT)
        s.cals[('m', 'p')] = {'in': 'm', 'out': 'p', 'rt_s': LI + LO, 'rt_ms': (LI + LO) * 1000, 'n_used': 8,
                              'spread_ms': 5, 'when': 't', 'stream_Li_ms': 62.7, 'stream_Lo_ms': 42.1}
        s.start({'song': 'auto', 'bpm': 80.0, 'mode': 'choir', 'click': False, 'input': 'mic'}, source=x,
                sim={'devs': ('m', 'p'), 'latency': (0.0627, LO)})
        t0 = time.time()
        while s.state != 'await' and time.time() - t0 < 120:
            time.sleep(0.05)
        eng = s.eng
        s.verdict({'run': s.status()['run'], 'verdict': '作废', 'note': 'timing eval v2'})
        time.sleep(1.1)
        return eng, list(LOG)
    finally:
        for k, v in old.items():
            setattr(CV6.ChoirStart, k, v)


def score_run(eng, log, truth):
    chord, detect = [], []
    for start, idx, sid in log:
        if sid != eng.song_id or idx is None or idx >= len(eng.parts):
            continue
        on = eng.parts[idx][0]
        if on in truth and on > 0:
            chord.append((start / SR + eng.Lo) - truth[on])
    for t, b in eng.pairs:
        if b in truth:
            detect.append(t - truth[b])
    c, d = np.array(chord) * 1000, np.array(detect) * 1000
    f = lambda a, fn: round(float(fn(a))) if len(a) else None  # noqa: E731
    per = sorted((round(on, 2), round(float(v))) for on, v in zip(
        [eng.parts[idx][0] for start, idx, sid in log if sid == eng.song_id and idx is not None and idx < len(eng.parts)
         and eng.parts[idx][0] in truth and eng.parts[idx][0] > 0], c))
    return {'per_chord': per, 'song': eng.song_id, 'chords': len(c), 'chord_median_ms': f(c, np.median),
            'chord_med_abs_from_target': f(np.abs(c + 35), np.median),
            'chord_p90_abs_from_target': f(np.abs(c + 35), lambda a: np.percentile(a, 90)),
            'detect_median_ms': f(d, np.median), 'detect_med_abs_ms': f(np.abs(d - np.median(d)) if len(d) else d,
                                                                         np.median),
            'n_detect': len(d), 'tempo': round(60 / eng.Tb, 1)}


BASE = {'DUR_WEIGHT': False}
VARIANTS = {
    'A before (fast always)': dict(BASE, TEMPO_WINDOW=4, TEMPO_MIN_SPAN=1.5, RECENCY=True, TEMPO_GAIN=0.6,
                                   PHASE_GAIN=0.8, TEMPO_ROBUST=False, TEMPO_ADAPTIVE=False, PHASE_JUMP=0),
    'F adaptive (8/4, 0.3, 0.6)': dict(BASE, TEMPO_WINDOW=8, TEMPO_MIN_SPAN=4.0, RECENCY=False, TEMPO_GAIN=0.3,
                                       PHASE_GAIN=0.6, TEMPO_ROBUST=False, TEMPO_ADAPTIVE=True, PHASE_JUMP=0),
    'I adaptive (6/3, 0.45, 0.7)': dict(BASE, TEMPO_WINDOW=6, TEMPO_MIN_SPAN=3.0, RECENCY=False, TEMPO_GAIN=0.45,
                                        PHASE_GAIN=0.7, TEMPO_ROBUST=False, TEMPO_ADAPTIVE=True, PHASE_JUMP=0),
    'J adaptive (8/4, 0.3, 0.75)': dict(BASE, TEMPO_WINDOW=8, TEMPO_MIN_SPAN=4.0, RECENCY=False, TEMPO_GAIN=0.3,
                                        PHASE_GAIN=0.75, TEMPO_ROBUST=False, TEMPO_ADAPTIVE=True, PHASE_JUMP=0),
}

if __name__ == '__main__':
    P3.MODELS.load()
    lib = P13.P12.P5.CHOIR
    res = {}
    for run_name, sid in RUNS.items():
        x = sf.read(os.path.join(ABS, 'runs_mini/live', run_name, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        sung = true_notes(x)
        for vname, st in VARIANTS.items():
            eng, log = replay(x, st)
            score = [(float(a), int(m) + (eng.S or 0)) for a, b, m in lib[sid]['melody']]
            truth = match(score, sung, lib[sid]['bpm'] if sid == 'v11' else 60.0 / eng.Tb)
            if vname.startswith('A'):
                print(f'{run_name} ({sid}, shift {eng.S}): {len(truth)} of {len(score)} melody notes found', flush=True)
            r = score_run(eng, log, truth)
            pe = np.abs(eng.diag['pred_err_ms'])
            r.update({'pred_med': round(float(np.median(pe))) if len(pe) else None,
                      'pred_p90': round(float(np.percentile(pe, 90))) if len(pe) else None,
                      'trend_updates': eng.diag.get('trend_updates', 0)})
            res[f'{run_name} | {vname}'] = r
            print(f'   {vname:32s}', json.dumps({k: v for k, v in r.items() if k != 'per_chord'}), flush=True)
            print('      per chord (beat, ms):', r['per_chord'], flush=True)
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1)
