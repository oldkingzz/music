"""Where do the choir's chords sound against v11's TRUE melody onsets? (run choir_1102)

For Vincent's v11 phone runs (the vocadito-11 recording played into the AirPods mic), the true onsets of her notes in
the run are measured offline (pyin on the run's microphone channel, steady voiced runs), matched in order to the score's
melody notes. The run is then replayed through page v13's choir (offline, tripwire, the run's calibrated latency) with a
given song library, logging when each part note is written; a chord sounds at write time + Lo. Reported per library:
chord time - true onset of the melody note on the same score beat (target: -35 ms, the engine's lead), median and p90 of
its distance to the target; and the follower's own prediction errors.

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/v11_timing_eval_v1.py
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
P3, P5 = P13.P3, P13.P12.P5
from dsp import SR  # noqa: E402

ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_choir', time.strftime('%H%M%S') + '_v11eval')
LOG = []
_init = CV6.ChoirStart.__init__


def _init_log(self, *a, **k):
    _init(self, *a, **k)
    sch = self.choir.schedule

    def schedule(start, dur, midi, tag=None):
        LOG.append((start, tag[1] if tag else None))
        return sch(start, dur, midi, tag=tag)
    self.choir.schedule = schedule


CV6.ChoirStart.__init__ = _init_log


def true_onsets(x, li):
    """Offline pyin notes of the microphone channel, in engine time (input time - Li)."""
    f0, vf, _ = librosa.pyin(x, fmin=70, fmax=800, sr=SR, frame_length=2048, hop_length=240)
    t = librosa.times_like(f0, sr=SR, hop_length=240)
    midi = np.where(vf, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    notes, cur = [], []
    for i, m in enumerate(midi):
        if np.isnan(m) or (cur and abs(m - np.median(midi[cur[-6:]])) > 0.8):
            if len(cur) >= 5:
                notes.append(cur)
            cur = [] if np.isnan(m) else [i]
            continue
        cur.append(i)
    if len(cur) >= 5:
        notes.append(cur)
    return [(float(t[n[0]]) - li, float(np.median(midi[n]))) for n in notes]


def match(score, sung, t_first):
    """Score melody notes (beat, midi) -> sung onset, in order: each score note takes the first unused sung note within
    +-1 semitone (pitch class) after the previous match and within 0.6 s of where a 122.45 bpm line from her first
    note puts it (re-anchored on each match)."""
    out, j, anchor_t, anchor_b = {}, 0, t_first, 0.0
    tb = 60.0 / 122.45
    for b, m in score:
        want = anchor_t + (b - anchor_b) * tb
        best = None
        for k in range(j, len(sung)):
            t, p = sung[k]
            if t > want + 0.6:
                break
            if abs(t - want) <= 0.6 and min((p - m) % 12, (m - p) % 12) <= 1.0:
                best = k
                break
        if best is not None:
            out[b] = sung[best][0]
            j, anchor_t, anchor_b = best + 1, sung[best][0], b
    return out


def run(x, lib):
    P5.CHOIR = lib
    LOG.clear()
    s = P3.Session(ROOT)
    s.cals[('m', 'p')] = {'in': 'm', 'out': 'p', 'rt_s': 0.1305, 'rt_ms': 130.5, 'n_used': 8, 'spread_ms': 5,
                          'when': 't', 'stream_Li_ms': 62.7, 'stream_Lo_ms': 42.1}
    s.start({'song': 'auto', 'bpm': 80.0, 'mode': 'choir', 'click': False, 'input': 'mic'}, source=x,
            sim={'devs': ('m', 'p'), 'latency': (0.0627, 0.0421)})
    t0 = time.time()
    while s.state != 'await' and time.time() - t0 < 120:
        time.sleep(0.05)
    eng = s.eng
    s.verdict({'run': s.status()['run'], 'verdict': '作废', 'note': 'v11 timing eval'})
    time.sleep(1.1)
    return eng, list(LOG)


P3.MODELS.load()
libs = {}
for name in ('choir_songs_v5.json', 'choir_songs_v6.json'):
    with open(os.path.join(ABS, 'futures/v4_selfiter', name)) as fh:
        libs[name] = json.load(fh)['songs']
res = {}
for run_name in ('113335_r2_mix_v2', '114908_r2_mix_v2'):
    w, sr = sf.read(os.path.join(ABS, 'runs_mini/live', run_name, 'run.wav'), dtype='float32', always_2d=True)
    x = w[:, 0]
    li = 0.1305 - 0.0421
    sung = true_onsets(x, li)
    for name, lib in libs.items():
        eng, log = run(x, lib)
        if eng.song_id != 'v11':
            print(run_name, name, 'not v11:', eng.song_id)
            continue
        score = [(float(a), int(m)) for a, b, m in lib['v11']['melody']]
        first = next((t for t, p in sung if abs(p - (eng.ref_her() if eng.ton else p)) < 1.0), sung[0][0])
        truth = match(score, sung, first)
        errs = []
        for start, idx in log:
            on = eng.parts[idx][0]
            if on in truth and on > 0:
                errs.append((start / SR + eng.Lo) - truth[on])
        e = np.array(errs) * 1000
        pe = np.abs(eng.diag['pred_err_ms'])
        r = {'matched_truth': len(truth), 'chords': len(e), 'median_ms': round(float(np.median(e))) if len(e) else None,
             'p90_abs_from_target_ms': round(float(np.percentile(np.abs(e + 35), 90))) if len(e) else None,
             'median_abs_from_target_ms': round(float(np.median(np.abs(e + 35)))) if len(e) else None,
             'pred_err_median_abs_ms': round(float(np.median(pe))) if len(pe) else None, 'tempo': round(60 / eng.Tb, 1),
             'errs_ms': [round(v) for v in e]}
        res[f'{run_name} {name}'] = r
        print(run_name, name, json.dumps({k: v for k, v in r.items() if k != 'errs_ms'}), flush=True)
        print('   chord - true onset (ms):', r['errs_ms'])
with open(os.path.join(ABS, 'runs_mini/choir_1102/v11_timing_eval_v1.json'), 'x') as fh:
    json.dump(res, fh, indent=1)
