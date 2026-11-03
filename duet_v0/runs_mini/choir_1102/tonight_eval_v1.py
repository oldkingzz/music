"""Tonight's runs (run choir_1102): the first live runs with the 19:59 latency calibration (round trip 174 ms, not the
130.5 ms used all afternoon), against the afternoon's phone runs replayed with that same calibration. Per run: chord
timing against the true onsets (offline pyin on the mic channel, minus the input latency; target -35 ms), the
follower's own prediction errors, opening tempo, one-voice stats, and how well the replay matches the output the
singer actually heard (envelope lag of the run's output channel). Tonight's runs also get the beat-by-beat trace.
Today's engine (choir_v6 defaults), offline (tripwire).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py tonight_eval_v1 900 .venv/bin/python runs_mini/choir_1102/tonight_eval_v1.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402  (tripwire, page v13, truth, replay, NOTES, voice_stats)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

E = V.E
OUT = os.path.join(E.ABS, 'runs_mini/choir_1102/tonight_eval_v1.json')
TONIGHT = ['195914_r2_mix_v2', '195940_r2_mix_v2']
AFTERNOON = {'113335_r2_mix_v2': 'v11', '114908_r2_mix_v2': 'v11', '124132_r2_mix_v2': 'v11',
             '132458_r2_mix_v2': 'v11', '113217_r2_mix_v2': 'twinkle', '114844_r2_mix_v2': 'twinkle'}
LO = 0.0421
LI = 0.1740 - LO                                       # 19:59 calibration
_acc = E.CV6.ChoirStart._accept
TRACE = []


def acc(self, note, c):
    b = self.mel[c][0]
    pred = self.t_of(b)
    _acc(self, note, c)
    TRACE.append((b, note['t'], pred, 60 / self.Tb))


E.CV6.ChoirStart._accept = acc


def env(y, hop=240):
    n = len(y) // hop
    return np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1))


def lag_ms(a, b, max_ms=400, hop=240):
    m = int(max_ms / 1000 * E.SR / hop)
    a, b = (a - a.mean()) / (a.std() + 1e-9), (b - b.mean()) / (b.std() + 1e-9)
    n = min(len(a), len(b))
    k = max(range(-m, m + 1), key=lambda k: float(np.dot(a[max(0, -k):n - max(0, k)], b[max(0, k):n - max(0, -k)])))
    return k * hop / E.SR * 1000


def evaluate(run, sid=None, trace=False):
    y = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0]
    x = y[:, 0]
    sung = E.true_notes(x)
    TRACE.clear()
    V.NOTES.clear()
    eng, _ = E.replay(x, {})
    sid = sid or eng.song_id
    lib = E.P13.P12.P5.CHOIR
    score = [(float(a), int(m) + (eng.S or 0)) for a, b, m in lib[sid]['melody']]
    truth = E.match(score, sung, lib[sid]['bpm'] if sid == 'v11' else 60.0 / eng.Tb)
    log = [(nt.i0, idx, song) for idx, song, nt in V.NOTES if nt is not None]
    r = E.score_run(eng, log, truth)
    pe = np.abs(eng.diag['pred_err_ms'])
    r.update({'pred_med': round(float(np.median(pe))) if len(pe) else None,
              'pred_p90': round(float(np.percentile(pe, 90))) if len(pe) else None, 'shift': eng.S})
    (t0, b0), (t1, b1) = eng.pairs[:2] if len(eng.pairs) >= 2 else ((0, 0), (0, 1))
    if b0 in truth and b1 in truth and t1 > t0:
        r['open_bpm'] = round(60 * (b1 - b0) / (t1 - t0), 1)
        r['open_true_bpm'] = round(60 * (b1 - b0) / (truth[b1] - truth[b0]), 1)
    r.update(V.voice_stats(eng))
    # the replay's own output against the output the singer heard (the run's channel 1)
    sim_root = max((os.path.join(E.ROOT, d) for d in os.listdir(E.ROOT) if os.path.isdir(os.path.join(E.ROOT, d))),
                   key=os.path.getmtime)
    sim = sf.read(os.path.join(sim_root, 'run.wav'), dtype='float32', always_2d=True)[0]
    n = min(len(sim), len(y))
    r['replay_vs_heard_lag_ms'] = lag_ms(env(y[:n, 1]), env(sim[:n, 1]))
    if trace:
        print(f"{'beat':>6} {'true':>7} | {'live-tr':>7} {'pred-tr':>7} {'bpm':>6}")
        for b, t, pred, bpm in TRACE:
            tr = truth.get(b)
            f = lambda v: '-' if tr is None else str(round((v - tr) * 1000))  # noqa: E731
            print(f"{b:6.2f} {'-' if tr is None else round(tr, 3):>7} | {f(t):>7} {f(pred):>7} {bpm:6.1f}")
        print('per chord (beat, ms; target -35):', sorted(set(r['per_chord'])))
    return r


if __name__ == '__main__':
    E.LI, E.LO = LI, LO
    E.P3.MODELS.load()
    res = {}
    keep = ('song', 'shift', 'chords', 'chord_med_abs_from_target', 'chord_p90_abs_from_target', 'chord_median_ms',
            'detect_median_ms', 'pred_med', 'pred_p90', 'open_bpm', 'open_true_bpm', 'two_at_once', 'holes',
            'replay_vs_heard_lag_ms')
    for run in TONIGHT:
        print('########', run, flush=True)
        r = evaluate(run, trace=True)
        res[run] = r
        print('   ', json.dumps({k: r.get(k) for k in keep}), flush=True)
    for run, sid in AFTERNOON.items():
        r = evaluate(run, sid)
        res[run] = r
        print(run, json.dumps({k: r.get(k) for k in keep}), flush=True)
        print('    per chord:', sorted(set(r['per_chord'])), flush=True)
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1)
