"""Duet v0: simulate live following on a recording and score it.

    python3 run_v0.py                 # synthetic singer (Greensleeves), all scenarios
    python3 run_v0.py --clip a.wav    # a 24 kHz mono recording instead

Writes outputs/<source>/<scenario>/{solo,duet_pred,duet_react}.wav and metrics.json,
plus outputs/<source>/summary.json. Nothing outside outputs/ is written.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import perform
import singer
from dsp import HOP, SR, read_wav, render_voice, write_wav
from follow import (Accompanist, FeatureTracker, Follower, analyse, offline_notes, plan_harmony,
                    profile_from_anchors)

HERE = Path(__file__).parent
CONSONANT = {0, 3, 4, 7, 8, 9}

# 'natural' = ordinary take-to-take variation against her rehearsal recording.
# 'expressive' = she sings it very differently from the rehearsal (hard case).
NATURAL = dict(sway=0.03, period=8.0, rit=0.10, phase=2.0)
EXPRESSIVE = dict(sway=0.07, period=8.0, rit=0.30, phase=0.7)
SCENARIOS = {
    'steady':      dict(sway=0.01, rit=0.0),
    'natural':     NATURAL,
    'expressive':  EXPRESSIVE,
    'expressive_rehearsed': dict(EXPRESSIVE, rehearsal=dict(sway=0.065, period=8.0, rit=0.26, phase=1.1, seed=9)),
    'pause':       dict(NATURAL, pause=(1, 1.5)),
    'skip':        dict(NATURAL, jump=(0, 1)),
    'noisy':       dict(NATURAL, room_db=-25.0, noise=[(13.0, 16.0, 0.0)]),
}


def find_gaps(F, min_len=0.12):
    """Silent stretches in the reference (breaths), from the voicing track."""
    gaps, t, v = [], F['t'], F['voiced']
    i = 0
    while i < len(v):
        if not v[i]:
            s = i
            while i < len(v) and not v[i]:
                i += 1
            if s > 0 and i < len(v) and t[i - 1] - t[s] >= min_len:
                gaps.append((float(t[s]), float(t[i - 1])))
        else:
            i += 1
    return gaps


def match(times, truth, tol=0.15):
    """For each time, the nearest truth time within tol (or None)."""
    truth = np.asarray(truth)
    out = []
    for x in times:
        k = int(np.argmin(np.abs(truth - x)))
        out.append(k if abs(truth[k] - x) <= tol else None)
    return out


def calibrate(x, true_notes, F, det, breaths):
    """Measure how late our detectors report onsets, per onset kind, on the synthetic singer.

    ref:  rehearsal-time note segmentation vs the true onsets.
    live: the whole live pipeline (follower + score-informed onsets) run on the same audio.
    """
    true_on = [n['on'] for n in true_notes]
    idx = match([n['on'] for n in det], true_on)
    ref_bias = {}
    for kind in ('attack', 'glide', 'repeat'):
        d = [nt['on'] - true_on[k] for nt, k in zip(det, idx) if k is not None and true_notes[k]['kind'] == kind]
        ref_bias[kind] = float(np.median(d)) if d else 0.0
    cal = [dict(nt, on=nt['on'] - ref_bias.get(nt['kind'], 0.0)) for nt in det]
    acc = Accompanist(cal, [], 0.0, breaths=breaths)
    _follow(x, F, acc)
    by_r = {nt['on']: k for k, nt in enumerate(cal)}
    live_bias = {}
    for kind in ('attack', 'glide', 'repeat'):
        d = [t - true_on[idx[by_r[r]]] for t, r in acc.anchors
             if idx[by_r[r]] is not None and cal[by_r[r]]['kind'] == kind]
        live_bias[kind] = float(np.median(d)) if d else 0.0
    found = sum(1 for k in match(true_on, [n['on'] for n in det]) if k is not None)
    return ref_bias, live_bias, {'true_notes': len(true_on), 'detected': len(det), 'matched': found,
                                 'anchored_on_reference': len(acc.anchors)}


def _follow(y, F, acc):
    ft, fol = FeatureTracker(), Follower(F)
    for k in range(len(y) // HOP):
        ft.step(y[k * HOP:(k + 1) * HOP])
        fol.step(ft.m[-1], ft.v[-1], ft.nov[-1])
        acc.step((k + 1) * HOP / SR, ft, fol, 0.0)


def run_scenario(name, spec, x, F, notes, plan, key, join_r, gaps, truth_on, out_dir, bias_live,
                 out_latency=0.005):
    spec = dict(spec)
    reh = spec.pop('rehearsal', None)
    profile = None
    if reh:                               # a rehearsal take: follow it, learn her timing habits
        ra = Accompanist(notes, [], 0.0, bias_live=bias_live, breaths=[(a + b) / 2 for a, b in gaps])
        _follow(perform.make(x, gaps, **reh).audio, F, ra)
        profile = profile_from_anchors(ra.anchors, len(x) / SR)
    perf = perform.make(x, gaps, **spec)
    y = perf.audio
    ft = FeatureTracker()
    fol = Follower(F)
    breaths = [(a + b) / 2 for a, b in gaps]
    acc = Accompanist(notes, plan, join_r, out_latency=out_latency, bias_live=bias_live, profile=profile,
                      breaths=breaths)
    step_ms, track_err = [], []
    for k in range(len(y) // HOP):
        now = (k + 1) * HOP / SR
        t0 = time.perf_counter()
        ft.step(y[k * HOP:(k + 1) * HOP])
        fol.step(ft.m[-1], ft.v[-1], ft.nov[-1])
        compute = time.perf_counter() - t0
        acc.step(now, ft, fol, compute)
        step_ms.append((time.perf_counter() - t0) * 1000)
        r_true = perf.live_to_ref(ft.t(ft.i))[0]
        if ft.v[-1] and not np.isnan(r_true):
            track_err.append((ft.t(ft.i), fol.ref_time() - r_true))

    # ground truth: when does each planned note's true onset happen in this performance?
    res = {}
    for which, src in (('pred', acc.pred), ('react', acc.react)):
        errs, status = [], {'played': 0, 'held': 0, 'missed': 0, 'not_decided': 0, 'wrong': 0}
        for j, p in enumerate(acc.plan):
            t_true = perf.ref_to_live(truth_on[p['note']]) if truth_on[p['note']] is not None else float('nan')
            v = src.get(j)
            if v is None:
                status['not_decided'] += 1
                continue
            status[v['status']] += 1
            if v['status'] == 'played':
                if np.isnan(t_true):
                    status['wrong'] += 1          # played a note she skipped
                else:
                    errs.append(v['at'] - t_true)
        a = np.abs(errs) if errs else np.array([np.nan])
        res[which] = {
            'n': len(errs), 'median_abs_ms': float(np.median(a) * 1000), 'p90_abs_ms': float(np.percentile(a, 90) * 1000),
            'mean_signed_ms': float(np.mean(errs) * 1000) if errs else None,
            'within30': float(np.mean(a <= 0.030)), 'within60': float(np.mean(a <= 0.060)), **status,
            'errors_ms': [round(e * 1000, 1) for e in errs],
            'false_entries': len(acc.false_entries) if which == 'pred' else 0,
        }
    te = np.array([e for _, e in track_err])
    lost_frames = np.abs(te) > 0.25
    # recovery: after an inserted pause or jump, how long until tracking is within 100 ms for 0.5 s
    recov = None
    seg_starts = [s[1][0] for s in perf.segments if s[0] == 'audio'][1:]
    if seg_starts:
        t_resume = seg_starts[0]
        tt = np.array([t for t, _ in track_err])
        good = (np.abs(te) < 0.1) & (tt >= t_resume)
        run = 0
        for t, g in zip(tt[tt >= t_resume], good[tt >= t_resume]):
            run = run + 1 if g else 0
            if run >= 50:
                recov = float(t - 0.5 - t_resume)
                break
    music_recov = None
    if seg_starts:                      # first machine note that lands within 60 ms after the break
        good = []
        for j, v in acc.pred.items():
            t_true = perf.ref_to_live(truth_on[acc.plan[j]['note']])
            if v['status'] == 'played' and not np.isnan(t_true) and v['at'] >= seg_starts[0] and abs(v['at'] - t_true) <= 0.06:
                good.append(v['at'])
        music_recov = float(min(good) - seg_starts[0]) if good else None
    cons = [((p['midi'] - p['h']) % 12) in CONSONANT for p in acc.plan]
    step_ms = np.array(step_ms)
    metrics = {
        'scenario': name, 'spec': {k: v for k, v in spec.items()}, 'key': key,
        'length_s': round(perf.length, 2), 'resets': fol.resets,
        'tracking_within100ms': float(np.mean(np.abs(te) < 0.1)), 'tracking_lost_frac': float(np.mean(lost_frames)),
        'recovery_s': recov, 'music_recovery_s': music_recov, 'harmony_consonant': float(np.mean(cons)) if cons else None,
        'step_ms_median': float(np.median(step_ms)), 'step_ms_p99': float(np.percentile(step_ms, 99)),
        'step_ms_max': float(step_ms.max()), 'pred': res['pred'], 'react': res['react'],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wav(out_dir / 'solo.wav', y)
    for which in ('pred', 'react'):
        notes = [(s, d, m) for s, d, m, _ in acc.machine_notes(which)]
        if which == 'pred':
            notes += [(at, 0.06, acc.plan[j]['h']) for at, j in acc.false_entries]
        mv = render_voice(notes, len(y))
        stereo = np.stack([0.85 * y + 0.35 * mv, 0.55 * y + 0.85 * mv], axis=1)
        stereo *= 0.9 / max(1e-6, np.abs(stereo).max())
        write_wav(out_dir / f'duet_{which}.wav', stereo)
    trace = {'track': [[round(t, 3), round(e * 1000, 1)] for t, e in track_err[::5]],
             'log': [[round(r[0], 3), round(r[1], 3), round(r[2], 3), round(r[4], 3), int(r[5])] for r in acc.log[::5]],
             'pred_notes': [[round(s, 3), round(d, 3), int(m), j] for s, d, m, j in acc.machine_notes('pred')],
             'react_notes': [[round(s, 3), round(d, 3), int(m), j] for s, d, m, j in acc.machine_notes('react')],
             'truth': [perf.ref_to_live(truth_on[p['note']]) if truth_on[p['note']] is not None else None for p in acc.plan]}
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    (out_dir / 'trace.json').write_text(json.dumps(trace))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', help='24 kHz mono wav to use instead of the synthetic singer')
    ap.add_argument('--only', help='comma-separated scenario names')
    ap.add_argument('--latency', type=float, default=0.005, help='output latency in seconds')
    args = ap.parse_args()

    # 1. calibrate detector delays on the synthetic singer (exact onsets known)
    s_notes = singer.score_notes()
    xs, true_s = singer.render(s_notes)
    Fs = analyse(xs)
    det_s = offline_notes(Fs)
    ref_bias, live_bias, det_stats = calibrate(xs, true_s, Fs, det_s, [(a + b) / 2 for a, b in find_gaps(Fs)])
    print('calibration (s): reference', ref_bias, 'live', live_bias, det_stats)

    if args.clip:
        x = read_wav(args.clip)
        src = Path(args.clip).stem
        F = analyse(x)
        det = offline_notes(F)
        truth_on = None
    else:
        x, F, det, src = xs, Fs, det_s, 'synthetic'
        true_on = [n['on'] for n in true_s]
        truth_on = [true_on[k] if k is not None else None for k in match([n['on'] for n in det], true_on)]
    for nt in det:                       # remove the measured detector delay
        nt['on'] -= ref_bias.get(nt['kind'], 0.0)
    if truth_on is None:
        truth_on = [nt['on'] for nt in det]
    plan, key = plan_harmony(det)
    gaps = find_gaps(F)
    dur = len(x) / SR
    join_r = next((nt['on'] for nt in det if nt['on'] >= 0.25 * dur and nt['kind'] == 'attack'),
                  next(nt['on'] for nt in det if nt['on'] >= 0.25 * dur))
    print(f'{src}: {dur:.1f} s, {len(det)} notes, key {key}, {len(gaps)} breaths at',
          [round((a + b) / 2, 2) for a, b in gaps], f'machine joins at {join_r:.2f} s')

    names = args.only.split(',') if args.only else list(SCENARIOS)
    base = HERE / 'outputs' / src
    summary = {'source': src, 'key': key, 'join_r': join_r, 'calibration': {'ref': ref_bias, 'live': live_bias},
               'detection': det_stats, 'scenarios': {}}
    for name in names:
        spec = dict(SCENARIOS[name])
        if ('pause' in spec and len(gaps) < 2) or ('jump' in spec and len(gaps) < 2):
            print(f'skip {name}: needs at least 2 breaths')
            continue
        m = run_scenario(name, spec, x, F, det, plan, key, join_r, gaps, truth_on, base / name, live_bias,
                         out_latency=args.latency)
        summary['scenarios'][name] = {k: v for k, v in m.items() if k not in ('spec',)}
        p, r = m['pred'], m['react']
        print(f"{name:8s} track<100ms {m['tracking_within100ms']:.0%} resets {m['resets']} "
              f"recov {m['recovery_s'] and round(m['recovery_s'],2)}/{m['music_recovery_s'] and round(m['music_recovery_s'],2)} | PRED med {p['median_abs_ms']:.0f} p90 {p['p90_abs_ms']:.0f} "
              f"<=30 {p['within30']:.0%} <=60 {p['within60']:.0%} played {p['played']} held {p['held']} "
              f"missed {p['missed']} wrong {p['wrong']} false {p['false_entries']} | REACT med {r['median_abs_ms']:.0f} "
              f"| cons {m['harmony_consonant']:.0%} | step {m['step_ms_median']:.2f}/{m['step_ms_p99']:.2f} ms")
    (base / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
