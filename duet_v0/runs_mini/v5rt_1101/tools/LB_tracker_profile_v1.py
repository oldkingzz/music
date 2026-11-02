"""The live tracker's real error profile (round 3 phase 1): Bach chorales rendered with futures/v5_rt/voice_v1.py
('sung' and 'sung_low', wired latency, count-in 4, 90 bpm, exactly as futures/v5_rt/live_bench_rt.py renders them) are
fed 10 ms at a time to live_rt.RTDuet with a silent null system (the tracker, segmenter and onset snapping do not depend
on the system), and every true note is matched to the tracker's answered notes with live_bench_rt.tracker_rows.

--split train: the profile for fitting (bench3.train_sets()['bach'], the first --n chorales; at least 60) ->
               runs_mini/v5rt_1101/models/LB_tracker_profile_v1.json
--split val:   the same on the 17 val chorales, REFERENCE ONLY (never for fitting) -> a separate JSON labelled val.

Reported (pooled and per voice): heard - true pitch distribution (semitones) overall, by true note duration (beats and
seconds at 90 bpm), by interval from her previous true note, by register (true MIDI); P(missed), P(split); detect lag
(ms after the true onset, input latency excluded) and snapped-onset error (beats); plus raw per-note rows so a model
can be fitted later. No sound device; only Bach chorales are rendered.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/tools/LB_tracker_profile_v1.py --split train --n 135 \
      --out runs_mini/v5rt_1101/models/LB_tracker_profile_v1.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402  (tripwire first)
import numpy as np  # noqa: E402

BUDGET_S = 480.0          # stop rendering new songs after this (the wrapper kills at 600 s)


class Null:
    """Silent system: the tracker report does not depend on Duet's decisions."""

    def start(self, meta):
        pass

    def plan(self, q, heard):
        return None

    def hear(self, i, onset, midi, prev_dur):
        return None


def dur_bin(d):
    if d <= 0.25:
        return '<=0.25'
    if d <= 0.5:
        return '0.5'
    if d <= 1.0:
        return '0.75-1'
    if d <= 2.0:
        return '1.5-2'
    return '>=3'


def iv_bin(iv):
    if iv is None:
        return 'first'
    a = abs(iv)
    s = 'up' if iv > 0 else 'down'
    if a == 0:
        return 'repeat'
    if a <= 2:
        return f'{s} 1-2'
    if a <= 4:
        return f'{s} 3-4'
    if a <= 7:
        return f'{s} 5-7'
    return f'{s} 8+'


def reg_bin(m):
    lo = (m // 5) * 5
    return f'{lo}-{lo + 4}'


def summarize(rows):
    """rows = tracker_rows rows (+ voice tag at the end). Distribution summary for one group."""
    n = len(rows)
    ans = [r for r in rows if r[4] > 0]
    err = Counter(int(r[5] - r[2]) for r in ans)
    lag = [r[6] for r in ans if r[6] is not None]
    snap = [r[8] for r in ans]
    out = {'n': n, 'p_missed': (n - len(ans)) / n if n else None,
           'p_split': sum(r[4] >= 2 for r in rows) / n if n else None,
           'p_pitch_right': err.get(0, 0) / n if n else None,
           'p_pc_wrong_of_answered': sum(v for d, v in err.items() if d % 12 != 0) / len(ans) if ans else None,
           'p_octave_of_answered': sum(v for d, v in err.items() if d != 0 and d % 12 == 0) / len(ans) if ans else None,
           'heard_minus_true': {str(d): v for d, v in sorted(err.items())},
           'heard_minus_true_share': {str(d): v / len(ans) for d, v in sorted(err.items())} if ans else {},
           'detect_ms_median': float(np.median(lag)) if lag else None,
           'detect_ms_p90': float(np.percentile(lag, 90)) if lag else None,
           'snap_err_exact': sum(abs(s) < 1e-6 for s in snap) / len(snap) if snap else None,
           'snap_err_mean_abs': float(np.mean(np.abs(snap))) if snap else None}
    return out


def by(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def profile(rows, T_b):
    return {'overall': summarize(rows),
            'by_duration_beats': by(rows, lambda r: dur_bin(r[1])),
            'by_duration_s_at_90bpm': by(rows, lambda r: f'{dur_bin(r[1])} beat = {r[1] * T_b:.2f} s'
                                         if r[1] in (0.25, 0.5, 1.0, 2.0) else dur_bin(r[1])),
            'by_interval_from_previous': by(rows, lambda r: iv_bin(r[3])),
            'by_register_true_midi': by(rows, lambda r: reg_bin(r[2])),
            'by_first_vs_rest': by(rows, lambda r: 'first note' if r[3] is None else 'later notes')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=('train', 'val'))
    ap.add_argument('--n', type=int, default=135)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    models = os.path.join(LBR.ABS, 'runs_mini', 'v5rt_1101', 'models')
    if not path.startswith(models + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit('--out must be a new .json under ' + models)
    c0 = time.perf_counter()
    songs = (LBR.bench3.train_sets()['bach'] if a.split == 'train' else LBR.bench3.eval_sets('val')['bach'])[:a.n]
    lat = LBR.LAT['wired']
    T_b = 60.0 / LBR.BPM
    rows = {'sung': [], 'sung_low': []}
    extra = Counter()
    done = Counter()
    sys_null = Null()
    for s in songs:
        if time.perf_counter() - c0 > BUDGET_S:
            break
        for voice in ('sung', 'sung_low'):
            _, res, trows, ex, _ = LBR.rendered_song(sys_null, 'null', s, voice, lat)
            rows[voice] += [r + [s['id']] for r in trows]
            extra[voice] += ex
            done[voice] += 1
    out = {'split': a.split,
           'label': ('TRAIN Bach renders: the tracker profile for fitting' if a.split == 'train'
                     else 'VAL Bach renders: REFERENCE ONLY, never for fitting (systems are tuned on val; no test split)'),
           'songs': int(done['sung']), 'song_ids': [s['id'] for s in songs[:done['sung']]],
           'setup': {'renderer': 'futures/v5_rt/voice_v1.py', 'bench': 'futures/v5_rt/live_bench_rt.py (tracker_rows)',
                     'latency': lat, 'bpm': LBR.BPM, 'count_in': LBR.COUNT_IN, 'engine': 'live_rt.RTDuet, silent null system',
                     'row_fields': ['nominal_onset_beat', 'nominal_dur_beats', 'true_midi', 'interval_from_previous',
                                    'answered_notes', 'heard_midi_first', 'detect_ms', 'onset_err_ms',
                                    'snapped_onset_err_beats', 'song_id'],
                     'matching': 'an answered note belongs to the latest true note with onset <= its start + 80 ms if it '
                                 'starts before that note offset + 30 ms; else extra'},
           'bench3_synthetic': 'bench3 (bench_rt heard): each note independently, p = 0.10, misheard as one of the other '
                               '11 pitch classes uniformly (nearest octave); never missed, never split, onsets exact',
           'profiles': {v: profile(rows[v], T_b) for v in rows},
           'pooled': profile(rows['sung'] + rows['sung_low'], T_b),
           'extra_answered': dict(extra),
           'tracker_summary': {v: LBR.tracker_summary([r[:9] for r in rows[v]], extra[v]) for v in rows},
           'rows': rows, 'seconds': round(time.perf_counter() - c0, 1),
           'code_sha256': {'live_bench_rt.py': LBR.file_sha(os.path.join(LBR.HERE, 'live_bench_rt.py')),
                           'voice_v1.py': LBR.file_sha(os.path.join(LBR.HERE, 'voice_v1.py'))}}
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1)
    for v in ('sung', 'sung_low'):
        o = out['profiles'][v]['overall']
        print(f"{a.split} {v}: songs {done[v]} notes {o['n']} missed {o['p_missed']:.3f} split {o['p_split']:.3f} "
              f"pitch right {o['p_pitch_right']:.3f} pc wrong (of answered) {o['p_pc_wrong_of_answered']:.3f} "
              f"octave {o['p_octave_of_answered']:.3f} detect {o['detect_ms_median']:.0f} ms "
              f"heard-true {o['heard_minus_true']}", flush=True)
    print(f"wrote {path} ({out['seconds']:.0f} s)", flush=True)


if __name__ == '__main__':
    main()
