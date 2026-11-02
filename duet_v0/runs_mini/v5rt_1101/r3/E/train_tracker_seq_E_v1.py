"""train_tracker_seq_E_v1 (run v5rt_1101, round 3, idea E): the live tracker's ANSWERED-NOTE SEQUENCES on TRAIN Bach
renders, for fitting a tracker-aware hear (the phase-1 profile keeps only the first answered note of each true note;
this keeps every answered note in the order a system's hear() receives them, split fragments included).

TRAIN only: bench3.train_sets()['bach'] (135 chorales), voices 'sung' and 'sung_low' rendered by futures/v5_rt/voice_v1.py
exactly as futures/v5_rt/live_bench_rt.py renders them (wired latency, 90 bpm, count-in 4), fed 10 ms at a time to
live_rt.RTDuet with a silent null system (the tracker, segmenter and onset snapping do not depend on the system).
Offline: live_bench_rt is imported first (its _NoSound tripwire); nothing is played; only Bach chorales are rendered.

Per answered note (the arguments hear() gets, plus the truth it belongs to by live_bench_rt.tracker_rows' matching):
  [song, voice, i, onset (snapped beats, as hear() gets it), onset_raw (beats), heard midi, prev_dur, detect_ms,
   k (true note index or -1 = extra), true midi of k, fragment number within k (0 = first answered), nominal onset of k,
   nominal duration of k, true midi of k-1 (None for the first), tonic]
Per song also the true notes [nominal on, nominal off, midi] (sung_low: the score - 12).

  .venv/bin/python runs_mini/v5rt_1101/r3/E/train_tracker_seq_E_v1.py --out runs_mini/v5rt_1101/models/E_tracker_seq_v1.json
"""
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402  (tripwire first)
import argparse  # noqa: E402
import bisect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

MODELS = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/models'
BUDGET_S = 500.0


class Null:
    def start(self, meta):
        pass

    def plan(self, q, heard):
        return None

    def hear(self, i, onset, midi, prev_dur):
        return None


def one(song, voice, lat, sysn):
    T_b = 60.0 / LBR.BPM
    facts = {'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'bpm': LBR.BPM}
    eng = LBR.make_engine(sysn, 'null', facts, LBR.COUNT_IN)
    t0h = eng.t0 + (lat['Lo'] - lat['Lo_told'])
    y, her, vf = LBR.VV.render(song, voice, t0h, T_b)
    x = np.concatenate([np.zeros(int(round(lat['Li'] * LBR.SR))), y.astype(np.float64), np.zeros(LBR.SR)])
    LBR.run_engine(eng, x, lat)
    starts = {i: c0 for c0, i in eng.by_start.items()}
    ons = [n[0] for n in her]
    seen = {}
    rows = []
    for ev in eng.events:
        i = ev['i']
        tau = starts[i] + lat['Li_told'] - lat['Li']
        k = bisect.bisect_right(ons, tau + LBR.MATCH_PRE) - 1
        if k >= 0 and tau <= her[k][1] + LBR.MATCH_POST:
            frag = seen.get(k, 0)
            seen[k] = frag + 1
            tm, non, ndur = int(her[k][2]), her[k][3], her[k][4] - her[k][3]
            pm = int(her[k - 1][2]) if k > 0 else None
        else:
            k, frag, tm, non, ndur, pm = -1, None, None, None, None, None
        rows.append([song['id'], voice, i, ev['onset_beats'], ev['onset_raw'], int(ev['her_heard']), ev['prev_dur'],
                     ev['detect_ms'], k, tm, frag, non, ndur, pm, int(song['tonic'])])
    true = [[n[3], n[4], int(n[2])] for n in her]
    return rows, true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(MODELS + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit('--out must be a new .json under ' + MODELS)
    c0 = time.perf_counter()
    songs = LBR.bench3.train_sets()['bach']
    lat = LBR.LAT['wired']
    sysn = Null()
    rows, true, done = [], {}, []
    for s in songs:
        if time.perf_counter() - c0 > BUDGET_S:
            break
        for voice in ('sung', 'sung_low'):
            r, t = one(s, voice, lat, sysn)
            rows += r
            true[s['id'] + '|' + voice] = t
        done.append(s['id'])
    out = {'split': 'train', 'label': 'TRAIN Bach renders (voice_v1 sung / sung_low, wired, 90 bpm, count-in 4) through '
                                       'RTDuet with a null system: every answered note in hear() order',
           'fields': ['song', 'voice', 'i', 'onset', 'onset_raw', 'heard', 'prev_dur', 'detect_ms', 'k', 'true_midi',
                      'frag', 'nominal_onset', 'nominal_dur', 'prev_true_midi', 'tonic'],
           'songs': len(done), 'song_ids': done, 'rows': rows, 'true': true,
           'seconds': round(time.perf_counter() - c0, 1),
           'code_sha256': {'live_bench_rt.py': LBR.file_sha(os.path.join(LBR.HERE, 'live_bench_rt.py')),
                           'voice_v1.py': LBR.file_sha(os.path.join(LBR.HERE, 'voice_v1.py'))}}
    with open(path, 'x') as fh:
        json.dump(out, fh)
    n = len(rows)
    fr = sum(1 for r in rows if r[10] not in (None, 0))
    ex = sum(1 for r in rows if r[8] < 0)
    print(f"train songs {len(done)}: answered {n}, split fragments {fr}, extra {ex}, {out['seconds']:.0f} s -> {path}",
          flush=True)


if __name__ == '__main__':
    main()
