"""Are the choir's clashes real (run choir_1102)? The page's 冲突 tag compares ONE Duet voice with the live tracker's
single reading at each heard onset, so tracker glitches and passing notes show up as clashes. Here, frame by frame
(10 ms), the true sung pitch (offline pyin on the mic, minus the input latency, tuning removed) is set against every
choir note that actually sounded (replay through page v14's engine, which matches what the singer heard). A frame is
harsh when any sounding voice is a semitone (or major 7th / minor 9th) from the sung note. Three shares per half of the
song (beats before / after the middle):
  live      the choir as it sounded
  on-beat   the score's A / T / B at the sung beat (true onsets) against the sung pitch: the choir with perfect timing
  score     the score's melody against its own A / T / B (the arrangement alone)
plus 'off-score': frames where the sung pitch class is not the score's melody note at that beat.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py clash_eval_v1 900 .venv/bin/python runs_mini/choir_1102/clash_eval_v1.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402  (tripwire, page v13, truth, replay, NOTES)

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

E = V.E
OUT = os.path.join(E.ABS, 'runs_mini/choir_1102/clash_eval_v1.json')
LO = 0.0421
RUNS = {'203926_r2_mix_v2': 0.174, '195940_r2_mix_v2': 0.174, '132458_r2_mix_v2': 0.1305,
        '124132_r2_mix_v2': 0.1305, '113335_r2_mix_v2': 0.1305, '114908_r2_mix_v2': 0.1305,
        '195914_r2_mix_v2': 0.174}
HARSH = (1, 11)


def at(seq, b):
    """The note of a (start, end, midi) list sounding at beat b, or None."""
    for a, e, m in seq:
        if a <= b < e:
            return int(m)
    return None


def harsh(m, others):
    return any(o is not None and (m - o) % 12 in HARSH for o in others)


if __name__ == '__main__':
    E.P3.MODELS.load()
    lib = E.P13.P12.P5.CHOIR
    res = {}
    for run, rt in RUNS.items():
        E.LI, E.LO = rt - LO, LO
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        V.NOTES.clear()
        eng, _ = E.replay(x, {})
        sid, S = eng.song_id, eng.S or 0
        song = lib[sid]
        sung = E.true_notes(x)
        score = [(float(a), int(m) + S) for a, b, m in song['melody']]
        truth = E.match(score, sung, song['bpm'] if sid == 'v11' else 60.0 / eng.Tb)
        tb = sorted((t, b) for b, t in truth.items())
        sounded = [(nt.i0 / E.SR + eng.Lo, nt.i0 / E.SR + eng.Lo + nt.dur, nt.midi)
                   for _, _, nt in V.NOTES if nt is not None and nt.done > 0 and nt.dur > 0]
        f0, vf, _ = librosa.pyin(x, fmin=70, fmax=800, sr=E.SR, frame_length=2048, hop_length=240)
        t = librosa.times_like(f0, sr=E.SR, hop_length=240) - E.LI
        midi = librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))
        tune = float(np.median((midi[vf] - np.round(midi[vf]) + 0.5) % 1.0 - 0.5))
        mel = [(a, b, m + S) for a, b, m in song['melody']]
        parts = {p: [(a, b, m + S) for a, b, m in v] for p, v in song['parts'].items()}
        end_b = max(b for _, b, _ in mel)
        rows = {h: {'frames': 0, 'live': 0, 'on_beat': 0, 'score': 0, 'off_score': 0} for h in ('first', 'second')}
        bad = []
        for i in np.flatnonzero(vf):
            if t[i] < tb[0][0] or t[i] > tb[-1][0] + 1.0:
                continue
            b = float(np.interp(t[i], [p[0] for p in tb], [p[1] for p in tb], right=np.nan))
            if np.isnan(b):
                b = tb[-1][1] + (t[i] - tb[-1][0]) / eng.Tb
            if b >= end_b:
                continue
            m = int(round(midi[i] - tune))
            r = rows['first' if b < end_b / 2 else 'second']
            live = [c for a, e, c in sounded if a <= t[i] < e]
            ideal = [at(v, b) for v in parts.values()]
            sm = at(mel, b)
            r['frames'] += 1
            r['live'] += harsh(m, live)
            r['on_beat'] += harsh(m, ideal)
            r['score'] += sm is not None and harsh(sm, ideal)
            r['off_score'] += sm is not None and (m - sm) % 12 != 0
            if harsh(m, live):
                bad.append((round(b, 2), m, sorted(live), sm))
        out = {'song': sid, 'tuning_cents': round(tune * 100)}
        for h, r in rows.items():
            n = max(r['frames'], 1)
            out[h] = {k: round(v / n, 3) for k, v in r.items() if k != 'frames'}
        # merge consecutive harsh frames into spans: (first beat, last beat, sung, choir, score melody)
        spans = []
        for b, m, live, sm in bad:
            if spans and b - spans[-1][1] < 0.15 and spans[-1][2] == m and spans[-1][3] == live:
                spans[-1][1] = b
            else:
                spans.append([b, b, m, live, sm])
        out['spans'] = [s for s in spans if s[1] - s[0] >= 0.1]
        res[run] = out
        print(run, sid, 'tuning %+d cents' % out['tuning_cents'], '| first', out['first'], '| second', out['second'],
              flush=True)
        for s in out['spans']:
            print('    harsh beats %5.2f-%5.2f  sung %d  choir %s  score melody %s' % tuple(s), flush=True)
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1)
