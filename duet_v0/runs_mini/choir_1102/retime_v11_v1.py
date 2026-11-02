"""Retime the v11 arrangement to the recording's own timing (run choir_1102, Vincent 2026-11-02: "Twinkle is very good,
the second one is not unpleasant but not on time").

v11's melody in choir_songs_v5.json is the vocadito-11 recording's notes snapped to a quarter-beat grid at 122.45 bpm,
but the singer sings freely (notes stretched or rushed by up to ~0.12 beat each, tempo drifting), so a follower that
predicts a steady tempo is late or early on many notes (his 11:49 run: median 79 ms, p90 174 ms; Twinkle 35 / 79 ms).
He tests v11 with that same recording, so its timing is known: each melody note's onset is moved to where the singer
actually sang it (v11_notes_v1.json, offline pyin, in beats at 122.45 bpm from her first note), and every part and
chord boundary is moved by the same piecewise-linear time warp (snapped onset -> sung onset), so the harmony changes
with her notes. Twinkle is unchanged. Writes futures/v4_selfiter/choir_songs_v6.json (new file).

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/retime_v11_v1.py
"""
import json
import math
import os

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
SRC = os.path.join(ABS, 'futures/v4_selfiter/choir_songs_v5.json')
RAW = os.path.join(ABS, 'runs_mini/choir_1102/v11_notes_v1.json')
OUT = os.path.join(ABS, 'futures/v4_selfiter/choir_songs_v6.json')

songs = json.load(open(SRC))
raw = json.load(open(RAW))['notes']
v11 = songs['songs']['v11']
qs, rs = [], []
for on, off, m in v11['melody']:
    cands = [r for r in raw if abs(r['beat_q'] - on) < 1e-9]
    same = [r for r in cands if r['midi'] % 12 == m % 12]
    pick = (same or cands)
    if not pick:
        raise SystemExit(f'no sung onset for the melody note at beat {on}')
    r = min(pick, key=lambda x: x['beat']) if not same else same[0]
    if on == 0.0:
        r = min(cands, key=lambda x: x['beat'])      # the scoop into her first note is where she starts
    qs.append(on)
    rs.append(r['beat'])
qs, rs = np.array(qs), np.array(rs)
assert np.all(np.diff(rs) > 0), 'sung onsets must keep the order of the notes'


def warp(b):
    """Snapped beat -> sung beat: piecewise linear through the note onsets, slope 1 beyond the ends."""
    if b <= qs[0]:
        return float(rs[0] + (b - qs[0]))
    if b >= qs[-1]:
        return float(rs[-1] + (b - qs[-1]))
    return float(np.interp(b, qs, rs))


new = dict(v11)
new['melody'] = [[round(warp(a), 4), round(warp(b), 4), m] for a, b, m in v11['melody']]
new['parts'] = {p: [[round(warp(a), 4), round(warp(b), 4), m] for a, b, m in seq] for p, seq in v11['parts'].items()}
new['chords'] = [[round(warp(a), 4), round(warp(b), 4), c] for a, b, c in v11['chords']]
new['K'] = int(math.ceil(warp(v11['K'])))
new['timing'] = 'sung onsets of the vocadito-11 recording (v11_notes_v1.json), parts warped to match (retime_v11_v1.py)'
shifts = [round(r - q, 3) for q, r in zip(qs, rs)]
print('onset moves (beats):', shifts)
print('max move', max(abs(x) for x in shifts), 'beats; K', v11['K'], '->', new['K'])
out = {'written': 'run choir_1102, retime_v11_v1.py (from choir_songs_v5.json)', 'songs': dict(songs['songs'], v11=new)}
with open(OUT, 'x') as fh:
    json.dump(out, fh, indent=1)
