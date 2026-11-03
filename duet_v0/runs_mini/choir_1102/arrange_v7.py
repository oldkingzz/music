"""v7 of arrange_v6.py: the choir library grows to 100 songs (Vincent 2026-11-02: "先把曲谱扩充到100首"): Twinkle,
v11 and Ode to Joy unchanged from choir_songs_v7.json, plus 97 songs from POP909 (MIT; duet_beat/bench/data/
pop909.json, train + val, single major key, melody + per-beat chords from the dataset). Per POP909 song:
- section: 64 beats (16 bars of 4/4) from the first downbeat at or after its first melody note, every bar regular;
- melody: its notes in the section, onsets and ends on the quarter-beat grid, one note at a time, at least 24 notes;
- chords: the dataset's per-beat scale degree (0..6 -> I ii iii IV V vi V; no chord -> the chord before), merged;
- tempo: the dataset's, kept only between 60 and 140 bpm; parts written an octave up when the melody sits low;
- voicing: arrange_v5.build (same rules as Twinkle / v11 / Ode); kept only when no part is a semitone from the melody
  more than 5 % of the time and its first 8 melody intervals differ from every song already kept.
Songs are taken in dataset order until 97 are kept. Titles are the dataset's ids (POP909 has no titles here).
Writes futures/v4_selfiter/choir_songs_v8.json (new file).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py arrange_v7 2400 .venv/bin/python runs_mini/choir_1102/arrange_v7.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arrange_v5 as A5  # noqa: E402
import arrange_v6 as A6  # noqa: E402

OUT = os.path.join(A5.ABS, 'futures/v4_selfiter/choir_songs_v8.json')
DEG = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'V']
KEYS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
N_POP, K = 97, 64


def q(b):
    return round(b * 4) / 4


def section(s):
    """(melody, chords) of the 64-beat section, both from beat 0, or None."""
    beats, sop = s['beats'], s['sop']
    if s['M'] != 4 or not sop:
        return None
    b0 = next((k for k in range(int(np.floor(sop[0][0])), len(beats)) if beats[k]['pos'] == 0), None)
    if b0 is None or b0 + K > len(beats) or any(beats[b0 + i]['pos'] != i % 4 for i in range(K)):
        return None
    mel = []
    for a, b, m in sop:
        a, b = q(a - b0), q(b - b0)
        if a < 0 or a >= K:
            continue
        b = min(max(b, a + 0.25), K)
        if mel and mel[-1][1] > a:
            mel[-1] = (mel[-1][0], a, mel[-1][2])
            if mel[-1][1] <= mel[-1][0]:
                mel.pop()
        mel.append((a, b, int(m)))
    if len(mel) < 24:
        return None
    spans, prev = [], 'I'
    for i in range(K):
        z = beats[b0 + i]['z']
        c = DEG[z] if 0 <= z < len(DEG) else prev
        prev = c
        if spans and spans[-1][2] == c:
            spans[-1][1] = i + 1.0
        else:
            spans.append([float(i), i + 1.0, c])
    return mel, spans


def opening(mel, n=8):
    d = [m for i, (_, _, m) in enumerate(mel) if i == 0 or m != mel[i - 1][2]]
    return tuple(np.diff(d[:n + 1]).tolist())


if __name__ == '__main__':
    with open(os.path.join(A5.ABS, 'futures/v4_selfiter/choir_songs_v7.json')) as fh:
        songs = json.load(fh)['songs']
    seen = {opening([tuple(n) for n in s['melody']]) for s in songs.values()}
    with open(os.path.join(A5.ABS, 'duet_beat/bench/data/pop909.json')) as fh:
        raw = json.load(fh)
    why = {}
    kept = 0
    for s in sorted(raw['train'] + raw['val'], key=lambda s: s['id']):
        if kept == N_POP:
            break
        if not 60 <= s['tempo_bpm'] <= 140:
            why['tempo'] = why.get('tempo', 0) + 1
            continue
        sec = section(s)
        if sec is None:
            why['section'] = why.get('section', 0) + 1
            continue
        mel, spans = sec
        op = opening(mel)
        if op in seen:
            why['same opening'] = why.get('same opening', 0) + 1
            continue
        med = float(np.median([m for _, _, m in mel]))
        shift = 12 if med < 64 else (-12 if med > 79 else 0)
        sid = s['id'].replace('/', '_')
        try:
            song = A5.build(sid, s['title'], KEYS[s['tonic']], s['tonic'], mel, K, float(s['tempo_bpm']), shift, spans)
        except Exception as e:                            # no voicing found for some window
            why['voicing: ' + type(e).__name__] = why.get('voicing: ' + type(e).__name__, 0) + 1
            continue
        semi = A6.semitone_share(song['melody'], song['parts'], shift)
        if semi > 0.05:
            why['semitone > 5 %'] = why.get('semitone > 5 %', 0) + 1
            continue
        song['semitone_share'] = semi
        songs[sid] = song
        seen.add(op)
        kept += 1
        print(f'{kept:3d} {sid} {song["key"]} {song["bpm"]:.0f} bpm, {len(mel)} notes, shift {shift}, '
              f'chords {len(spans)}, semitone {semi}', flush=True)
    print('skipped:', why, '| library:', len(songs), flush=True)
    assert len(songs) == 100, len(songs)
    with open(OUT, 'x') as fh:
        json.dump({'written': 'run choir_1102, arrange_v7.py (twinkle, v11, ode from arrange_v6; 97 POP909)',
                   'songs': songs}, fh, indent=1)
