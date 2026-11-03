"""v8 of arrange_v7.py (v7 kept only 55 POP909 songs: 146 had a part a semitone from the melody more than 5 % of the
time, 66 had no regular 16-bar section from the first melody note, 54 found no voicing). Here each POP909 song tries
every 16-bar section starting on a downbeat 0, 16, 32, ... bars after its first melody note, and keeps its section
with the fewest semitone clashes; songs are then taken by that share (lowest first, at most 8 %), with distinct
openings, until 97 are kept. Everything else as v7. Writes futures/v4_selfiter/choir_songs_v8.json (new file).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py arrange_v8 3600 .venv/bin/python runs_mini/choir_1102/arrange_v8.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arrange_v7 as A7  # noqa: E402

A5, A6 = A7.A5, A7.A6
MAX_SEMI = 0.08


def sections(s):
    """Every regular 64-beat section from a downbeat 0, 16, 32 ... bars after the first melody note."""
    beats = s['beats']
    first = next((k for k in range(int(np.floor(s['sop'][0][0])), len(beats)) if beats[k]['pos'] == 0), None)
    out = []
    for b0 in range(first if first is not None else len(beats), len(beats) - A7.K + 1, 64):
        sub = dict(s, beats=beats[b0:], sop=[[a - b0, b - b0, m] for a, b, m in s['sop'] if a >= b0])
        sec = A7.section(sub)
        if sec is not None:
            out.append((b0, sec))
    return out


if __name__ == '__main__':
    with open(os.path.join(A5.ABS, 'futures/v4_selfiter/choir_songs_v7.json')) as fh:
        songs = json.load(fh)['songs']
    seen = {A7.opening([tuple(n) for n in s['melody']]) for s in songs.values()}
    with open(os.path.join(A5.ABS, 'duet_beat/bench/data/pop909.json')) as fh:
        raw = json.load(fh)
    cands, why = [], {}
    for s in sorted(raw['train'] + raw['val'], key=lambda s: s['id']):
        if s['M'] != 4 or not 60 <= s['tempo_bpm'] <= 140 or not s['sop']:
            why['meter / tempo'] = why.get('meter / tempo', 0) + 1
            continue
        best = None
        for b0, (mel, spans) in sections(s):
            med = float(np.median([m for _, _, m in mel]))
            shift = 12 if med < 64 else (-12 if med > 79 else 0)
            sid = s['id'].replace('/', '_')
            try:
                song = A5.build(sid, s['title'], A7.KEYS[s['tonic']], s['tonic'], mel, A7.K, float(s['tempo_bpm']),
                                shift, spans)
            except Exception:                              # no voicing found for some window
                continue
            semi = A6.semitone_share(song['melody'], song['parts'], shift)
            if best is None or semi < best[0]:
                song.update({'semitone_share': semi, 'section_from_beat': b0})
                best = (semi, song)
        if best is None:
            why['no section / voicing'] = why.get('no section / voicing', 0) + 1
            continue
        cands.append(best)
        print(s['id'], 'best section semitone', best[0], 'from beat', best[1]['section_from_beat'], flush=True)
    kept = 0
    for semi, song in sorted(cands, key=lambda c: (c[0], c[1]['id'])):
        if kept == A7.N_POP or semi > MAX_SEMI:
            break
        op = A7.opening([tuple(n) for n in song['melody']])
        if op in seen:
            why['same opening'] = why.get('same opening', 0) + 1
            continue
        songs[song['id']] = song
        seen.add(op)
        kept += 1
    semis = [s['semitone_share'] for s in songs.values() if 'semitone_share' in s]
    print('skipped:', why, '| candidates', len(cands), '| library', len(songs), '| POP909 semitone share max',
          max(semis), 'median', float(np.median(semis)), flush=True)
    assert len(songs) == 100, len(songs)
    with open(A7.OUT, 'x') as fh:
        json.dump({'written': 'run choir_1102, arrange_v8.py (twinkle, v11, ode from arrange_v6; 97 POP909)',
                   'songs': songs}, fh, indent=1)
