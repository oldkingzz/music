"""v9 of arrange_v6.py: the 10-song demo library (Vincent 2026-11-02: "扩充到10首曲子 ... 这个就是明天我拿过去的demo").
Twinkle, v11 and Ode to Joy are copied from choir_songs_v7.json; seven public-domain tunes most people know are added,
all 4/4 and written in C as usually sung: Frere Jacques (两只老虎), Mary Had a Little Lamb, Jingle Bells (chorus),
When the Saints Go Marching In, Old MacDonald, Auld Lang Syne (友谊地久天长) and Going Home (Dvorak Largo, 念故乡).
Chords come from arrange_v5's chord DP (2-beat windows) unless written by hand below; each song is checked with
arrange_v6.semitone_share (share of melody time some part is a semitone from her). Parts sung an octave up when the
melody's median is below E4 (arrange_v8's rule) or it goes below C4 (Frere Jacques' low G: no voicing under it). Writes futures/v4_selfiter/choir_songs_v9.json (new file).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py arrange_v9 300 .venv/bin/python runs_mini/choir_1102/arrange_v9.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arrange_v6 as A6  # noqa: E402

A5 = A6.A5
OUT = os.path.join(A5.ABS, 'futures/v4_selfiter/choir_songs_v9.json')
G3, A3, B3, C4, D4, E4, F4, G4, A4 = 55, 57, 59, 60, 62, 64, 65, 67, 69

# id: (title, bpm, [(midi, beats), ...], chords or None for the DP)
SONGS = {
    'frere': ('Frere Jacques (两只老虎)', 100.0,
              [(C4, 1), (D4, 1), (E4, 1), (C4, 1)] * 2 + [(E4, 1), (F4, 1), (G4, 2)] * 2
              + [(G4, .5), (A4, .5), (G4, .5), (F4, .5), (E4, 1), (C4, 1)] * 2 + [(C4, 1), (G3, 1), (C4, 2)] * 2,
              # (I under the whole round would put the F of bars 3-4 on E: those two beats go to V)
              # and the G A G F run of bars 5-6 goes to V too)
              # (tried, off: 1-beat V spans under each F, no voicing found)
              [[0, 8, 'I'], [8, 10, 'V'], [10, 12, 'I'], [12, 14, 'V'], [14, 16, 'I'],
               [16, 18, 'V'], [18, 20, 'I'], [20, 22, 'V'], [22, 24, 'I'], [24, 32, 'I']]),
    'mary': ('Mary Had a Little Lamb', 100.0,
             [(E4, 1), (D4, 1), (C4, 1), (D4, 1), (E4, 1), (E4, 1), (E4, 2), (D4, 1), (D4, 1), (D4, 2),
              (E4, 1), (G4, 1), (G4, 2), (E4, 1), (D4, 1), (C4, 1), (D4, 1), (E4, 1), (E4, 1), (E4, 1), (E4, 1),
              (D4, 1), (D4, 1), (E4, 1), (D4, 1), (C4, 4)], None),
    'jingle': ('Jingle Bells (chorus)', 112.0,
               [(E4, 1), (E4, 1), (E4, 2), (E4, 1), (E4, 1), (E4, 2), (E4, 1), (G4, 1), (C4, 1.5), (D4, .5),
                (E4, 4), (F4, 1), (F4, 1), (F4, 1.5), (F4, .5), (F4, 1), (E4, 1), (E4, 1), (E4, .5), (E4, .5),
                (E4, 1), (D4, 1), (D4, 1), (E4, 1), (D4, 2), (G4, 2),
                (E4, 1), (E4, 1), (E4, 2), (E4, 1), (E4, 1), (E4, 2), (E4, 1), (G4, 1), (C4, 1.5), (D4, .5),
                (E4, 4), (F4, 1), (F4, 1), (F4, 1.5), (F4, .5), (F4, 1), (E4, 1), (E4, 1), (E4, .5), (E4, .5),
                (G4, 1), (G4, 1), (F4, 1), (D4, 1), (C4, 4)],
               # (tried, off: the DP's chords, iii and vi under "jingle bells": harsh 0.31, not the tune people know)
               [[0, 16, 'I'], [16, 20, 'IV'], [20, 24, 'I'], [24, 32, 'V'], [32, 48, 'I'], [48, 52, 'IV'],
                [52, 56, 'I'], [56, 60, 'V7'], [60, 64, 'I']]),
    'saints': ('When the Saints Go Marching In', 112.0,
               [(C4, 1), (E4, 1), (F4, 1), (G4, 5)] * 2
               + [(C4, 1), (E4, 1), (F4, 1), (G4, 2), (E4, 2), (C4, 2), (E4, 2), (D4, 5),
                  (E4, 1), (E4, 1), (D4, 1), (C4, 3), (C4, 1), (E4, 2), (G4, 2), (G4, 1), (F4, 4),
                  (E4, 1), (F4, 1), (G4, 2), (E4, 2), (C4, 2), (D4, 2), (C4, 6)],
               # (tried, off: the DP's chords, V7 every other bar under the held G)
               [[0, 26, 'I'], [26, 32, 'V'], [32, 44, 'I'], [44, 48, 'IV'], [48, 56, 'I'], [56, 58, 'V'],
                [58, 64, 'I']]),
    'macdonald': ('Old MacDonald Had a Farm', 104.0,
                  [(C4, 1), (C4, 1), (C4, 1), (G3, 1), (A3, 1), (A3, 1), (G3, 2), (E4, 1), (E4, 1), (D4, 1),
                   (D4, 1), (C4, 3), (G3, 1)]
                  + [(C4, 1), (C4, 1), (C4, 1), (G3, 1), (A3, 1), (A3, 1), (G3, 2), (E4, 1), (E4, 1), (D4, 1),
                     (D4, 1), (C4, 4)], None),
    'auld': ('Auld Lang Syne (友谊地久天长)', 80.0,
             [(G3, 1), (C4, 1.5), (C4, .5), (C4, 1), (E4, 1), (D4, 1.5), (C4, .5), (D4, 1), (E4, 1),
              (C4, 1.5), (C4, .5), (E4, 1), (G4, 1), (A4, 3), (A4, 1), (G4, 1.5), (E4, .5), (E4, 1), (C4, 1),
              (D4, 1.5), (C4, .5), (D4, 1), (E4, 1), (C4, 1.5), (A3, .5), (A3, 1), (G3, 1), (C4, 3)], None),
    'goinghome': ('Going Home (Dvorak, 念故乡)', 66.0,
                  [(E4, 1), (G4, 1), (G4, 2), (E4, 1), (D4, 1), (C4, 2), (D4, 1), (E4, 1), (G4, 1), (E4, 1),
                   (D4, 4), (E4, 1), (G4, 1), (G4, 2), (E4, 1), (D4, 1), (C4, 2), (D4, 1), (E4, 1), (D4, 1.5),
                   (C4, .5), (C4, 4)], None),
}


def melody(notes):
    mel, t = [], 0.0
    for m, d in notes:
        mel.append((t, t + d, m))
        t += d
    assert t % 4 == 0, t
    return mel, int(t)


if __name__ == '__main__':
    with open(os.path.join(A5.ABS, 'futures/v4_selfiter/choir_songs_v7.json')) as fh:
        songs = json.load(fh)['songs']
    for sid, (title, bpm, notes, chords) in SONGS.items():
        mel, K = melody(notes)
        shift = 12 if np.median([m for _, _, m in mel]) < 64 or min(m for _, _, m in mel) < C4 else 0
        chords = [[float(a), float(b), c] for a, b, c in chords] if chords else None
        songs[sid] = s = A5.build(sid, title, 'C', 0, mel, K, bpm, shift, chords)
        print(sid, K, 'beats', round(K * 60 / bpm, 1), 's | chords', [(a, c) for a, b, c in s['chords']])
    for sid, s in songs.items():
        o = [m for _, _, m in s['melody']]
        first = [o[0]] + [m for p, m in zip(o, o[1:]) if m != p]
        print(f"{sid:10s} opening {first[1] - first[0]:+d} | harsh {s['harsh_share_vs_melody']} | semitone "
              f"{A6.semitone_share(s['melody'], s['parts'], s['melody_s_shift'])} | cost {s['voicing_cost']}")
    assert len(songs) == 10
    with open(OUT, 'x') as fh:
        json.dump({'written': 'run choir_1102, arrange_v9.py (twinkle, v11, ode from arrange_v6; 7 public-domain '
                              'tunes)', 'songs': songs}, fh, indent=1, ensure_ascii=False)
