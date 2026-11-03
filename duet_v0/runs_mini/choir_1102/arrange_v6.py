"""v6 of arrange_v5.py: a third song for the choir rehearsal mode, Ode to Joy (Beethoven, Symphony No. 9; public
domain), 16 bars in C major at 96 bpm (64 beats, about 40 s: longer than v11's 13 s, Vincent 2026-11-02: "这个音频还是
有点短 ... 然后我们再拉入第三首歌"). Melody as usually sung (bars 4, 8 and 16 dotted, bars 10 and 11 with the two
eighths, bar 12 down to G3), written an octave up for the parts (s_shift 12, like v11). Chords by hand, one per bar
or half bar, no chord shorter than 1.5 beats; every passing F on V (no part a semitone from her). Twinkle and v11 are
copied unchanged from choir_songs_v5.json. Writes futures/v4_selfiter/choir_songs_v7.json (new file).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py arrange_v6 300 .venv/bin/python runs_mini/choir_1102/arrange_v6.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arrange_v5 as A5  # noqa: E402

OUT = os.path.join(A5.ABS, 'futures/v4_selfiter/choir_songs_v7.json')
C4, D4, E4, F4, G4, G3 = 60, 62, 64, 65, 67, 55
BARS = [  # (midi, beats) per bar
    [(E4, 1), (E4, 1), (F4, 1), (G4, 1)], [(G4, 1), (F4, 1), (E4, 1), (D4, 1)],
    [(C4, 1), (C4, 1), (D4, 1), (E4, 1)], [(E4, 1.5), (D4, 0.5), (D4, 2)],
    [(E4, 1), (E4, 1), (F4, 1), (G4, 1)], [(G4, 1), (F4, 1), (E4, 1), (D4, 1)],
    [(C4, 1), (C4, 1), (D4, 1), (E4, 1)], [(D4, 1.5), (C4, 0.5), (C4, 2)],
    [(D4, 1), (D4, 1), (E4, 1), (C4, 1)], [(D4, 1), (E4, 0.5), (F4, 0.5), (E4, 1), (C4, 1)],
    [(D4, 1), (E4, 0.5), (F4, 0.5), (E4, 1), (D4, 1)], [(C4, 1), (D4, 1), (G3, 2)],
    [(E4, 1), (E4, 1), (F4, 1), (G4, 1)], [(G4, 1), (F4, 1), (E4, 1), (D4, 1)],
    [(C4, 1), (C4, 1), (D4, 1), (E4, 1)], [(D4, 1.5), (C4, 0.5), (C4, 2)]]
# (tried, off: one chord per bar with V7 over the passing F: harsh share 0.56, the F of bars 1 / 5 / 13 a semitone
# from the I chord's E.) Now every passing F sits on V (G B D, no F), so no part is a semitone from the melody.
CHORDS = [[0.0, 2.0, 'I'], [2.0, 8.0, 'V'], [8.0, 13.5, 'I'], [13.5, 16.0, 'V'],
          [16.0, 18.0, 'I'], [18.0, 24.0, 'V'], [24.0, 28.0, 'I'], [28.0, 29.5, 'V'], [29.5, 32.0, 'I'],
          [32.0, 34.0, 'V'], [34.0, 36.0, 'I'], [36.0, 38.0, 'V'], [38.0, 40.0, 'I'], [40.0, 44.0, 'V'],
          [44.0, 46.0, 'I'], [46.0, 48.0, 'V'],
          [48.0, 50.0, 'I'], [50.0, 56.0, 'V'], [56.0, 60.0, 'I'], [60.0, 61.5, 'V'], [61.5, 64.0, 'I']]


def semitone_share(mel, parts, shift):
    """Share of melody time where some part is a semitone (or major 7th) from her: the clashes that sound sour."""
    bad, tot = 0.0, 0.0
    for on, off, m in mel:
        tot += off - on
        for seq in parts.values():
            for a, b, p in seq:
                ov = A5.overlap(on, off, a, b)
                if ov > 0 and abs(p - (m + shift)) % 12 in (1, 11):
                    bad += ov
    return round(bad / tot, 3)


def melody():
    mel, t = [], 0.0
    for bar in BARS:
        for m, d in bar:
            mel.append((t, t + d, m))
            t += d
    assert t == 64.0
    return mel


if __name__ == '__main__':
    with open(os.path.join(A5.ABS, 'futures/v4_selfiter/choir_songs_v5.json')) as fh:
        songs = json.load(fh)['songs']
    songs['ode'] = A5.build('ode', 'Ode to Joy (Beethoven)', 'C', 0, melody(), 64, 96.0, 12, CHORDS)
    s = songs['ode']
    print('ode chords', [(a, c) for a, b, c in s['chords']])
    for v in 'ATB':
        print('  ', v, [(a, p) for a, b, p in s['parts'][v]])
    print('  harsh share vs melody', s['harsh_share_vs_melody'], 'voicing cost', s['voicing_cost'],
          '| twinkle', songs['twinkle']['harsh_share_vs_melody'], 'v11', songs['v11']['harsh_share_vs_melody'])
    print('  semitone share', {k: semitone_share(v['melody'], v['parts'], v['melody_s_shift'])
                               for k, v in songs.items()})
    with open(OUT, 'x') as fh:
        json.dump({'written': 'run choir_1102, arrange_v6.py (twinkle, v11 from arrange_v5)', 'songs': songs}, fh,
                  indent=1)
