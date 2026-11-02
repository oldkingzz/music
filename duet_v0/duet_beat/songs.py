"""Two songs the model never saw, with chords written by hand, plus the 'slow harmony' style setting.

twinkle      Twinkle Twinkle Little Star, C major, 4/4, one note per beat, chords change per half bar.
arpeggios    the user's own tune (10/24 chat): do mi so do mi so | re fa la ... | xi re so ... | do mi so ...
             in 6/8 (two dotted-quarter beats per bar, three notes per beat), I ii V I, played 3 times.

slow_harmony(model) is a hand-set style prior, not learned: chords mostly stay inside a bar and change
on bar lines, as in most pop and folk songs (Bach changes chord almost every beat).
"""
import copy

import numpy as np

from theory import CT

C4 = 60
_TWINKLE = [  # (bar of 4 beats) -> notes as (MIDI, beats)
    [(60, 1), (60, 1), (67, 1), (67, 1)], [(69, 1), (69, 1), (67, 2)],
    [(65, 1), (65, 1), (64, 1), (64, 1)], [(62, 1), (62, 1), (60, 2)],
    [(67, 1), (67, 1), (65, 1), (65, 1)], [(64, 1), (64, 1), (62, 2)],
    [(67, 1), (67, 1), (65, 1), (65, 1)], [(64, 1), (64, 1), (62, 2)],
    [(60, 1), (60, 1), (67, 1), (67, 1)], [(69, 1), (69, 1), (67, 2)],
    [(65, 1), (65, 1), (64, 1), (64, 1)], [(62, 1), (62, 1), (60, 2)],
]
# chord per beat (0=I, 3=IV, 4=V), the usual folk harmonisation
_TWINKLE_Z = [0, 0, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0,
              0, 0, 3, 3, 0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 4, 4,
              0, 0, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0]
# the user's arpeggios: one bar = two beats of three eighth notes (beat = dotted quarter)
_ARP_BARS = [((60, 64, 67), 0), ((62, 65, 69), 1), ((59, 62, 67), 4), ((60, 64, 67), 0)]


def _song(sid, title, sop, z, M, bpm, tonic=0):
    beats = [{'pos': k % M, 'z': zz, 'fit': 1.0, 'onset_pcs': []} for k, zz in enumerate(z)]
    for b in beats:                         # Bach-style onset set = the true chord's tones (for M2b)
        b['onset_pcs'] = sorted(CT[b['z']])
    return {'id': sid, 'title': title, 'tonic': tonic, 'M': M, 'beat_q': 1.0, 'K': len(z),
            'beats': beats, 'sop': sop, 'bpm': bpm}


def twinkle():
    sop, t = [], 0.0
    for bar in _TWINKLE:
        for m, d in bar:
            sop.append([t, t + d, m])
            t += d
    return _song('twinkle', 'Twinkle Twinkle Little Star', sop, _TWINKLE_Z, 4, 100)


def arpeggios(times=3):
    sop, z, t = [], [], 0.0
    for _ in range(times):
        for notes, chord in _ARP_BARS:
            for _beat in range(2):
                for m in notes:
                    sop.append([t, t + 1 / 3, m])
                    t += 1 / 3
                z.append(chord)
    s = _song('arpeggios', 'Your arpeggios: do mi so | re fa la | xi re so | do mi so', sop, z, 2, 72)
    s['grid'] = 1 / 3                       # E17 checks onsets against a triplet grid for this song
    return s


def slow_harmony(model, stay_weak=0.9, stay_half=0.6, stay_bar=0.3):
    """Copy of the model whose transitions favour keeping the chord inside a bar (hand-set)."""
    m = copy.deepcopy(model)
    eye = np.eye(model.A4.shape[1])
    mix = lambda A, s: s * eye + (1 - s) * A  # noqa: E731
    m.A4 = np.stack([mix(model.A4[0], stay_bar), mix(model.A4[1], stay_weak),
                     mix(model.A4[2], stay_half), mix(model.A4[3], stay_weak)])
    m.A2 = np.stack([mix(model.A2[0], stay_bar), mix(model.A2[1], stay_weak)])
    m.style = 'slow'
    return m
