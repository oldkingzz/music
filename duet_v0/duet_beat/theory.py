"""Music constants for duet_beat (spec: claude.ai/artifact/R74zwgC3EBAwQ8bniSh3bZ).

Major keys only (decision Q4) and the seven diatonic triads only (decision Q1). Every pitch class
here is relative to the tonic: 0 = the home note, so the same tables serve every major key.
"""
MAJOR = (0, 2, 4, 5, 7, 9, 11)                      # the 7 scale degrees, semitones above the tonic
CHORDS = ('I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°')
NAMES_C = ('C', 'Dm', 'Em', 'F', 'G', 'Am', 'B°')    # the same chords spelled in C major
N_CHORDS = len(CHORDS)

# Triad on degree i: the scale notes i, i+2, i+4 (two stacked thirds). ROOT[i] is its root.
ROOT = MAJOR
CT = tuple(frozenset({MAJOR[i], MAJOR[(i + 2) % 7], MAJOR[(i + 4) % 7]}) for i in range(7))

PC_NAMES = ('C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B')

ALTO = (53, 77)                                    # F3..F5, the alto range Duet may use (E13)
ALTO_MID = (ALTO[0] + ALTO[1]) / 2

# Two-voice consonances by interval class (M3): unison/octave, 3rds, 5th, 6ths. The 4th (5) is
# a dissonance in two-voice writing, so it is left out on purpose.
CONSONANT = frozenset({0, 3, 4, 7, 8, 9})

# Tie order when two triads explain a beat equally well: the more common functions first.
TIE_ORDER = (0, 4, 3, 5, 1, 2, 6)                  # I, V, IV, vi, ii, iii, vii°

# The repository's rule harmony (follow.plan_harmony): try these intervals below her note.
RULE_STEPS = (3, 4, 8, 9)


def in_scale(pc):
    return pc % 12 in MAJOR


def chord_name(z, tonic=0):
    """Roman numeral plus the chord spelled in the actual key, e.g. 'V (G)' in C."""
    root = (tonic + ROOT[z]) % 12
    suffix = {1: 'm', 2: 'm', 5: 'm', 6: '°'}.get(z, '')
    return f'{CHORDS[z]} ({PC_NAMES[root]}{suffix})'


assert [sorted(c) for c in CT] == [[0, 4, 7], [2, 5, 9], [4, 7, 11], [0, 5, 9], [2, 7, 11], [0, 4, 9], [2, 5, 11]]
