"""A synthetic singer: sings a melody on an 'ah' vowel with glides, vibrato and breaths.

It stands in for a real recording so every note onset is known exactly.
Onset definitions: after a breath, where the attack starts; between legato notes,
the middle of the pitch glide; for a repeated pitch, the bottom of the amplitude dip.
"""
import numpy as np

from dsp import SR, colored_noise, midi_to_hz

# Greensleeves (traditional), A minor, 3/4: (beat from the pickup, length in beats, MIDI note)
GREENSLEEVES = [
    (0, 1, 69),
    (1, 2, 72), (3, 1, 74), (4, 1.5, 76), (5.5, .5, 77), (6, 1, 76), (7, 2, 74), (9, 1, 71),
    (10, 1.5, 67), (11.5, .5, 69), (12, 1, 71),
    (13, 2, 72), (15, 1, 69), (16, 1.5, 69), (17.5, .5, 68), (18, 1, 69), (19, 2, 71), (21, 1, 68),
    (22, 2, 64), (24, 1, 69),
    (25, 2, 72), (27, 1, 74), (28, 1.5, 76), (29.5, .5, 77), (30, 1, 76), (31, 2, 74), (33, 1, 71),
    (34, 1.5, 67), (35.5, .5, 69), (36, 1, 71),
    (37, 1.5, 72), (38.5, .5, 71), (39, 1, 69), (40, 1.5, 68), (41.5, .5, 66), (42, 1, 68), (43, 3, 69),
]
# beat of the note that follows a breath -> breath length in seconds
BREATHS = {13: 0.16, 24: 0.24, 37: 0.14}

FORMANTS_AH = [(730, 90, 1.0), (1090, 110, 0.5), (2440, 170, 0.25)]


def score_notes(melody=GREENSLEEVES, breaths=BREATHS, bpm=84, lead=0.6, jitter=0.010, seed=1):
    """Turn a beat-based melody into (onset, offset, midi) in seconds, with small human timing jitter."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    on = [lead + p * beat + (rng.normal(0, jitter) if i else 0.0) for i, (p, d, m) in enumerate(melody)]
    notes = []
    for i, (p, d, m) in enumerate(melody):
        if i + 1 < len(melody):
            off = on[i + 1] - breaths.get(melody[i + 1][0], 0.0)
        else:
            off = on[i] + d * beat
        notes.append((on[i], off, m))
    return notes


def _formant_gain(f):
    g = np.full_like(f, 0.04)
    for fc, bw, amp in FORMANTS_AH:
        g += amp / np.sqrt(1.0 + ((f - fc) / (bw / 2.0)) ** 2)
    return g


def render(notes, seed=2, tail=0.8, glide=0.06):
    """Render notes [(on, off, midi)] -> (audio, list of dicts with on/off/midi/kind)."""
    rng = np.random.default_rng(seed)
    n = int((notes[-1][1] + tail) * SR)
    t = np.arange(n) / SR
    midi = np.full(n, np.nan)
    amp = np.zeros(n)
    depth = np.zeros(n)
    info = []
    for i, (on, off, m) in enumerate(notes):
        me = m + rng.normal(0.0, 0.10)                      # intonation error, semitones
        i0, i1 = int(on * SR), int(off * SR)
        midi[i0:i1] = me
        amp[i0:i1] = 0.75 + 0.25 * np.sin(np.pi * i / len(notes))
        tt = t[i0:i1] - on
        depth[i0:i1] = 0.30 * np.clip((tt - 0.12) / 0.3, 0.0, 1.0)
        prev_off = notes[i - 1][1] if i else -1.0
        legato = i > 0 and on - prev_off < 0.03
        kind = 'attack' if not legato else ('repeat' if notes[i - 1][2] == m else 'glide')
        info.append({'on': on, 'off': off, 'midi': m, 'kind': kind, 'midi_sung': me})
    # pitch glides and amplitude shaping at note boundaries
    for i, nt in enumerate(info):
        on, off = nt['on'], nt['off']
        if nt['kind'] == 'glide':
            a, b = int((on - glide / 2) * SR), int((on + glide / 2) * SR)
            w = 0.5 - 0.5 * np.cos(np.pi * (np.arange(b - a)) / (b - a))
            midi[a:b] = info[i - 1]['midi_sung'] + (nt['midi_sung'] - info[i - 1]['midi_sung']) * w
            amp[a:b] = np.maximum(amp[a:b], amp[b])
        if nt['kind'] in ('glide', 'repeat'):
            dip = 0.35 if nt['kind'] == 'glide' else 0.8
            sl = slice(int((on - 0.08) * SR), int((on + 0.08) * SR))
            amp[sl] *= 1.0 - dip * np.exp(-((t[sl] - on) / 0.025) ** 2)
        if nt['kind'] == 'attack':
            sl = slice(int(on * SR), int((on + 0.07) * SR))
            amp[sl] *= 0.5 - 0.5 * np.cos(np.pi * (t[sl] - on) / 0.07)
        nxt_on = info[i + 1]['on'] if i + 1 < len(info) else off + 1.0
        if nxt_on - off >= 0.03:
            sl = slice(int((off - 0.08) * SR), int(off * SR))
            amp[sl] *= 0.5 + 0.5 * np.cos(np.pi * (t[sl] - (off - 0.08)) / 0.08)
    # keep the phase running through silences
    valid = ~np.isnan(midi)
    idx = np.where(valid, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    midi_f = midi[idx]
    midi_f[: np.argmax(valid)] = midi[np.argmax(valid)]
    drift = np.cumsum(rng.normal(0, 1, n // 240 + 1))
    drift = np.repeat((drift - np.convolve(drift, np.ones(50) / 50, 'same')) * 0.004, 240)[:n]
    vib_rate = 5.5 + 0.3 * np.sin(2 * np.pi * 0.11 * t)
    vib = np.sin(2 * np.pi * np.cumsum(vib_rate) / SR)
    f0 = midi_to_hz(midi_f + depth * vib + drift)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    y = np.zeros(n)
    for h in range(1, 41):
        fh = h * f0
        keep = np.clip((5000.0 - fh) / 500.0, 0.0, 1.0)
        if not keep.any():
            break
        y += keep * _formant_gain(fh) / h * np.sin(h * phase)
    y *= amp
    y += 0.012 * colored_noise(n, rng, -0.3) * amp          # breathiness
    for i, nt in enumerate(info):                          # audible in-breaths before phrases
        if nt['kind'] == 'attack' and i > 0:
            sl = slice(int((nt['on'] - 0.12) * SR), int((nt['on'] - 0.02) * SR))
            m = sl.stop - sl.start
            y[sl] += 0.01 * colored_noise(m, rng, 0.2) * np.hanning(m)
    y *= 0.7 / np.max(np.abs(y))
    return y.astype(np.float32), info
