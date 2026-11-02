"""voice_v1: her voice for the live benchmark futures/v5_rt/live_bench_rt.py (run v5rt_1101, round 3 phase 1; frozen
with that file after phase 1, sha256 of both in runs_mini/v5rt_1101/LIVE_BENCH_SHA.txt). Licences: only Bach chorales
(public domain) and Twinkle may ever be rendered; never pop / pop909 / folk songs. Nothing here plays sound: it returns
arrays (and the bench feeds them to an engine offline).

Her timing (the same for every voice): she follows the click; her note i starts at t0 + onset_i * T_b + jitter_i with
jitter_i ~ N(0, 15 ms) (numpy default_rng([crc32(song id), 1])). A note that ends where the next one starts (legato in
the score) ends at that next note's jittered onset; any other note (before a rest, the last note) ends at its nominal
offset t0 + off_i * T_b. A note never lasts less than 50 ms. These are her TRUE notes (seconds, MIDI as sung).

Voices (render(song, voice, t0, T_b) -> audio from acoustic time 0 at SR, her true notes, facts):
  clean     audio.render_clean on her true notes with audio.render_her's 30 ms gap at the end of each note: no vibrato,
            no glides, no drift (30 ms attack and 40 ms release ramps). A tracker sanity check.
  sung      singer.render (its code copied below, formulas unchanged: 'ah' vowel, vibrato growing from 0.12 s, 60 ms
            glides at legato boundaries, amplitude dips at legato boundaries, drift, its own N(0, 0.10 st) intonation per
            note, breathiness, in-breaths before attacks; its random draws come from default_rng([crc32(id), 2]) in
            singer.render's order) plus
            - onset scoops: each non-legato note (singer.render's kind 'attack': the first note, or a note that starts
              >= 30 ms after the previous one ended) gets a scoop with probability 1/2: it starts d ~ U(0.5, 1.5)
              semitones flat and reaches its pitch after s ~ U(0.06, 0.12) s (raised-cosine approach);
            - per-note detune N(0, 0.15 st), added to singer.render's own intonation error;
            (scoop and detune draws: default_rng([crc32(id), 3]), four draws per note in note order)
            - room noise: dsp.colored_noise(slope -0.5) over the whole recording (default_rng([crc32(id), 4])), its RMS
              40 dB below the RMS of her voice during her notes.
  sung_low  'sung' one octave lower: every note - 12 with the same random draws, so only the octave differs.
"""
import os
import sys
import zlib

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
for _p in (ABS, os.path.join(ABS, 'duet_beat')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import singer  # noqa: E402  (repo root: _formant_gain)
from audio import render_clean  # noqa: E402  (duet_beat/audio.py)
from dsp import SR, colored_noise, midi_to_hz  # noqa: E402

VOICES = ('clean', 'sung', 'sung_low')
JITTER = 0.015          # s, sd of her onset timing around the click grid
MIN_NOTE = 0.05         # s
CLEAN_GAP = 0.03        # s, audio.render_her('clean')'s gap at each note's end
SCOOP_P = 0.5
SCOOP_CENTS = (50.0, 150.0)
SCOOP_S = (0.06, 0.12)
DETUNE_ST = 0.15
NOISE_DB = -40.0
ALLOWED = ('bach/', 'twinkle')   # song ids that may be rendered (licences)


def seed(song, k):
    return np.random.default_rng([zlib.crc32(str(song['id']).encode()), k])


def check_allowed(song):
    sid = str(song.get('id', ''))
    if not (sid.startswith('bach/') or sid == 'twinkle'):
        raise SystemExit(f'voice_v1 renders only Bach chorales and Twinkle, not {sid!r}')


def true_notes(song, t0, T_b, shift=0):
    """Her TRUE notes [(on_s, off_s, midi + shift, nominal_on_beats, nominal_off_beats)] sorted by onset."""
    sop = sorted(song['sop'], key=lambda n: n[0])
    rng = seed(song, 1)
    jit = rng.normal(0.0, JITTER, len(sop))
    ons = [t0 + float(on) * T_b + float(j) for (on, _, _), j in zip(sop, jit)]
    out = []
    for i, (on, off, m) in enumerate(sop):
        if i + 1 < len(sop) and abs(float(sop[i + 1][0]) - float(off)) < 1e-9:
            e = ons[i + 1]
        else:
            e = t0 + float(off) * T_b
        e = max(e, ons[i] + MIN_NOTE)
        out.append((ons[i], e, int(m) + shift, float(on), float(off)))
    return out


def _sung(notes, rng_s, rng_x, tail=0.8, glide=0.06):
    """singer.render (copied: same formulas, same draw order from rng_s) plus onset scoops and per-note detune drawn
    from rng_x. notes = [(on_s, off_s, midi)]. Returns (audio float64 before room noise, info per note)."""
    n = int((notes[-1][1] + tail) * SR)
    t = np.arange(n) / SR
    midi = np.full(n, np.nan)
    amp = np.zeros(n)
    depth = np.zeros(n)
    info = []
    for i, (on, off, m) in enumerate(notes):
        me = m + rng_s.normal(0.0, 0.10)                    # singer.render's intonation error, semitones
        det = rng_x.normal(0.0, DETUNE_ST)                  # + the per-note detune
        sc = rng_x.random() < SCOOP_P
        d = rng_x.uniform(*SCOOP_CENTS) / 100.0
        s = rng_x.uniform(*SCOOP_S)
        me = me + det
        i0, i1 = int(on * SR), int(off * SR)
        midi[i0:i1] = me
        amp[i0:i1] = 0.75 + 0.25 * np.sin(np.pi * i / len(notes))
        tt = t[i0:i1] - on
        depth[i0:i1] = 0.30 * np.clip((tt - 0.12) / 0.3, 0.0, 1.0)
        prev_off = notes[i - 1][1] if i else -1.0
        legato = i > 0 and on - prev_off < 0.03
        kind = 'attack' if not legato else ('repeat' if notes[i - 1][2] == m else 'glide')
        info.append({'on': on, 'off': off, 'midi': m, 'kind': kind, 'midi_sung': me, 'detune': det,
                     'scoop': (d, s) if (kind == 'attack' and sc) else None})
    # pitch glides and amplitude shaping at note boundaries (singer.render), then the scoops
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
        if nt['scoop'] is not None:                          # starts d flat, reaches the pitch after s seconds
            d, s = nt['scoop']
            a, b = int(on * SR), min(int((on + s) * SR), int(off * SR))
            if b > a:
                w = 0.5 - 0.5 * np.cos(np.pi * np.arange(b - a) / (b - a))
                midi[a:b] = nt['midi_sung'] - d * (1.0 - w)
    # keep the phase running through silences
    valid = ~np.isnan(midi)
    idx = np.where(valid, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    midi_f = midi[idx]
    midi_f[: np.argmax(valid)] = midi[np.argmax(valid)]
    drift = np.cumsum(rng_s.normal(0, 1, n // 240 + 1))
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
        y += keep * singer._formant_gain(fh) / h * np.sin(h * phase)
    y *= amp
    y += 0.012 * colored_noise(n, rng_s, -0.3) * amp          # breathiness
    for i, nt in enumerate(info):                              # audible in-breaths before phrases
        if nt['kind'] == 'attack' and i > 0:
            sl = slice(int((nt['on'] - 0.12) * SR), int((nt['on'] - 0.02) * SR))
            m = sl.stop - sl.start
            y[sl] += 0.01 * colored_noise(m, rng_s, 0.2) * np.hanning(m)
    y *= 0.7 / np.max(np.abs(y))
    return y, info


def render(song, voice, t0, T_b):
    """Her voice for one song. Returns (audio float32 from acoustic time 0, her true notes as true_notes() gives them
    with the voice's own pitch, facts dict)."""
    check_allowed(song)
    if voice not in VOICES:
        raise ValueError(voice)
    shift = -12 if voice == 'sung_low' else 0
    notes = true_notes(song, t0, T_b, shift)
    plain = [(on, off, m) for on, off, m, _, _ in notes]
    if voice == 'clean':
        y = render_clean([(on, max(on + 0.02, off - CLEAN_GAP), m) for on, off, m in plain])
        return y.astype(np.float32), notes, {'voice': voice, 'scooped': 0, 'notes': len(notes)}
    y, info = _sung(plain, seed(song, 2), seed(song, 3))
    mask = np.zeros(len(y), bool)
    for on, off, _ in plain:
        mask[int(on * SR):int(off * SR)] = True
    rms_v = float(np.sqrt(np.mean(y[mask] ** 2)))
    noise = colored_noise(len(y), seed(song, 4), -0.5) * rms_v * 10.0 ** (NOISE_DB / 20.0)
    y = y + noise
    facts = {'voice': voice, 'notes': len(notes), 'scooped': sum(nt['scoop'] is not None for nt in info),
             'attack_notes': sum(nt['kind'] == 'attack' for nt in info),
             'repeat_notes': sum(nt['kind'] == 'repeat' for nt in info),
             'glide_notes': sum(nt['kind'] == 'glide' for nt in info),
             'voice_rms': rms_v, 'noise_rms': float(np.sqrt(np.mean(noise ** 2))),
             'peak': float(np.max(np.abs(y)))}
    return y.astype(np.float32), notes, facts
