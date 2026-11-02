"""Make a "live performance" from a reference recording, with known ground truth.

The reference is re-timed (tempo sway, slowing into breaths), and optionally gets
a pause, a skipped passage, or noise. Because we do the re-timing, we know for every
live instant which reference instant is sounding, so any onset can be scored.
"""
import numpy as np

from dsp import SR, colored_noise, wsola


class Performance:
    def __init__(self, audio, segments, length):
        self.audio = audio
        self.segments = segments          # [('audio', lt, rt) | ('silence', t0, t1)]
        self.length = length

    def live_to_ref(self, t):
        """Reference time sounding at live time t (NaN during inserted silence)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.full(t.shape, np.nan)
        for seg in self.segments:
            if seg[0] != 'audio':
                continue
            lt, rt = seg[1], seg[2]
            sel = (t >= lt[0]) & (t < lt[-1])
            out[sel] = np.interp(t[sel], lt, rt)
        return out

    def ref_to_live(self, r):
        """First live time at which reference time r sounds (NaN if it was skipped)."""
        for seg in self.segments:
            if seg[0] == 'audio' and seg[2][0] <= r <= seg[2][-1]:
                return float(np.interp(r, seg[2], seg[1]))
        return float('nan')


def tempo_ratio(r, gaps, sway, period, rit, rit_len, phase=0.7):
    """Live seconds per reference second at reference time r."""
    k = 1.0 + sway * np.sin(2 * np.pi * r / period + phase)
    for g0, _ in gaps:
        if g0 - rit_len <= r < g0:
            k += rit * ((r - (g0 - rit_len)) / rit_len) ** 2
    return k


def make(x, gaps, sway=0.0, period=8.0, rit=0.0, rit_len=2.5, pause=None, jump=None,
         noise=(), room_db=-60.0, seed=3, phase=0.7):
    """Build a Performance.

    gaps: [(start, end)] silent stretches in the reference (breaths), in seconds.
    pause: (gap_index, seconds) inserts silence in the middle of that gap.
    jump: (gap_a, gap_b) jumps from the middle of gap_a to the middle of gap_b.
    noise: [(live_start, live_end, dB relative to the voice RMS)].
    """
    rng = np.random.default_rng(seed)
    dur = len(x) / SR
    centers = [(a + b) / 2 for a, b in gaps]
    events = {}
    if pause:
        events[centers[pause[0]]] = ('pause', pause[1])
    if jump:
        events[centers[jump[0]]] = ('jump', centers[jump[1]])
    dr = 0.001
    segments, lt, rt = [], [0.0], [0.0]
    t, r = 0.0, 0.0
    pending = sorted(events)
    while r < dur:
        if pending and r >= pending[0]:
            kind, val = events[pending.pop(0)]
            segments.append(('audio', np.array(lt), np.array(rt)))
            if kind == 'pause':
                segments.append(('silence', t, t + val))
                t += val
            else:
                r = val
                pending = [p for p in pending if p > r]
            lt, rt = [t], [r]
        t += tempo_ratio(r, gaps, sway, period, rit, rit_len, phase) * dr
        r += dr
        lt.append(t)
        rt.append(r)
    segments.append(('audio', np.array(lt), np.array(rt)))
    length = t
    n = int(length * SR) + 1
    y = np.zeros(n)
    fade = int(0.01 * SR)
    for seg in segments:
        if seg[0] != 'audio' or len(seg[1]) < 3:
            continue
        slt, srt = seg[1], seg[2]
        i0 = int(round(slt[0] * SR))
        m = int(round((slt[-1] - slt[0]) * SR))
        part = wsola(x, lambda tc, slt=slt, srt=srt: np.interp(slt[0] + tc, slt, srt), m)
        part[:fade] *= np.linspace(0, 1, fade)
        part[-fade:] *= np.linspace(1, 0, fade)
        y[i0:i0 + m] += part[: n - i0]
    voice_rms = np.sqrt(np.mean(x[np.abs(x) > 0.02] ** 2)) if np.any(np.abs(x) > 0.02) else 0.1
    y += voice_rms * 10 ** (room_db / 20) * colored_noise(n, rng, -0.5)
    for a, b, db in noise:
        i0, i1 = int(a * SR), min(n, int(b * SR))
        burst = colored_noise(i1 - i0, rng, -0.3) * np.hanning(i1 - i0) ** 0.3
        y[i0:i1] += voice_rms * 10 ** (db / 20) * burst
    return Performance(y.astype(np.float32), segments, length)
