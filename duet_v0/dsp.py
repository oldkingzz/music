"""Signal tools shared by the Duet v0 prototype (numpy only).

Everything runs at SR = 24 kHz. One streaming step is HOP = 240 samples (10 ms);
pitch is estimated on the last WIN = 960 samples (40 ms) with YIN.
"""
import wave

import numpy as np

SR = 24000
HOP = 240
WIN = 960
FMIN, FMAX = 70.0, 1000.0
TAU_MAX = int(SR / FMIN)        # longest period searched (samples)
TAU_MIN = int(SR / FMAX)        # shortest period searched (samples)
W_INT = WIN - TAU_MAX           # YIN integration window
NFFT = 2048


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.asarray(f, dtype=float) / 440.0)


def read_wav(path):
    """Read a PCM wav file as mono float32. It must already be 24 kHz."""
    with wave.open(str(path), 'rb') as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, '<i2').astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float32) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(raw, '<i4').astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f'{path}: unsupported sample width {sw}')
    x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        raise ValueError(f'{path} is {sr} Hz. Convert first: afconvert -f WAVE -d LEI16@{SR} -c 1 IN OUT.wav')
    return x


def write_wav(path, x, sr=SR):
    """Write mono (n,) or multichannel (n, ch) float audio as 16-bit PCM."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def yin(frame, thresh=0.15):
    """YIN pitch on one WIN-sample frame. Returns (f0 Hz, aperiodicity 0..1, rms)."""
    x = np.asarray(frame, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    a = x[:W_INT]
    r = np.fft.irfft(np.conj(np.fft.rfft(a, NFFT)) * np.fft.rfft(x, NFFT), NFFT)[:TAU_MAX + 1]
    cs = np.concatenate(([0.0], np.cumsum(x * x)))
    taus = np.arange(TAU_MAX + 1)
    d = cs[W_INT] + (cs[taus + W_INT] - cs[taus]) - 2.0 * r
    d[0] = 0.0
    dn = np.ones_like(d)
    dn[1:] = d[1:] * taus[1:] / np.maximum(np.cumsum(d[1:]), 1e-12)
    seg = dn[TAU_MIN:]
    below = np.nonzero(seg < thresh)[0]
    if below.size:
        k = int(below[0])
        while k + 1 < seg.size and seg[k + 1] < seg[k]:
            k += 1
    else:
        k = int(np.argmin(seg))
    tau = k + TAU_MIN
    tauf = float(tau)
    if 1 <= tau < TAU_MAX:
        s0, s1, s2 = dn[tau - 1], dn[tau], dn[tau + 1]
        den = s0 - 2.0 * s1 + s2
        if abs(den) > 1e-12:
            tauf = tau + float(np.clip(0.5 * (s0 - s2) / den, -1.0, 1.0))
    return SR / tauf, float(dn[tau]), rms


def colored_noise(n, rng, slope=-0.5):
    """Unit-RMS noise; slope -0.5 on amplitude gives a pink (-3 dB/octave) tilt."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    spec *= (f / 1000.0) ** slope
    y = np.fft.irfft(spec, n)
    return y / (np.sqrt(np.mean(y * y)) + 1e-12)


def wsola(x, src_center, n_out, tol=96):
    """Time-scale x without changing pitch (WSOLA).

    src_center(t) maps an output frame-centre time (s) to the source time (s) that
    should sound there. Each frame may slide up to `tol` samples to stay in phase.
    """
    N, Hs = WIN, WIN // 2
    win = np.hanning(N + 1)[:N]
    pad = N + tol
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    out = np.zeros(n_out + N)
    norm = np.zeros(n_out + N)
    prev = None
    m = 0
    while m * Hs < n_out:
        o = m * Hs
        s = int(round(src_center((o + N / 2) / SR) * SR - N / 2))
        s = min(max(s, -N // 2), len(x))
        if prev is not None:
            nat = xp[pad + prev + Hs: pad + prev + Hs + N]
            if np.dot(nat, nat) > 1e-8:
                seg = xp[pad + s - tol: pad + s + tol + N]
                s = s - tol + int(np.argmax(np.correlate(seg, nat, 'valid')))
        out[o:o + N] += xp[pad + s: pad + s + N] * win
        norm[o:o + N] += win
        prev = s
        m += 1
    norm[norm < 0.05] = 1.0
    return (out / norm)[:n_out]


def render_voice(notes, n, level=0.22):
    """Soft 'oo' voice for the machine. notes: list of (start_s, dur_s, midi)."""
    out = np.zeros(n)
    for start, dur, m in notes:
        i0 = int(round(start * SR))
        length = int((dur + 0.08) * SR)
        if i0 >= n or length <= 0:
            continue
        t = np.arange(length) / SR
        vib = 0.15 * np.sin(2 * np.pi * 5.0 * t) * np.clip((t - 0.15) / 0.2, 0.0, 1.0)
        ph = 2 * np.pi * np.cumsum(midi_to_hz(m + vib)) / SR
        y = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
        env = np.clip(t / 0.025, 0.0, 1.0) * np.clip((dur + 0.06 - t) / 0.06, 0.0, 1.0)
        i1 = min(n, i0 + length)
        out[i0:i1] += (y * env * level)[: i1 - i0]
    return out
