"""Self-tests on hand-made cases (asserts). .venv/bin/python futures/d5_live_fx/test_fx.py"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fx  # noqa: E402

SR = 24000


def tone(f0, dur, sr=SR, formants=((730, 90), (1090, 110), (2440, 170)), amp=0.3, vib=0.0):
    t = np.arange(int(dur * sr)) / sr
    f = f0 * 2 ** (vib * np.sin(2 * np.pi * 5.5 * t) / 12)
    ph = 2 * np.pi * np.cumsum(f) / sr
    y = np.zeros_like(t)
    for h in range(1, 40):
        fh = h * f0
        if fh > 0.45 * sr:
            break
        g = 0.03 + sum(1 / np.sqrt(1 + ((fh - fc) / (bw / 2)) ** 2) for fc, bw in formants)
        y += g / h * np.sin(h * ph)
    return amp * y / np.max(np.abs(y))


def pitch_of(y, sr=SR):
    yin = fx.StreamYin(sr, 70, 1000)
    fs = []
    for k in range(yin.N, len(y), 240):
        f, ap = yin(y[k - yin.N:k])
        if ap < 0.2:
            fs.append(f)
    return float(np.median(fs)) if fs else np.nan


def env_db(y, f0, sr=SR, grid=np.linspace(300, 3500, 200)):
    """Spectral envelope: log magnitude at the harmonics of f0, interpolated on a grid (middle 0.4 s)."""
    m = len(y) // 2
    seg = y[m - 4800:m + 4800] * np.hanning(9600)
    S = np.abs(np.fft.rfft(seg, 1 << 15))
    fr = np.fft.rfftfreq(1 << 15, 1 / sr)
    hs = np.arange(1, int(4000 / f0) + 1) * f0
    amp = [S[(fr > h - 0.3 * f0) & (fr < h + 0.3 * f0)].max() for h in hs]
    return np.interp(grid, hs, 20 * np.log10(np.array(amp) + 1e-9))


def centroid(y, sr=SR):
    S = np.abs(np.fft.rfft(y)) ** 2
    fr = np.fft.rfftfreq(len(y), 1 / sr)
    return float(np.sum(fr * S) / np.sum(S))


def main():
    # 1. diatonic third below, C major
    tb = {60: 57, 62: 59, 64: 60, 65: 62, 67: 64, 69: 65, 71: 67, 72: 69}
    for n, t in tb.items():
        assert fx.third_below(n, 0) == t, (n, fx.third_below(n, 0))
    for n in range(48, 84):                               # any key, any note: a third (3 or 4) below
        for ton in range(12):
            assert n - fx.third_below(n, ton) in (3, 4), (n, ton)
    # 2. YIN accuracy, female window
    y = fx.StreamYin(SR, 140, 1000)
    assert y.N == 2 * 172, y.N
    for f0 in (150, 165, 220, 330, 440, 660, 880):
        x = tone(f0, 0.1)
        f, ap = y(x[-y.N:])
        assert abs(f / f0 - 1) < 0.01 and ap < 0.15, (f0, f, ap)
    # 3. shifter pitch: a sustained note goes a diatonic third down (A4 in C major -> F4, 4 semitones)
    res = {}
    for f0, ton in ((440.0, 0), (261.63, 0), (196.0, 0), (659.26, 0)):
        x = tone(f0, 1.0)
        out, e = fx.run_open(x, ton, aec=False)
        note = int(round(float(fx.hz_to_midi(f0))))
        want = float(fx.midi_to_hz(fx.third_below(note, ton))) * f0 / float(fx.midi_to_hz(note))
        got = pitch_of(out[4800:])
        res[f0] = (got, want)
        assert abs(got / want - 1) < 0.01, (f0, got, want)
        r = np.sqrt(np.mean(out[4800:] ** 2)) / np.sqrt(np.mean(x[4800:] ** 2))
        assert 0.6 < r < 1.4, ('gain', f0, r)
    # 4. formants preserved: harmonic envelope of the output vs the input, vs a naive resampling shifter
    f0 = 220.0
    x = tone(f0, 1.0)
    out, _ = fx.run_open(x, 0, aec=False)          # A3 -> F3, 4 semitones down
    ratio = 2 ** (-4 / 12)
    naive = np.interp(np.arange(0, len(x)) / ratio, np.arange(len(x)), x)   # resample: formants move too
    naive = np.interp(np.arange(len(x)), np.arange(len(naive)) * 1.0, naive)
    naive = np.interp(np.arange(len(x)) * ratio, np.arange(len(x)), x)
    ex, eo, en = env_db(x, f0), env_db(out, f0 * ratio), env_db(naive, f0 * ratio)
    d_psola = float(np.mean(np.abs(ex - eo - np.mean(ex - eo))))
    d_naive = float(np.mean(np.abs(ex - en - np.mean(ex - en))))
    c_psola = centroid(out[4800:]) / centroid(x[4800:])
    c_naive = centroid(naive[4800:]) / centroid(x[4800:])
    assert abs(c_psola - 1) < 0.05 and abs(c_naive - ratio) < 0.05, (d_psola, d_naive, c_psola, c_naive)
    # 5. gate: silence in -> silence out; noise in -> silence out
    rng = np.random.default_rng(0)
    out, e = fx.run_open(0.003 * rng.standard_normal(SR), 0, aec=False)
    assert np.max(np.abs(out)) < 1e-6, np.max(np.abs(out))
    print('test_fx OK', {k: tuple(round(v, 1) for v in vv) for k, vv in res.items()},
          'envelope error dB psola %.2f naive %.2f; centroid ratio psola %.3f naive %.3f' % (d_psola, d_naive, c_psola, c_naive))
    return {'shift_pitch_hz_got_want': res, 'envelope_err_db_psola': d_psola, 'envelope_err_db_naive_resample': d_naive,
            'centroid_ratio_psola': c_psola, 'centroid_ratio_naive': c_naive}


if __name__ == '__main__':
    main()
