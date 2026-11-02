"""A5.1: algorithmic latency in simulation + per-block compute time.

    .venv/bin/python futures/d5_live_fx/latency.py            -> results/latency.json
    OMP_NUM_THREADS=1 ... taskpolicy -c background .venv/bin/python futures/d5_live_fx/latency.py --timing-only --tag ecore
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fx  # noqa: E402
from test_fx import tone  # noqa: E402

SR = 24000
OUT = HERE / 'results'


def envelope(y, ms=1.0):
    n = max(1, int(ms * SR / 1000))
    return np.sqrt(np.convolve(y * y, np.ones(n) / n, 'same'))


def track(y, hop=24):
    yin = fx.StreamYin(SR, 150, 1000)
    t, m = [], []
    for k in range(yin.N, len(y), hop):
        f, ap = yin(y[k - yin.N:k])
        t.append(k / SR)
        m.append(float(fx.hz_to_midi(f)) if ap < 0.25 and np.sqrt(np.mean(y[k - yin.N:k] ** 2)) > 1e-3 else np.nan)
    return np.array(t), np.array(m)


def onset_latency(block):
    """Silence then a sung note (5 ms attack): when does the harmony reach -6 dB of its steady level?"""
    res = []
    for f0 in (196.0, 262.0, 440.0, 660.0):
        t0 = 0.3
        x = np.concatenate([np.zeros(int(t0 * SR)), tone(f0, 0.7)])
        x[int(t0 * SR):int(t0 * SR) + 120] *= np.linspace(0, 1, 120)
        y, _ = fx.run_open(x, 0, aec=False, block=block)
        e = envelope(y)
        ss = np.median(e[int(0.6 * SR):int(0.9 * SR)])
        i = int(np.argmax(e > ss * 0.5))
        ex = envelope(x[:len(y)])
        ix = int(np.argmax(ex > np.median(ex[int(0.6 * SR):int(0.9 * SR)]) * 0.5))
        res.append({'f0': f0, 'onset_ms': 1000 * (i - ix) / SR})
    return res


def path_delay(block):
    """Steady note with 30 Hz amplitude modulation: delay of the harmony's envelope vs hers."""
    res = []
    for f0 in (196.0, 262.0, 440.0, 660.0):
        x = tone(f0, 1.5)
        t = np.arange(len(x)) / SR
        x = x * (1 + 0.5 * np.sin(2 * np.pi * 30 * t))
        y, _ = fx.run_open(x, 0, aec=False, block=block)
        a = envelope(x[:len(y)], 2)[SR // 2:]
        b = envelope(y, 2)[SR // 2:]
        a, b = a - a.mean(), b - b.mean()
        lags = np.arange(0, int(0.03 * SR))
        cc = [np.dot(a[:len(a) - L], b[L:]) for L in lags]
        res.append({'f0': f0, 'delay_ms': 1000 * float(lags[int(np.argmax(cc))]) / SR})
    return res


def pitch_step(block):
    """She steps A4 -> C5 (phase-continuous). Times measured with the same tracker on input and output,
    so the tracker's own lag cancels: leave = harmony leaves its old note, settle = within 0.5 st of
    the new diatonic third (A4) and stays."""
    res = []
    for (m0, m1) in ((69, 72), (72, 69), (64, 65), (67, 71)):
        t1 = 0.6
        n = int(1.2 * SR)
        mid = np.where(np.arange(n) < t1 * SR, m0, m1).astype(float)
        f = fx.midi_to_hz(mid)
        ph = 2 * np.pi * np.cumsum(f) / SR
        x = sum((0.5 if h == 1 else 0.3 / h) * np.sin(h * ph) for h in range(1, 12)) * 0.2
        y, _ = fx.run_open(x, 0, aec=False, block=block)
        ti, mi = track(x[:len(y)])
        to, mo = track(y)
        old, new = fx.third_below(m0, 0), fx.third_below(m1, 0)
        cross_in = ti[np.argmax((ti > t1) & (np.abs(mi - m1) < 0.5))]
        after = to > t1
        leave = to[np.argmax(after & (np.abs(mo - old) > 0.5))]
        ok = np.abs(mo - new) < 0.5
        idx = np.nonzero(after & ~ok)[0]
        settle = to[idx[-1] + 1] if idx.size and idx[-1] + 1 < len(to) else t1
        res.append({'from': m0, 'to': m1, 'leave_ms': 1000 * (leave - cross_in), 'settle_ms': 1000 * (settle - cross_in),
                    'interim_interval': int(m1 - (m1 - (m0 - old)))})
    return res


def timing(block, sr, dur=20.0, partitions=64):
    """Per-block compute time over a sung signal with the full engine (AEC + gate + PSOLA + limiter)."""
    rng = np.random.default_rng(0)
    melody = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62]
    x = np.concatenate([tone(float(fx.midi_to_hz(m)), 0.5, sr=sr, vib=0.3) for m in melody])
    x = np.tile(x, int(np.ceil(dur * sr / len(x))))[:int(dur * sr)]
    x = x + 0.002 * rng.standard_normal(len(x))
    e = fx.Harmonizer(tonic=0, sr=sr, block=block, partitions=partitions)
    ref = np.zeros(block)
    ts = []
    n = len(x) // block * block
    for k in range(0, n, block):
        mic = x[k:k + block] + 0.1 * np.roll(ref, 3)
        t0 = time.perf_counter()
        ref = e.process(mic)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts[50:]) * 1e6
    return {'sr': sr, 'block': block, 'block_ms': 1000 * block / sr, 'partitions': partitions,
            'aec_ms': 1000 * partitions * block / sr,
            'mean_us': float(ts.mean()), 'p99_us': float(np.percentile(ts, 99)), 'max_us': float(ts.max()),
            'load_mean': float(ts.mean() / (1e6 * block / sr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing-only', action='store_true')
    ap.add_argument('--tag', default='m4')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tm = [timing(64, 24000), timing(128, 48000, partitions=64)]
    if a.timing_only:
        (OUT / f'timing_{a.tag}.json').write_text(json.dumps(tm, indent=1))
        print(json.dumps(tm, indent=1))
        return
    res = {'block': 64, 'sr': SR, 'onset': onset_latency(64), 'path_delay': path_delay(64),
           'pitch_step': pitch_step(64), f'timing_{a.tag}': tm}
    (OUT / 'latency.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
