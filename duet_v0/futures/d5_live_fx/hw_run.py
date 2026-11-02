"""Board / hardware runs for D5 (NOT run by the builder: it opens audio devices). See HARDWARE_TEST.md.

    python futures/d5_live_fx/hw_run.py devices
    python futures/d5_live_fx/hw_run.py roundtrip  --sr 48000 --block 128            # cable/acoustic loopback, engine bypassed
    python futures/d5_live_fx/hw_run.py stereo     --sr 48000 --block 128            # 2-in/2-out: out L -> in L (=mic), engine -> out R -> in R
    python futures/d5_live_fx/hw_run.py live       --sr 48000 --block 128 --tonic 0 --seconds 180 [--gain-db 0] [--no-guard]

Recordings go to futures/d5_live_fx/local_only/hw/ (her voice: local only).
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
from dsp_io import write_wav  # noqa: E402

OUT = HERE / 'local_only' / 'hw'


def clicks(sr, n_clicks=10, gap=0.5):
    x = np.zeros(int((n_clicks * gap + 1.0) * sr))
    for k in range(n_clicks):
        i = int((0.5 + k * gap) * sr)
        x[i:i + 48] = np.hanning(48) * 0.5
    return x


def lag_of(a, b, sr, max_s=0.2):
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    cc = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)[:int(max_s * sr)]
    return 1000.0 * int(np.argmax(cc)) / sr


def roundtrip(a):
    import sounddevice as sd
    x = clicks(a.sr)
    y = sd.playrec(x, samplerate=a.sr, channels=1, blocksize=a.block, latency='low', dtype='float32')
    sd.wait()
    y = y[:, 0]
    OUT.mkdir(parents=True, exist_ok=True)
    write_wav(OUT / 'roundtrip.wav', np.stack([x, y], 1), a.sr)
    ms = lag_of(x, y, a.sr)
    print(f'round trip (out -> in, engine bypassed): {ms:.2f} ms  peak in {20 * np.log10(np.max(np.abs(y)) + 1e-9):.1f} dBFS')
    return ms


def stereo(a):
    """Out L plays a sung test phrase (synthetic), looped by cable to in L = the 'mic'; the engine's output
    goes to out R, looped by cable to in R. Latency mic-in -> speaker-out = delay(in R vs in L) as heard
    by the same ADC, which includes ADC + processing + DAC + all buffers."""
    import sounddevice as sd
    from test_fx import tone
    sr, B = a.sr, a.block
    melody = [69, 72, 74, 72, 69, 67, 69]
    src = np.concatenate([np.concatenate([tone(float(fx.midi_to_hz(m)), 0.4, sr=sr), np.zeros(int(0.1 * sr))])
                          for m in melody]) * 0.5
    eng = fx.Harmonizer(tonic=0, sr=sr, block=B, aec=False, guard=False)
    rec = {'L': [], 'R': []}
    pos = [0]
    dts = []

    def cb(indata, outdata, frames, t, status):
        t0 = time.perf_counter()
        y = eng.process(indata[:, 0].astype(float))
        i = pos[0]
        seg = src[i:i + frames]
        outdata[:, 0] = np.pad(seg, (0, frames - len(seg)))
        outdata[:, 1] = y
        pos[0] += frames
        rec['L'].append(indata[:, 0].copy())
        rec['R'].append(indata[:, 1].copy())
        dts.append(time.perf_counter() - t0)

    with sd.Stream(samplerate=sr, blocksize=B, channels=2, latency='low', dtype='float32', callback=cb):
        sd.sleep(int(1000 * (len(src) / sr + 1.0)))
    L, R = np.concatenate(rec['L']), np.concatenate(rec['R'])
    OUT.mkdir(parents=True, exist_ok=True)
    write_wav(OUT / 'stereo_loop.wav', np.stack([L, R], 1), sr)
    eL = np.convolve(L ** 2, np.ones(48) / 48, 'same')
    eR = np.convolve(R ** 2, np.ones(48) / 48, 'same')
    ms = lag_of(eL - eL.mean(), eR - eR.mean(), sr, 0.1)
    d = np.array(dts) * 1000
    res = {'mic_to_speaker_ms': ms, 'compute_ms_mean': float(d.mean()), 'compute_ms_p99': float(np.percentile(d, 99)),
           'compute_ms_max': float(d.max()), 'block_ms': 1000 * B / sr}
    print(json.dumps(res, indent=1))
    (OUT / 'stereo_loop.json').write_text(json.dumps(res, indent=1))


def live(a):
    import sounddevice as sd
    sr, B = a.sr, a.block
    eng = fx.Harmonizer(tonic=a.tonic, sr=sr, block=B, aec=not a.no_guard, guard=not a.no_guard,
                        out_gain=10 ** (a.gain_db / 20))
    if not a.no_guard:
        print('sound check: soft noise from the speaker for 2 s, please stay silent')
        rng = np.random.default_rng(0)
        ref = (10 ** (-35 / 20) * rng.standard_normal(2 * sr)).astype('float32')
        mic = sd.playrec(ref, samplerate=sr, channels=1, blocksize=B, latency='low')[:, 0]
        sd.wait()
        print('sound check ERLE (dB):', eng.calibrate(mic.astype(float), ref.astype(float)))
    n = int(a.seconds * sr)
    mic_rec, out_rec, dts = np.zeros(n + B), np.zeros(n + B), []
    pos = [0]

    def cb(indata, outdata, frames, t, status):
        t0 = time.perf_counter()
        y = eng.process(indata[:, 0].astype(float))
        outdata[:, 0] = y
        i = pos[0]
        if i + frames <= n:
            mic_rec[i:i + frames] = indata[:, 0]
            out_rec[i:i + frames] = y
        pos[0] += frames
        dts.append(time.perf_counter() - t0)

    print(f'live: sing now ({a.seconds:.0f} s, key tonic pc {a.tonic})')
    with sd.Stream(samplerate=sr, blocksize=B, channels=1, latency='low', dtype='float32', callback=cb):
        sd.sleep(int(1000 * a.seconds))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = time.strftime('%H%M%S') + ('_noguard' if a.no_guard else '') + f'_g{a.gain_db:+.0f}'
    write_wav(OUT / f'live_{tag}.wav', np.stack([mic_rec[:n], out_rec[:n]], 1), sr)
    d = np.array(dts) * 1000
    print(f'compute per block: mean {d.mean():.3f} ms, p99 {np.percentile(d, 99):.3f} ms, max {d.max():.3f} ms '
          f'(block {1000 * B / sr:.2f} ms); saved live_{tag}.wav (L = mic, R = speaker)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('devices', 'roundtrip', 'stereo', 'live'))
    ap.add_argument('--sr', type=int, default=48000)
    ap.add_argument('--block', type=int, default=128)
    ap.add_argument('--tonic', type=int, default=0)
    ap.add_argument('--seconds', type=float, default=180.0)
    ap.add_argument('--gain-db', type=float, default=0.0)
    ap.add_argument('--no-guard', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'devices':
        import sounddevice as sd
        print(sd.query_devices())
    else:
        {'roundtrip': roundtrip, 'stereo': stereo, 'live': live}[a.cmd](a)
