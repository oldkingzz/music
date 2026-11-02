"""Wait for a sound source (Vincent's phone) near the MacBook mic, record it, stop after it goes quiet.
Usage: listen_rec.py OUT.wav [--max 90] [--quiet 2.5]. Prints the levels; writes OUT.wav (48 kHz mono)."""
import argparse, queue, time
import numpy as np, sounddevice as sd, soundfile as sf
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--max', type=float, default=90)
ap.add_argument('--quiet', type=float, default=4.0); ap.add_argument('--rise', type=float, default=12.0)
a = ap.parse_args()
SR, B = 48000, 2400                                   # 50 ms blocks
q = queue.Queue()
with sd.InputStream(samplerate=SR, channels=1, blocksize=B, device=2, dtype='float32',
                    callback=lambda x, f, t, s: q.put(x[:, 0].copy())):
    blocks, lv, t0 = [], [], time.time()
    started, last_loud = None, None
    while time.time() - t0 < a.max:
        x = q.get(); blocks.append(x)
        l = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12); lv.append(l)
        floor = np.median(lv[:40]) if len(lv) >= 10 else -43.0
        i = len(blocks) - 1
        loud = [v > floor + a.rise for v in lv[-10:]]
        if l > floor + a.rise and started is not None:
            last_loud = i
        if started is None and len(loud) == 10 and sum(loud) >= 6:      # sustained: 0.3 s of the last 0.5 s
            started = i - 9; last_loud = i
            print(f'heard sustained sound at {started * 0.05:.1f} s (level {l:.1f} dBFS, floor {floor:.1f})', flush=True)
        if started is not None and (i - last_loud) * 0.05 > a.quiet:
            break
x = np.concatenate(blocks)
if started is None:
    print(f'no sound above floor+{a.rise} dB in {a.max:.0f} s; floor {np.median(lv):.1f} dBFS, max {max(lv):.1f}')
    sf.write(a.out.replace('.wav', '_nothing.wav'), x, SR)
else:
    s0 = max(0, (started - 10) * B); y = x[s0:]
    sf.write(a.out, y, SR)
    act = np.array(lv[started:]); print(f'recorded {len(y)/SR:.1f} s -> {a.out}; loud blocks median {np.median(act):.1f} dBFS, peak {20*np.log10(np.abs(y).max()+1e-12):.1f} dBFS, floor {np.median(lv[:max(1, started)]):.1f}')
