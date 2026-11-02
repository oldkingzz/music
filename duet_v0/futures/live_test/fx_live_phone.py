"""D5 live on the MacBook: the phone plays her near the mic; the engine (iter2) harmonizes the mic in real
time and plays a diatonic third below through the MacBook speaker. Records L = mic (what the room heard),
R = engine output. Usage: fx_live_phone.py TONIC SECONDS"""
import sys, time
from pathlib import Path
import numpy as np, sounddevice as sd, soundfile as sf
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'd5_live_fx' / 'iter2'))
import fx  # noqa: E402
fx.DEFAULTS.update({'drop_db': 16, 'onset_db': 26, 'f0_tau': 0.015, 'mode': 'follow'})   # iter2 chosen config v2
tonic, secs = int(sys.argv[1]), float(sys.argv[2])
SR, B, LAT, DEV = 48000, 128, 0.005, (2, 3)
eng = fx.Harmonizer(tonic=tonic, sr=SR, block=B)
rng = np.random.default_rng(0)
ref = (10 ** (-35 / 20) * rng.standard_normal(2 * SR)).astype('float32')
mic = sd.playrec(ref, samplerate=SR, channels=1, blocksize=B, latency=LAT, device=DEV)[:, 0]
sd.wait()
print('sound check ERLE dB:', eng.calibrate(mic.astype(float), ref.astype(float)), flush=True)
n = int(secs * SR)
mic_rec, out_rec, dts, xr = np.zeros(n + B), np.zeros(n + B), [], [0]
pos = [0]
def cb(indata, outdata, frames, t, status):
    if status:
        xr[0] += 1
    t0 = time.perf_counter()
    y = eng.process(indata[:, 0].astype(float))
    outdata[:, 0] = y
    i = pos[0]
    if i + frames <= n:
        mic_rec[i:i + frames] = indata[:, 0]; out_rec[i:i + frames] = y
    pos[0] += frames
    dts.append(time.perf_counter() - t0)
print(f'LIVE now for {secs:.0f} s: play the clip on the phone', flush=True)
with sd.Stream(samplerate=SR, blocksize=B, channels=1, latency=LAT, dtype='float32', callback=cb, device=DEV) as s:
    print('stream latency in/out ms:', [round(1000 * v, 1) for v in s.latency], flush=True)
    sd.sleep(int(1000 * secs))
mic_rec, out_rec = mic_rec[:n], out_rec[:n]
stamp = time.strftime('%H%M%S')
sf.write(HERE / f'fx_phone_{stamp}.wav', np.stack([mic_rec, out_rec], 1), SR)
d = np.array(dts) * 1000
lv = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)  # noqa: E731
w = int(0.1 * SR)
mic_db = np.array([lv(mic_rec[i:i + w]) for i in range(0, n - w, w)])
out_db = np.array([lv(out_rec[i:i + w]) for i in range(0, n - w, w)])
floor = np.median(np.sort(mic_db)[:max(1, len(mic_db) // 5)])
singing = mic_db > floor + 10
print(f'saved fx_phone_{stamp}.wav; compute per block mean {d.mean():.3f} ms p99 {np.percentile(d, 99):.3f} ms; xruns {xr[0]}')
print(f'mic floor {floor:.1f} dBFS; loud (she plays) {100 * singing.mean():.0f}% of the window; '
      f'Duet output above -40 dBFS while she plays {100 * np.mean(out_db[singing] > -40) if singing.any() else 0:.0f}%, '
      f'while quiet {100 * np.mean(out_db[~singing] > -40):.0f}%; output peak {out_db.max():.1f} dBFS; mic peak {mic_db.max():.1f}')
print(stamp)
