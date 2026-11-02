"""Real-room tests on this MacBook (built-in speaker + mic), run by the lead after Vincent said 开始.

  roundtrip  clicks out -> mic, 5 repeats: the Mac's own audio round trip (no engine)
  fx         D5 live: a vocadito singer is played from the speaker (stands in for her in the room),
             the mic hears her + Duet, the D5 engine harmonizes the mic in real time and its output is
             mixed into the same speaker. Records mic (L) and engine output (R). Scores Duet's output
             pitch against her annotated notes, checks for howling and for the gate after she stops.
  rerecord   plays vocadito excerpts from the speaker, records the mic: real-room input for D1/D2/D4.
Recordings stay in futures/live_test/ (local only).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd
import scipy.signal as ss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'd5_live_fx'))
import fx  # noqa: E402
import sim5  # noqa: E402

SR, B = 48000, 128
LAT = 0.005                                  # smallest buffer the built-in devices accept (reported in 75 ms, out 45 ms)
GOOD = {0, 3, 4, 7, 8, 9}
HARSH = {1, 2, 6, 10, 11}


def clicks(n=5, gap=0.6):
    x = np.zeros(int(SR * (0.3 + n * gap)), 'float32')
    for k in range(n):
        i = int(SR * (0.3 + k * gap))
        x[i:i + 48] = 0.5 * np.hanning(48)
    return x


def roundtrip():
    x = clicks()
    out = []
    for rep in range(3):
        y = sd.playrec(x, samplerate=SR, channels=1, blocksize=B, latency=LAT, device=(2, 3), dtype='float32')[:, 0]
        sd.wait()
        c = ss.correlate(y, x, mode='full', method='fft')[len(x) - 1:]
        out.append(1000 * int(np.argmax(np.abs(c[:int(0.8 * SR)]))) / SR)
    with sd.Stream(samplerate=SR, blocksize=B, channels=1, latency=LAT) as s:
        rep = [round(1000 * v, 1) for v in s.latency]
    return {'roundtrip_ms': out, 'reported_in_out_ms': rep}


def excerpt(i):
    x, notes, tonic, tun = sim5.load_excerpt(i)
    x = ss.resample_poly(x, SR // sim5.SR, 1)          # sim5 audio is 24 kHz
    return np.asarray(x, float), notes, tonic


def pitch_track(y, hop=480):
    import librosa
    z = librosa.resample(y, orig_sr=SR, target_sr=16000)
    f0, v, p = librosa.pyin(z, fmin=70, fmax=1100, sr=16000, frame_length=1024, hop_length=160)
    f0 = np.where(v & (p > 0.3), f0, np.nan)
    return f0, 0.01


def score(out, start_s, notes, rel_db=-40.0):
    f0, hop = pitch_track(out)
    lvl = np.array([20 * np.log10(np.sqrt(np.mean(out[int(k * hop * SR):int((k + 1) * hop * SR)] ** 2)) + 1e-12)
                    for k in range(len(f0))])
    tot = dict(good=0, harsh=0, fourth=0, silent=0, n=0)
    for on, off, m in notes:                      # sim5.load_excerpt gives (onset_s, offset_s, midi)
        for k in range(int((start_s + on) / hop), int((start_s + off) / hop)):
            if k >= len(f0):
                break
            tot['n'] += 1
            if lvl[k] < rel_db or np.isnan(f0[k]):
                tot['silent'] += 1
                continue
            h = int(round(69 + 12 * np.log2(f0[k] / 440.0)))
            iv = abs(int(round(m)) - h) % 12
            tot['good'] += iv in GOOD
            tot['harsh'] += iv in HARSH
            tot['fourth'] += iv == 5
    n = max(tot.pop('n'), 1)
    return {k: round(v / n, 4) for k, v in tot.items()}


def fx_live(ids, her_db=-14.0, gap_s=2.5):
    res = []
    for i in ids:
        x, notes, tonic = excerpt(i)
        x = x / (np.abs(x).max() + 1e-9) * 10 ** (her_db / 20)
        eng = fx.Harmonizer(tonic=tonic, sr=SR, block=B)
        rng = np.random.default_rng(0)
        ref = (10 ** (-35 / 20) * rng.standard_normal(2 * SR)).astype('float32')
        mic = sd.playrec(ref, samplerate=SR, channels=1, blocksize=B, latency=LAT, device=(2, 3))[:, 0]
        sd.wait()
        erle = eng.calibrate(mic.astype(float), ref.astype(float))
        lead = int(0.5 * SR)
        her = np.concatenate([np.zeros(lead), x, np.zeros(int(gap_s * SR))])
        n = len(her)
        mic_rec, out_rec, dts = np.zeros(n + B), np.zeros(n + B), []
        pos = [0]

        def cb(indata, outdata, frames, t, status):
            t0 = time.perf_counter()
            y = eng.process(indata[:, 0].astype(float))
            p = pos[0]
            h = her[p:p + frames] if p + frames <= n else np.zeros(frames)
            if len(h) < frames:
                h = np.pad(h, (0, frames - len(h)))
            outdata[:, 0] = np.clip(h + y, -1, 1)
            if p + frames <= n:
                mic_rec[p:p + frames] = indata[:, 0]
                out_rec[p:p + frames] = y
            pos[0] += frames
            dts.append(time.perf_counter() - t0)

        with sd.Stream(samplerate=SR, blocksize=B, channels=1, latency=LAT, dtype='float32', callback=cb, device=(2, 3)):
            sd.sleep(int(1000 * n / SR) + 300)
        mic_rec, out_rec = mic_rec[:n], out_rec[:n]
        sf.write(HERE / f'fx_room_v{i}.wav', np.stack([mic_rec, out_rec], 1), SR)
        end = (lead + len(x)) / SR
        tail = out_rec[int(end * SR):]
        fr = np.array([20 * np.log10(np.sqrt(np.mean(tail[k:k + 480] ** 2)) + 1e-12) for k in range(0, len(tail) - 480, 480)])
        below = np.argmax(fr < -40) * 0.01 if np.any(fr < -40) else None
        f, t, S = ss.spectrogram(mic_rec, SR, nperseg=4096, noverlap=3584)
        SdB = 10 * np.log10(S + 1e-15)
        pk = SdB.max(0) - np.median(SdB, 0)
        d = np.array(dts) * 1000
        res.append({'id': i, 'tonic': tonic, 'sound_check_erle_db': erle, 'score': score(out_rec, lead / SR, notes),
                    'gate_release_s_to_-40dBFS': below, 'tail_max_dbfs': round(float(fr.max()), 1) if len(fr) else None,
                    'mic_peak_dbfs': round(float(20 * np.log10(np.abs(mic_rec).max() + 1e-12)), 1),
                    'out_peak_dbfs': round(float(20 * np.log10(np.abs(out_rec).max() + 1e-12)), 1),
                    'max_peak_to_median_db': round(float(pk.max()), 1),
                    'mic_level_last_vs_first_s_db': round(float(20 * np.log10((np.std(mic_rec[-SR:]) + 1e-9) / (np.std(mic_rec[lead:lead + SR]) + 1e-9))), 1),
                    'compute_ms_mean_p99': [round(float(d.mean()), 3), round(float(np.percentile(d, 99)), 3)]})
        print(json.dumps(res[-1]), flush=True)
    return res


def rerecord(ids, her_db=-10.0):
    out = {}
    for i in ids:
        x, notes, tonic = excerpt(i)
        x = (x / (np.abs(x).max() + 1e-9) * 10 ** (her_db / 20)).astype('float32')
        play = np.concatenate([np.zeros(int(0.3 * SR), 'float32'), x, np.zeros(int(0.8 * SR), 'float32')])
        y = sd.playrec(play, samplerate=SR, channels=1, blocksize=B, latency=LAT, device=(2, 3), dtype='float32')[:, 0]
        sd.wait()
        c = ss.correlate(y, play, mode='full', method='fft')[len(play) - 1:]
        lag = int(np.argmax(np.abs(c[:int(1.0 * SR)])))
        y = y[lag:lag + len(play)]
        p = HERE / f'rerec_v{i}.wav'
        sf.write(p, y, SR)
        sf.write(HERE / f'orig_v{i}.wav', play, SR)
        out[i] = {'file': str(p), 'lag_ms': round(1000 * lag / SR, 1),
                  'peak_dbfs': round(float(20 * np.log10(np.abs(y).max() + 1e-12)), 1)}
        print(i, out[i], flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('roundtrip', 'fx', 'rerecord'))
    ap.add_argument('--ids', default='37,2')
    a = ap.parse_args()
    ids = [int(s) for s in a.ids.split(',')]
    r = {'roundtrip': roundtrip, 'fx': lambda: fx_live(ids), 'rerecord': lambda: rerecord(ids)}[a.cmd]()
    stamp = time.strftime('%H%M%S')
    json.dump(r, open(HERE / f'{a.cmd}_{stamp}.json', 'w'), indent=1, default=float)
    print(json.dumps(r, default=float)[:2000])
