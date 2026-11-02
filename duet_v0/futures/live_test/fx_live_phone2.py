"""D5 live on the MacBook, v2 of the test: same as fx_live_phone.py, but every second the mic, the engine output
and the gate state are appended to disk, so an interrupted run still leaves its recording; audio glitches
(xruns) are counted per second. Usage: fx_live_phone2.py TONIC SECONDS [OUT_GAIN_DB]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'd5_live_fx' / 'iter2'))
import fx  # noqa: E402
fx.DEFAULTS.update({'drop_db': 16, 'onset_db': 26, 'f0_tau': 0.015, 'mode': 'follow'})   # iter2 chosen config v2

tonic, secs = int(sys.argv[1]), float(sys.argv[2])
gain = 10 ** ((float(sys.argv[3]) if len(sys.argv) > 3 else 0.0) / 20)
PRE = 10 ** ((float(sys.argv[4]) if len(sys.argv) > 4 else 0.0) / 20)   # input gain before the engine (quiet phone)
SR, B, LAT = 48000, 128, 0.005


def pick(kind, prefer):
    """Device index by name (indices shift when headphones connect). Never the iPhone microphone."""
    devs = sd.query_devices()
    for want in prefer:
        for i, d in enumerate(devs):
            ch = d['max_input_channels'] if kind == 'in' else d['max_output_channels']
            if ch > 0 and want in d['name'] and 'iPhone' not in d['name']:
                return i
    raise SystemExit(f'no {kind} device among {prefer}')


IN = pick('in', ['MacBook Air麦克风', 'MacBook Air Microphone'])
OUT = pick('out', ['外置耳机', 'External Headphones', 'AirPods', 'MacBook Air扬声器'])
DEV = (IN, OUT)
MONITOR = 0.8 if 'MacBook Air扬声器' not in sd.query_devices(OUT)['name'] else 0.0   # headphones: also hear her via the mic
print('input:', sd.query_devices(IN)['name'], '| output:', sd.query_devices(OUT)['name'], '| monitor', MONITOR, flush=True)
stamp = time.strftime('%H%M%S')
base = HERE / f'fx_live2_{stamp}'
eng = fx.Harmonizer(tonic=tonic, sr=SR, block=B, out_gain=gain, aec=(MONITOR == 0.0))   # headphones: no echo path
if MONITOR == 0.0:                                   # speaker: learn the echo path first (2 s soft noise)
    rng = np.random.default_rng(0)
    ref = (10 ** (-35 / 20) * rng.standard_normal(2 * SR)).astype('float32')
    mic = sd.playrec(ref, samplerate=SR, channels=1, blocksize=B, latency=LAT, device=DEV)[:, 0]
    sd.wait()
    print('sound check ERLE dB:', eng.calibrate(mic.astype(float), ref.astype(float)), flush=True)

n = int(secs * SR)
mic_rec, out_rec = np.zeros(n + B, np.float32), np.zeros(n + B, np.float32)
gate = np.zeros(n // B + 2, np.int8)
pos, xr, dts = [0], [], []


def cb(indata, outdata, frames, t, status):
    if status:
        xr.append(pos[0] / SR)
    t0 = time.perf_counter()
    xin = np.clip(PRE * indata[:, 0].astype(float), -1, 1)
    y = eng.process(xin)
    outdata[:, 0] = np.clip(y + MONITOR * xin, -1, 1)
    i = pos[0]
    if i + frames <= n:
        mic_rec[i:i + frames] = indata[:, 0]
        out_rec[i:i + frames] = y
        gate[i // B] = int(eng.gate_open)
    pos[0] += frames
    dts.append(time.perf_counter() - t0)


written = 0
with open(f'{base}_mic.f32', 'ab') as fm, open(f'{base}_out.f32', 'ab') as fo:
    with sd.Stream(samplerate=SR, blocksize=B, channels=1, latency=LAT, dtype='float32', callback=cb, device=DEV) as s:
        print('LIVE: play the clip now; stream latency in/out ms', [round(1000 * v, 1) for v in s.latency], flush=True)
        while pos[0] < n:
            sd.sleep(1000)
            p = min(pos[0], n)
            mic_rec[written:p].tofile(fm); out_rec[written:p].tofile(fo); fm.flush(); fo.flush()
            written = p
m, o = mic_rec[:n], out_rec[:n]
sf.write(f'{base}.wav', np.stack([m, o], 1), SR)
d = np.array(dts) * 1000
info = {'xruns': len(xr), 'xrun_times_s': [round(v, 2) for v in xr[:50]], 'compute_ms_mean': float(d.mean()),
        'compute_ms_p99': float(np.percentile(d, 99)), 'gate_open_share': float(gate[:n // B].mean())}
json.dump(info, open(f'{base}.json', 'w'), indent=1)
print(json.dumps(info), flush=True)
