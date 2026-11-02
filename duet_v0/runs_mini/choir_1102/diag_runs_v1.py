"""What was really sung in Vincent's choir runs (09:53:41 and 09:54:17, AirPods Max microphone), against what the live
tracker heard. Offline pyin (non-causal, robust) on run.wav's left channel (the microphone), notes = steady voiced runs;
then the live events from notes.json. Also the level and the spectrum's upper edge (Bluetooth hands-free bandwidth).

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/diag_runs_v1.py
"""
import json
import os

import librosa
import numpy as np
import soundfile as sf

ABS = '/Users/san/claude_sandbox/music/duet_v0'
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def nn(m):
    m = int(round(m))
    return f'{NAMES[m % 12]}{m // 12 - 1}'


for run in ('095341_r2_mix_v2', '095417_r2_mix_v2'):
    d = json.load(open(os.path.join(ABS, 'runs_mini/live', run, 'notes.json')))
    x, sr = sf.read(os.path.join(ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)
    mic = x[:, 0]
    rms = librosa.feature.rms(y=mic, frame_length=2048, hop_length=240)[0]
    S = np.abs(np.fft.rfft(mic[: sr * 20] * np.hanning(min(len(mic), sr * 20))))
    f = np.fft.rfftfreq(min(len(mic), sr * 20), 1 / sr)
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    print(f'=== {run}: {len(mic) / sr:.1f} s, mic peak {np.max(np.abs(mic)):.3f}, rms p90 {np.percentile(rms, 90):.4f}, '
          f'99 % of energy below {f[np.searchsorted(cum, 0.99)]:.0f} Hz, Li told {d["Li"]:.3f}')
    f0, vf, vp = librosa.pyin(mic, fmin=60, fmax=800, sr=sr, frame_length=2048, hop_length=240)
    t = librosa.times_like(f0, sr=sr, hop_length=240)
    midi = np.where(vf, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    notes, cur = [], []
    for i, m in enumerate(midi):
        if np.isnan(m):
            if len(cur) >= 5:
                notes.append(cur)
            cur = []
            continue
        if cur and abs(m - np.median(midi[cur[-6:]])) > 0.8:
            if len(cur) >= 5:
                notes.append(cur)
            cur = []
        cur.append(i)
    if len(cur) >= 5:
        notes.append(cur)
    print('pyin notes (start s, dur s, pitch, cents, spread of the note in semitones):')
    for n in notes:
        mm = midi[n]
        med = float(np.median(mm))
        print(f'  {t[n[0]]:6.2f} {t[n[-1]] - t[n[0]] + 0.01:5.2f}  {nn(med):4s} {100 * (med - round(med)):+4.0f}c  '
              f'wobble p10-p90 {np.percentile(mm, 90) - np.percentile(mm, 10):.2f} st')
    print('live tracker heard (engine time s, midi, song beat):')
    ev = d['events']
    T_b = 60 / d['bpm']
    t0c = d.get('click_t0') or d['t0']
    print('  ', [(round(t0c + e['onset_raw'] * T_b, 2), nn(e['her_heard'])) for e in ev[:40]])
