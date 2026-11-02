"""Offline transcription of futures/live_test/phone/test_her_v11_countin.wav (vocadito 11 with a count-in) for the
choir rehearsal mode: librosa pyin over the whole file (non-causal), notes = steady voiced runs, onsets on the
122.45 bpm grid from her first note. Prints notes and a pitch-class histogram; writes v11_notes_v1.json.

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/v11_transcribe_v1.py
"""
import json
import os

import librosa
import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
WAV = os.path.join(ABS, 'futures/live_test/phone/test_her_v11_countin.wav')
OUT = os.path.join(ABS, 'runs_mini/choir_1102/v11_notes_v1.json')
BPM = 122.45
HOP = 240

y, sr = librosa.load(WAV, sr=24000, mono=True)
f0, vflag, vprob = librosa.pyin(y, fmin=70, fmax=700, sr=sr, frame_length=2048, hop_length=HOP)
t = librosa.times_like(f0, sr=sr, hop_length=HOP)
midi = np.where(vflag, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)

# segment: a note = consecutive voiced frames whose pitch stays within 0.7 st of the running median
notes, cur = [], []
for i, m in enumerate(midi):
    if np.isnan(m):
        if len(cur) >= 4:
            notes.append(cur)
        cur = []
        continue
    if cur and abs(m - np.median(midi[cur[-6:]])) > 0.7:
        if len(cur) >= 4:
            notes.append(cur)
        cur = []
    cur.append(i)
if len(cur) >= 4:
    notes.append(cur)
rows = [{'on': float(t[n[0]]), 'off': float(t[n[-1]] + HOP / sr), 'midi_f': float(np.median(midi[n]))} for n in notes]
rows = [r for r in rows if r['off'] - r['on'] >= 0.06 and r['on'] > 2.5]   # skip the count-in clicks
T_b = 60.0 / BPM
t0 = rows[0]['on']
for r in rows:
    r['beat'] = round((r['on'] - t0) / T_b, 3)
    r['beat_q'] = round(round(r['beat'] * 4) / 4, 2)
    r['dur_b'] = round((r['off'] - r['on']) / T_b, 2)
    r['midi'] = int(round(r['midi_f']))
    r['cents'] = round(100 * (r['midi_f'] - r['midi']))
hist = np.zeros(12)
for r in rows:
    hist[r['midi'] % 12] += r['off'] - r['on']
names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
print('t0', round(t0, 3), 'notes', len(rows))
for r in rows:
    print(f"{r['beat']:7.2f} q{r['beat_q']:6.2f} dur {r['dur_b']:5.2f}  {names[r['midi'] % 12]}{r['midi'] // 12 - 1} "
          f"({r['midi']}) {r['cents']:+4d}c")
print('pc time (s):', {names[k]: round(float(hist[k]), 2) for k in np.argsort(-hist)})
# global tuning offset (she may be sharp/flat overall)
print('median cents', float(np.median([100 * (r['midi_f'] - round(r['midi_f'])) for r in rows])))
with open(OUT, 'x') as fh:
    json.dump({'t0': t0, 'bpm': BPM, 'notes': rows}, fh, indent=1)
