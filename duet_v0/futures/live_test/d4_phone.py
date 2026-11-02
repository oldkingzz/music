"""D4 style switch on a real recording: notes from D1's transcription of the audio, beat from estimate_beat,
key from the melody; renders Bach, pop, jazz back to back with her recording. Usage: d4_phone.py NAME WAV TIMING_JSON"""
import json, math, os, sys
import numpy as np
import soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'd4_styles'))
import styles as S  # noqa: E402
from live.rehearsal import estimate_beat  # noqa: E402
from harmony.model import detect_key_melody  # noqa: E402

name, wav, tj = sys.argv[1:4]
tn = json.load(open(tj))['notes']                       # [onset_s, dur_s, midi] from D1's transcription
notes = [{'on': float(a), 'off': float(a + d), 'midi': int(m)} for a, d, m in tn]
P = float(estimate_beat(notes, []))
ons = np.array([n['on'] for n in notes])
best = max(((float(np.mean(np.abs(((ons - ons[0] - ph) / (P / 2)) - np.round((ons - ons[0] - ph) / (P / 2))) < 0.24)), ph)
            for ph in np.linspace(0, P, 40, endpoint=False)))
t0 = ons[0] + best[1]
while t0 - P > -1e-6:
    t0 -= P
K = int(math.ceil((notes[-1]['off'] - t0) / P)) + 1
t, mode = detect_key_melody(notes)
x, sr = sf.read(wav, dtype='float32')
x = x.mean(1) if x.ndim > 1 else x
if sr != S.SR:
    import librosa
    x = librosa.resample(x, orig_sr=sr, target_sr=S.SR)
mel = S.Melody(name, [(n['on'], n['off'], n['midi']) for n in notes], t0, P, 4, [k % 4 for k in range(K)], t, mode, x,
               'recorded on the MacBook mic', 'local')
print(f'{name}: {len(notes)} notes, beat {P:.3f} s ({60/P:.0f} BPM), grid fit {best[0]:.2f}, key tonic {t} {mode}')
seq, her, gap = [], None, np.zeros(int(0.8 * S.SR), np.float32)
for style, fn in (('bach', S.style_bach), ('pop', S.style_pop), ('jazz', S.style_jazz)):
    o = fn(mel)
    h, a = S.no_clash(mel.notes, o['events'], 'harsh')['all'], S.no_clash(mel.notes, o['events'], 'avoid')['all']
    y, her = S.render_style(mel, style, o, her)
    seq += [y, gap]
    print(f'  {style}: no-clash vs the transcribed notes harsh {h:.3f} avoid {a:.3f}; chords {o["chords"][:12]}')
out = os.path.join(HERE, 'd4_out', f'{name}_bach_pop_jazz.mp3')
os.makedirs(os.path.dirname(out), exist_ok=True)
print('bytes', S.to_mp3(np.concatenate(seq), out, 96), out)
