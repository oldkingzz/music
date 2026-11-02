"""Same engine as the live run (iter2, config v2, A major), offline: no MacBook latency, no speaker-to-mic loop.
Tells apart engine problems (choppy gate, wrong pitch) from live-path problems (late output, feedback)."""
import sys, subprocess
from pathlib import Path
import numpy as np, soundfile as sf, librosa
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'd5_live_fx' / 'iter2'))
import fx  # noqa: E402
fx.DEFAULTS.update({'drop_db': 16, 'onset_db': 26, 'f0_tau': 0.015, 'mode': 'follow'})
SR = 48000
def mp3(path, y):
    sf.write(path.with_suffix('.wav'), y, SR)
    subprocess.run(['/opt/homebrew/bin/lame', '--quiet', '-b', '96', str(path.with_suffix('.wav')), str(path)], check=True)
for name, wav in (('phone', 'phone_take_d1_song.wav'), ('clean', 'orig_v11.wav')):
    x, sr = sf.read(HERE / wav)
    x = x.mean(1) if x.ndim > 1 else x
    x = x / (np.abs(x).max() + 1e-9) * 0.5
    y, eng = fx.run_open(x, 9, sr=SR, block=128, aec=False)
    x = x[:len(y)]
    z, zy = librosa.resample(x, orig_sr=SR, target_sr=16000), librosa.resample(y, orig_sr=SR, target_sr=16000)
    f0x, vx, _ = librosa.pyin(z, fmin=70, fmax=1000, sr=16000, frame_length=1024, hop_length=160)
    f0y, vy, _ = librosa.pyin(zy, fmin=60, fmax=1000, sr=16000, frame_length=1024, hop_length=160)
    lvl = np.array([20 * np.log10(np.sqrt(np.mean(y[i:i + 480] ** 2)) + 1e-12) for i in range(0, len(y) - 480, 480)])
    L = min(len(lvl), len(f0x), len(f0y))
    lvl, f0x, vx, f0y = lvl[:L], f0x[:L], vx[:L], f0y[:L]
    on = lvl > -40
    sounding = on[vx].mean() if vx.any() else 0
    toggles = np.sum(np.abs(np.diff(on[vx].astype(int)))) / (vx.sum() * 0.01)
    both = vx & ~np.isnan(f0y) & ~np.isnan(f0x)
    iv = np.round(12 * np.log2(f0x[both] / f0y[both]))
    third = np.mean(np.isin(iv, [3, 4]))
    octave_err = np.mean(np.isin(np.abs(iv), [12, 15, 16, 19, 24]) | (iv < 0))
    print(f'{name}: Duet sounds during {100*sounding:.0f}% of her voiced time; on/off switches {toggles:.1f} per second of singing; '
          f'harmony a 3rd below her {100*third:.0f}% of co-voiced frames, octave/wrong-side {100*octave_err:.0f}%')
    mp3(HERE / f'fx_offline_{name}.mp3', np.clip(0.9 * x + 0.9 * y, -1, 1))
print('done')
