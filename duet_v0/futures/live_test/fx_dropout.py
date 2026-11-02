import sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'd5_live_fx' / 'iter2'))
import fx  # noqa: E402
SR = 48000
fx.DEFAULTS.update({'drop_db': 16, 'onset_db': 26, 'f0_tau': 0.015, 'mode': 'follow'})
env = lambda s: np.array([20 * np.log10(np.sqrt(np.mean(s[i:i + 480] ** 2)) + 1e-12) for i in range(0, len(s) - 480, 480)])  # noqa: E731
for name, wav in (('phone', 'phone_take_d1_song.wav'), ('clean', 'orig_v11.wav')):
    x, _ = sf.read(HERE / wav); x = x / (np.abs(x).max() + 1e-9) * 0.5
    y, eng = fx.run_open(x, 9, sr=SR, block=128, aec=False)
    ex, ey = env(x[:len(y)]), env(y)
    z = librosa.resample(x, orig_sr=SR, target_sr=16000)
    f0, v, _ = librosa.pyin(z, fmin=70, fmax=1000, sr=16000, frame_length=1024, hop_length=160)
    L = min(len(ex), len(ey), len(v)); ex, ey, v = ex[:L], ey[:L], v[:L]
    loud = v & (ex > np.percentile(ex[v], 25))                    # she is clearly singing (top 75% of her voiced frames)
    drop = loud & (ey < ex - 25)                                     # Duet more than 25 dB under her = gap
    runs = np.diff(np.concatenate([[0], drop.astype(int), [0]]))
    starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
    lens = (ends - starts) * 10
    print(f'{name}: she clearly sings {loud.sum()/100:.1f} s; Duet drops out {100*drop.sum()/max(loud.sum(),1):.0f}% of that, '
          f'{len(lens)} gaps (median {np.median(lens) if len(lens) else 0:.0f} ms, longest {lens.max() if len(lens) else 0} ms); '
          f'Duet level vs hers when sounding: {np.median((ey - ex)[loud & ~drop]):.1f} dB')
    for a, b in list(zip(starts, ends))[:12]:
        print(f'   gap {a/100:.2f}-{b/100:.2f} s, her pitch {np.nanmedian(f0[a:b]) if np.any(~np.isnan(f0[a:b])) else float("nan"):.0f} Hz')
