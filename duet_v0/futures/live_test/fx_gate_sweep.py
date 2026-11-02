"""Why is the live harmony choppy? Sweep the engine's gate settings offline on the clean clip and the phone take."""
import sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'd5_live_fx' / 'iter2'))
import fx  # noqa: E402
SR = 48000
V2 = {'drop_db': 16, 'onset_db': 26, 'f0_tau': 0.015, 'mode': 'follow'}
clips = {}
for name, wav in (('phone', 'phone_take_d1_song.wav'), ('clean', 'orig_v11.wav')):
    x, _ = sf.read(HERE / wav); x = x / (np.abs(x).max() + 1e-9) * 0.5
    z = librosa.resample(x, orig_sr=SR, target_sr=16000)
    f0, v, _ = librosa.pyin(z, fmin=70, fmax=1000, sr=16000, frame_length=1024, hop_length=160)
    clips[name] = (x, v)
def measure(cfg):
    fx.DEFAULTS.clear(); fx.DEFAULTS.update({**V2, **cfg}); out = {}
    for name, (x, v) in clips.items():
        y, _ = fx.run_open(x, 9, sr=SR, block=128, aec=False)
        lvl = np.array([20 * np.log10(np.sqrt(np.mean(y[i:i + 480] ** 2)) + 1e-12) for i in range(0, len(y) - 480, 480)])
        L = min(len(lvl), len(v)); on, vv = lvl[:L] > -40, v[:L]
        out[name] = (on[vv].mean(), np.sum(np.abs(np.diff(on[vv].astype(int)))) / (vv.sum() * 0.01), on[~vv].mean())
    return out
cfgs = {'v2 (live run)': {}, 'unv_close 40': {'unv_close': 40}, 'unv_close 40 + release 0.15': {'unv_close': 40, 'release_s': 0.15},
        'unv_close 60 + release 0.15 + hyst_blocks 6': {'unv_close': 60, 'release_s': 0.15, 'hyst_blocks': 6},
        'ap_stay 0.6 + unv_close 60 + release 0.15': {'ap_stay': 0.6, 'unv_close': 60, 'release_s': 0.15},
        'drop off + onset 30 + unv 60 + rel 0.15': {'drop_db': None, 'onset_db': 30, 'unv_close': 60, 'release_s': 0.15}}
for k, c in cfgs.items():
    try:
        r = measure(c)
        print(f'{k:45s} ' + '  '.join(f'{n}: sounds {100*a:.0f}% of her singing, {b:.1f} switches/s, on while she is unvoiced {100*u:.0f}%' for n, (a, b, u) in r.items()), flush=True)
    except Exception as e:
        print(k, 'error', e)
