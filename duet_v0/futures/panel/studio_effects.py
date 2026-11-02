"""Offline effects for the Duet Studio page, one subcommand each (run in a subprocess by studio.py).

  styles WAV TIMING_JSON OUTDIR   Bach, pop and jazz arrangements (D4), one mp3 each, plus styles.json
  fx     WAV TONIC OUT_MP3        the live harmonizer (D5, iter2 settings) run on the recording
  duet   WAV TIMING_JSON OUT_MP3  the original beat-by-beat Duet with the leaderboard policy, run on the recording

TIMING_JSON is D1's timing.json for the same recording (transcribed notes and key).
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
DUET = os.path.dirname(os.path.dirname(HERE))
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
LAME = '/opt/homebrew/bin/lame'


def mp3(path, y, sr):
    wav = path[:-4] + '.wav'
    sf.write(wav, np.clip(y, -1, 1).astype(np.float32), sr)
    subprocess.run([LAME, '--quiet', '-b', '128', wav, path], check=True)


def key_of(timing):
    k = timing.get('key', 'C major').split()
    t = NAMES.index(k[0]) if k[0] in NAMES else 0
    mode = k[1] if len(k) > 1 else 'major'
    return t, mode


def beat_grid(notes):
    """Tempo and the first beat from her transcribed notes (D4's method: estimate_beat + best phase)."""
    sys.path.insert(0, DUET)
    from live.rehearsal import estimate_beat
    nd = [{'on': float(a), 'off': float(a + d), 'midi': int(m)} for a, d, m in notes]
    P = float(estimate_beat(nd, []))
    ons = np.array([n['on'] for n in nd])
    fit, ph = max(((float(np.mean(np.abs(((ons - ons[0] - p) / (P / 2)) - np.round((ons - ons[0] - p) / (P / 2))) < 0.24)), p)
                   for p in np.linspace(0, P, 40, endpoint=False)))
    t0 = ons[0] + ph
    while t0 - P > -1e-6:
        t0 -= P
    return P, t0, fit, nd


def cmd_styles(wav, timing_path, outdir):
    sys.path.insert(0, os.path.join(DUET, 'futures', 'd4_styles'))
    import styles as S
    from harmony.model import detect_key_melody
    timing = json.load(open(timing_path))
    P, t0, fit, nd = beat_grid(timing['notes'])
    K = int(math.ceil((nd[-1]['off'] - t0) / P)) + 1
    t, mode = detect_key_melody(nd)
    x, sr = sf.read(wav, dtype='float32')
    x = x.mean(1) if x.ndim > 1 else x
    if sr != S.SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=S.SR)
    mel = S.Melody('studio', [(n['on'], n['off'], n['midi']) for n in nd], t0, P, 4, [k % 4 for k in range(K)], t, mode, x,
                   'recording', 'local')
    her, info = None, {'bpm': round(60 / P, 1), 'grid_fit': round(fit, 2), 'key': f'{NAMES[t]} {mode}'}
    for style, fn in (('bach', S.style_bach), ('pop', S.style_pop), ('jazz', S.style_jazz)):
        o = fn(mel)
        y, her = S.render_style(mel, style, o, her)
        mp3(os.path.join(outdir, f'{style}.mp3'), y, S.SR)
        info[style] = {'no_clash': round(S.no_clash(mel.notes, o['events'], 'avoid' if style != 'bach' else 'harsh')['all'], 3),
                       'chords': [c.split(':')[0] if isinstance(c, str) else c for c in o['chords'][:16]]}
    json.dump(info, open(os.path.join(outdir, 'styles.json'), 'w'), indent=1, default=str)


def cmd_fx(wav, tonic, out):
    sys.path.insert(0, os.path.join(DUET, 'futures', 'd5_live_fx', 'iter2'))
    import fx
    fx.DEFAULTS.update({'drop_db': 16, 'onset_db': 26, 'f0_tau': 0.015, 'mode': 'follow'})
    x, sr = sf.read(wav, dtype='float32')
    x = x.mean(1) if x.ndim > 1 else x
    if sr != 48000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=48000)
        sr = 48000
    x = x / (np.abs(x).max() + 1e-9) * 0.5
    y, _ = fx.run_open(x.astype(float), int(tonic), sr=sr, block=128, aec=False)
    mp3(out, 0.9 * x[:len(y)] + 0.9 * y, sr)


def cmd_duet(wav, timing_path, out):
    for p in (os.path.join(DUET, 'live'), DUET, os.path.join(DUET, 'duet_beat'), HERE):
        sys.path.insert(0, p)
    from dataclasses import replace
    import live_mic as lm
    import policy_live as PL
    from planner import Config
    timing = json.load(open(timing_path))
    P, t0, fit, nd = beat_grid(timing['notes'])
    tonic, mode = key_of(timing)
    if mode == 'minor':
        tonic = (tonic + 3) % 12                       # Duet's chords are those of the relative major
    res = json.load(open(os.path.join(DUET, 'duet_beat', 'out', 'results.json')))
    base = dict(res['tuning']['chosen']['main'])
    base['alto'] = tuple(base['alto'])
    x, sr = sf.read(wav, dtype='float32')
    x = x.mean(1) if x.ndim > 1 else x
    if sr != lm.SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=lm.SR)
    K = int(math.ceil((len(x) / lm.SR - t0) / P))
    cfg = replace(Config(**base), bpm=60.0 / P, mode='long', J=2)
    eng = PL.PolicyDuet(PL.load_policy(), lm.load_model('bach'), cfg, tonic=tonic, M=4, K=max(K, 4), count_in=0,
                        Li=0.0, Lo=0.0, monitor=False, log=lambda s: None)
    eng.t0, eng.T_b = t0, P
    eng._click = lambda n: np.zeros(n)
    ys = []
    xx = np.concatenate([x, np.zeros(lm.SR)])
    for i in range(0, len(xx) - lm.HOP, lm.HOP):
        ys.append(eng.process(xx[i:i + lm.HOP].astype(float)))
    y = np.concatenate(ys)
    her = xx[:len(y)]
    her = her / (np.sqrt(np.mean(her[np.abs(her) > 1e-4] ** 2)) + 1e-9) * 0.1
    yy = y / (np.sqrt(np.mean(y[np.abs(y) > 1e-4] ** 2)) + 1e-9) * 0.08 if np.any(np.abs(y) > 1e-4) else y
    mix = her + yy
    mp3(out, mix * 0.89 / (np.abs(mix).max() + 1e-9), lm.SR)
    sung = [b for b in eng.beats if b['h'] >= 0]
    json.dump({'bpm': round(60 / P, 1), 'grid_fit': round(fit, 2), 'key': f'{NAMES[tonic]} major', 'beats': len(eng.beats),
               'sung': len(sung)}, open(out[:-4] + '.json', 'w'))


if __name__ == '__main__':
    c = sys.argv[1]
    {'styles': cmd_styles, 'fx': cmd_fx, 'duet': cmd_duet}[c](*sys.argv[2:])
