"""v2 of ode_live_align_v1.py for run 211009 (the v3 file with parted notes, rated 差 again). Was it the Ode phone file, and where are its true onsets? The run's mic channel is aligned with
futures/live_test/phone/ode_C_96bpm_lead3s.wav by cross-correlating the two pitch tracks (pyin, 10 ms), so every
written note's true time in the run is known (no transcription). Then: which written notes the live tracker heard as
a new note (an onset in the follower's accepted list within 120 ms), repeated notes listed apart, and the follower's
beat map against the true beat times. Offline (tripwire).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py ode_live_align_v1 600 .venv/bin/python runs_mini/choir_1102/ode_live_align_v1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tonight_eval_v1 as T  # noqa: E402  (tripwire, replay, accept trace)
import live_panel_v15 as P15  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

E = T.E


def track(x):
    f0, vf, _ = librosa.pyin(x, fmin=70, fmax=800, sr=E.SR, frame_length=2048, hop_length=240)
    return np.where(vf, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), 0.0)


if __name__ == '__main__':
    E.LI, E.LO = T.LI, T.LO
    E.P3.MODELS.load()
    x = sf.read(os.path.join(E.ABS, 'runs_mini/live/211009_r2_mix_v2/run.wav'), dtype='float32', always_2d=True)[0][:, 0]
    f = sf.read(os.path.join(E.ABS, 'futures/live_test/phone/ode_C_96bpm_lead3s_v3.wav'), dtype='float32', always_2d=True)[0][:, 0]
    a, b = track(x), track(f)
    best = None
    for k in range(-len(b) + 50, len(a) - 50):           # lag of the file inside the run, in 10 ms frames
        i0, j0 = max(0, k), max(0, -k)
        n = min(len(a) - i0, len(b) - j0)
        if n < 300:
            continue
        aa, bb = a[i0:i0 + n], b[j0:j0 + n]
        both = (aa > 0) & (bb > 0)
        if both.sum() < 200:
            continue
        score = float(np.mean(np.abs(aa[both] - bb[both]) < 0.5)) * both.sum()
        if best is None or score > best[0]:
            best = (score, k, float(np.mean(np.abs(aa[both] - bb[both]) < 0.5)), int(both.sum()))
    _, k, agree, nb = best
    lag = k * 0.01
    print(f'file starts {lag:+.2f} s into the run; pitch agrees on {agree:.0%} of {nb} frames both voiced', flush=True)
    ode = P15.P5.CHOIR['ode']
    sb = 60.0 / ode['bpm']
    true_t = {a0: lag + 3.0 + a0 * sb - E.LI for a0, _, _ in ode['melody']}   # file: 3 s silence, then 96 bpm
    T.TRACE.clear()
    eng, _ = E.replay(x, {})
    heard = [n['t'] for n in eng.hnotes]
    prev = None
    rows = []
    for a0, b0, m in ode['melody']:
        t = true_t[a0]
        hit = min((abs(h - t) for h in heard), default=9) < 0.12
        rows.append((a0, m, m == prev, hit))
        prev = m
    rep = [r for r in rows if r[2]]
    new = [r for r in rows if not r[2]]
    print(f'written notes heard as a new note: repeats {sum(r[3] for r in rep)}/{len(rep)}, '
          f'others {sum(r[3] for r in new)}/{len(new)}', flush=True)
    print('missed:', [(r[0], r[1], 'repeat' if r[2] else '') for r in rows if not r[3]][:30], flush=True)
    print('follower: beat, its time - true time (ms), bpm', flush=True)
    for bt, t, pred, bpm in T.TRACE:
        tt = true_t.get(bt)
        print('   %6.2f %8s %6.1f' % (bt, '-' if tt is None else round((t - tt) * 1000), bpm), flush=True)
    # level and voicing of the mic over time (1 s steps): where did the tracker lose her?
    ft = np.asarray(eng.ft_t)
    for s0 in range(0, int(len(x) / E.SR)):
        seg = x[s0 * E.SR:(s0 + 1) * E.SR]
        db = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        v = a[s0 * 100:(s0 + 1) * 100]
        hv = [n for n in eng.hnotes if s0 <= n['t'] + E.LI < s0 + 1]
        print('  %2d s  %5.1f dB  pyin voiced %3d%%  tracker notes %d' % (s0, db, int(np.mean(v > 0) * 100), len(hv)),
              flush=True)
    print('diag', {k: v for k, v in eng.diag.items() if not isinstance(v, list)}, flush=True)
