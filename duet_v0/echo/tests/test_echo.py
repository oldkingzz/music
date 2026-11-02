"""Checks for echo/sim.py and echo/guard.py. Plain asserts, no pytest needed:

    .venv/bin/python echo/tests/test_echo.py

(Also pytest-compatible: every check is a test_* function.) Takes ~1 minute.
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import guard as G  # noqa: E402
import sim  # noqa: E402

from dsp import HOP, SR, colored_noise, hz_to_midi, render_voice, yin  # noqa: E402
from follow import FeatureTracker  # noqa: E402

CTX = sim.synthetic_context()


def test_closed_loop_without_bleed_reproduces_run_v0():
    # Golden values = run_v0.py with the shipped follow.py (follower v2 + amendment A2): pred median_abs_ms /
    # p90_abs_ms of outputs/synthetic/{steady,natural}/metrics.json, re-run 2026-10-16 (sha256 of follow.py
    # 8a3bcdf5...). The v1 values from git b3468fe were steady 5.63 / 16.47 and natural 10.84 / 37.50 ms.
    # The test checks that echo/sim.py's closed loop (no bleed) IS the run_v0 loop, so it follows the
    # shipped follower; update these if follow.py changes.
    for scen, med, p90 in (('steady', 6.5917054359996285, 13.713041857648989),
                           ('natural', 20.837070877155917, 42.483825400290165)):
        m, _ = sim.run(CTX, scen, None)
        assert m['played'] == 26 and m['missed'] == 0, m
        assert abs(m['median_abs_ms'] - med) < 0.5 and abs(m['p90_abs_ms'] - p90) < 1.0, (scen, m['median_abs_ms'])


def test_streamed_voice_matches_offline_render():
    m, ex = sim.run(CTX, 'natural', -30, keep_audio=True)
    off = render_voice([(s, d, k) for s, d, k, _ in ex['acc'].machine_notes('pred')], len(ex['spk']))
    c = np.corrcoef(off, ex['spk'])[0, 1]
    assert c > 0.995, c


def test_bleed_calibration():
    m, _ = sim.run(CTX, 'natural', -12)
    assert abs(m['bleed_db_measured'] - (-12)) < 3.0, m['bleed_db_measured']


def test_room_ir_physics():
    h, info = sim.room_ir(acoustic_ms=2.0, rt60=0.3)
    assert info['delay_ms'] == 22.0
    k = int(0.022 * SR)
    assert abs(int(np.argmax(np.abs(h))) - k) <= 3
    # energy decay: 10 dB of extra decay per RT60/6 = 50 ms in the tail
    e = np.cumsum(h[::-1] ** 2)[::-1]
    i1, i2 = k + int(0.05 * SR), k + int(0.15 * SR)
    slope = 10 * np.log10(e[i2] / e[i1]) / 0.1          # dB per second (Schroeder integral)
    assert -260 < slope < -140, slope                    # -200 dB/s for RT60 = 0.3 s


def test_aec_converges_on_white_noise():
    rng = np.random.default_rng(1)
    x = 0.1 * rng.standard_normal(4 * SR)
    h, _ = sim.room_ir()
    d = np.convolve(x, h)[:len(x)] * 0.05
    a = G.EchoCanceller(mu=0.8, reg=0.01, two_path=False)
    e = np.concatenate([a.process(d[i:i + HOP], x[i:i + HOP]) for i in range(0, len(x), HOP)])
    erle = 10 * np.log10(np.sum(d[-SR:] ** 2) / np.sum(e[-SR:] ** 2))
    assert erle > 25, erle
    assert abs(a.delay_samples() - 528) <= 2


def test_soundcheck_aec_survives_double_talk():
    h, _ = sim.room_ir()
    g = sim.echo_gain(h, 0.0, CTX['singer_rms'])
    rng = np.random.default_rng(2)
    notes, t = [], 0.2
    while t < 12:
        d = rng.uniform(0.3, 0.8)
        notes.append((t, d, int(rng.integers(55, 70))))
        t += d + 0.02
    ref = render_voice(notes, 12 * SR)
    echo = g * sim.EchoPath(h).step(ref)
    sing = CTX['x'][:len(ref)].astype(float)
    ref_sc = sim.SOUNDCHECK_RMS * colored_noise(2 * SR, rng, -0.5)
    gd = G.EchoGuard(aec=True, mask=False)
    sc = gd.calibrate(g * sim.EchoPath(h).step(ref_sc), ref_sc)
    assert sc > 20, sc
    mic = echo + sing
    o = np.concatenate([gd.cancel(mic[i:i + HOP], ref[i:i + HOP]) for i in range(0, len(mic), HOP)])
    erle = 10 * np.log10(np.sum(echo ** 2) / np.sum((o - sing) ** 2))
    assert erle > 20, erle


def test_comb_recovers_singer_pitch_under_harmony():
    t = np.arange(4000) / SR
    her = sum(np.sin(2 * np.pi * 440 * k * t) / k for k in range(1, 6))
    mach = 5 * render_voice([(0, 0.2, 65)], 4000)
    mix = her + mach
    assert abs(hz_to_midi(yin(mix[-960:])[0]) - 69) > 12          # plain YIN: sub-harmonic
    z = G.comb_remove(mix, SR / 349.228)
    f0, ap, _ = yin(z[-960:])
    assert abs(hz_to_midi(f0) - 69) < 0.2 and ap < 0.1


def test_guard_is_transparent_without_machine_voice():
    rng = np.random.default_rng(3)
    x = 0.2 * rng.standard_normal(50 * HOP)
    gd = G.EchoGuard(aec=True, mask=True)
    ft_a, ft_b = FeatureTracker(), FeatureTracker()
    for i in range(50):
        c = x[i * HOP:(i + 1) * HOP]
        ft_a.step(c)
        ft_b.step(gd.cancel(c, np.zeros(HOP)))
        gd.mask(ft_b, [60])
    assert np.allclose(np.nan_to_num(ft_a.m), np.nan_to_num(ft_b.m)) and ft_a.v == ft_b.v


def test_guard_does_not_hurt_without_bleed():
    base, _ = sim.run(CTX, 'natural', None)
    m, _ = sim.run(CTX, 'natural', None, guard=G.EchoGuard(), soundcheck=2.0)
    assert m['played'] == base['played'] and abs(m['median_abs_ms'] - base['median_abs_ms']) < 2.0


def test_delay_estimate():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(SR)
    h, _ = sim.room_ir(acoustic_ms=3.0)
    assert abs(G.estimate_delay(np.convolve(x, h)[:SR], x) - 552) <= 2


def test_guard_cost_fits_real_time():
    gd = G.EchoGuard()
    ft = FeatureTracker()
    rng = np.random.default_rng(5)
    ref = render_voice([(0.0, 2.0, 62)], 2 * SR)
    mic = 0.1 * rng.standard_normal(2 * SR) + ref
    t0 = time.perf_counter()
    for i in range(0, 2 * SR, HOP):
        ft.step(gd.cancel(mic[i:i + HOP], ref[i:i + HOP]))
        gd.mask(ft, [66])
    per_frame_ms = (time.perf_counter() - t0) * 1000 / (2 * SR / HOP)
    assert per_frame_ms < 3.0, per_frame_ms          # budget: 10 ms per frame for everything


if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f'PASS {name}  ({time.time() - t0:.1f} s)')
            except AssertionError as e:
                fails += 1
                print(f'FAIL {name}: {e}')
    print('all passed' if not fails else f'{fails} failed')
    sys.exit(1 if fails else 0)
