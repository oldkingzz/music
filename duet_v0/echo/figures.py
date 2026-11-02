"""Mechanism figures for the slides (independent of the big sweep).

    .venv/bin/python echo/figures.py

1. figures/why_it_fails.png  - what the pitch tracker reports with the machine at -6 dB:
   YIN locks onto the common sub-harmonic of the two voices (a major third 5:4 has a common
   period two octaves below her), the follower jumps, and the duet collapses.
2. figures/aec_erle.png      - echo-return-loss enhancement over a song: cold start vs a 2 s
   sound check, in double talk (she sings the whole time), and after the echo path changes.
Also writes results/aec_erle.json and results/why_it_fails.json with the numbers shown.
"""
import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guard as G  # noqa: E402
import sim  # noqa: E402
from sweep import perturbed  # noqa: E402

from dsp import HOP, SR, colored_noise  # noqa: E402
from follow import FeatureTracker  # noqa: E402

INK, INK2, GRID = '#1d1d1f', '#6e6e73', '#e5e5ea'
BLUE, ORANGE, AQUA, GRAY, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#8a8985', '#eda100'


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=11, length=0)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def track(x, n):
    ft = FeatureTracker()
    for k in range(n):
        ft.step(x[k * HOP:(k + 1) * HOP])
    return ft


def why_it_fails(ctx, h, out):
    L = -6
    m0, ex0 = sim.run(ctx, 'natural', L, ir=h, keep_audio=True)
    g = G.EchoGuard(aec=False, mask=True)
    m1, ex1 = sim.run(ctx, 'natural', L, guard=g, ir=h, keep_audio=True, soundcheck=2.0)
    n = len(ex0['ft'].m)
    clean = track(ex0['perf'].audio, n)
    t = np.array([ex0['ft'].t(i) for i in range(n)])
    raw0 = np.array(ex0['ft'].raw)
    truth = np.where(clean.v, np.array(clean.raw), np.nan)
    machine = track(ex0['spk'], n)
    mach = np.where(machine.v, np.array(machine.raw), np.nan)
    jump = np.abs(np.diff(mach, prepend=mach[0])) > 2                 # hide note-change blips of the plot line
    mach_plot = np.where(jump | np.roll(jump, -1), np.nan, mach)
    sel = (t >= 10.5) & (t <= 16.5)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    style(ax)
    ax.plot(t[sel], truth[sel], color=INK, lw=2.4, label='Her voice (truth)')
    ax.plot(t[sel], mach_plot[sel], color=GRAY, lw=2.4, label='Machine voice (harmony)')
    ax.plot(t[sel], raw0[sel], '.', color=ORANGE, ms=5, label='What the tracker hears, bleed −6 dB')
    sub = sel & (raw0 < 50)
    ax.annotate('her A4 + machine F4 (a 5:4 third) share one period:\nthe tracker reports F2, two octaves down',
                xy=(12.25, 41.2), xytext=(13.3, 50),
                color=INK, fontsize=11, arrowprops=dict(arrowstyle='->', color=INK2))
    jumps = [a for a in ex0['acc'].anchors if a[0] > 10.5 and abs(a[1] - float(ex0['perf'].live_to_ref(a[0])[0])) > 0.3]
    if jumps:
        tj = jumps[0][0]
        ax.axvline(tj, color=ORANGE, lw=1.2, ls='--')
        ax.text(tj + 0.08, 76.5, 'follower jumps\nto the wrong bar', color=ORANGE, fontsize=11, va='top')
    ax.set_ylim(36, 79)
    ax.set_xlabel('Time, s', color=INK, fontsize=12)
    ax.set_ylabel('Pitch, MIDI note', color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
    fig.savefig(out / 'why_it_fails.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    both = np.array(ex0['ft'].v) & np.array(clean.v) & ~np.isnan(mach)
    wrong = both & (np.abs(raw0 - np.array(clean.raw)) > 0.5)
    raw1 = np.array(ex1['ft'].raw)
    both1 = np.array(ex1['ft'].v) & np.array(clean.v) & ~np.isnan(mach)
    wrong1 = both1 & (np.abs(raw1 - np.array(clean.raw)) > 0.5)
    return {'bleed_db': L,
            'frames_both_voiced': int(both.sum()), 'frames_wrong_pitch_no_guard': int(wrong.sum()),
            'frames_wrong_pitch_below_her_by_12plus': int((wrong & (raw0 < np.array(clean.raw) - 11)).sum()),
            'frames_wrong_pitch_with_mask': int(wrong1.sum()), 'frames_both_voiced_with_mask': int(both1.sum()),
            'no_guard': {k: m0[k] for k in ('played', 'missed', 'held', 'resets', 'tracking_within100ms', 'median_abs_ms')},
            'mask': {k: m1[k] for k in ('played', 'missed', 'held', 'resets', 'tracking_within100ms', 'median_abs_ms')},
            'mask_stats': m1.get('guard')}


def erle_curve(echo, res, win=0.5):
    n = int(win * SR)
    out = []
    for i in range(0, len(echo) - n, n):
        e = np.sum(echo[i:i + n] ** 2)
        r = np.sum(res[i:i + n] ** 2)
        out.append(10 * np.log10(e / r) if e > 1e-9 and r > 0 else np.nan)
    return np.arange(len(out)) * win + win / 2, np.array(out)


def aec_erle(ctx, h, out):
    # machine voice and her voice from a clean (no-bleed) natural performance
    _, ex = sim.run(ctx, 'natural', None, ir=h, keep_audio=True)
    ref = ex['spk']
    sing = ex['perf'].audio[:len(ref)].astype(float)
    g = sim.echo_gain(h, 0.0, ctx['singer_rms'])
    echo = g * sim.EchoPath(h).step(ref)
    h2 = perturbed(h, seed=101)
    k = int(0.6 * len(ref))
    echo2 = echo.copy()
    echo2[k:] = (g * sim.EchoPath(h2).step(ref))[k:]
    rng = np.random.default_rng(11)
    ns = 2 * SR
    ref_sc = sim.SOUNDCHECK_RMS * colored_noise(ns, rng, -0.5)
    mic_sc = g * sim.EchoPath(h).step(ref_sc) + ctx['singer_rms'] * 1e-3 * colored_noise(ns, rng, -0.5)

    def run(guard, mic):
        o = np.zeros(len(mic) // HOP * HOP)
        for i in range(0, len(o), HOP):
            o[i:i + HOP] = guard.cancel(mic[i:i + HOP], ref[i:i + HOP])
        return o

    cases = {}
    cold = G.EchoGuard(aec=True, mask=False)
    warm = G.EchoGuard(aec=True, mask=False)
    sc_erle = warm.calibrate(mic_sc, ref_sc)
    n = len(ref) // HOP * HOP
    for name, gd, e in (('cold start, she sings throughout', cold, echo),
                        ('2 s sound check, she sings throughout', copy.deepcopy(warm), echo),
                        ('sound check, then the echo path changes', copy.deepcopy(warm), echo2)):
        mic = e + sing
        o = run(gd, mic)
        tt, er = erle_curve(e[:n], o - sing[:n])
        active = np.array([np.sum(ref[int(t0 * SR - 0.25 * SR):int(t0 * SR + 0.25 * SR)] ** 2) > 1e-3 for t0 in tt])
        er[~active] = np.nan
        cases[name] = (tt, er)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    style(ax)
    cols = [YELLOW, BLUE, ORANGE]
    t_join = ctx['join_r']
    for (name, (tt, er)), c in zip(cases.items(), cols):
        ax.plot(tt, er, color=c, lw=2.4, label=name)
    ax.axvline(k / SR, color=INK2, lw=1, ls=':')
    ax.text(k / SR + 0.2, 33, 'path changes', color=INK2, fontsize=10)
    ax.set_xlim(t_join - 0.5, len(ref) / SR)
    ax.set_ylim(-15, 42)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel('Time in the song, s (machine joins at %.0f s)' % t_join, color=INK, fontsize=12)
    ax.set_ylabel('Echo removed (ERLE), dB', color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
    fig.savefig(out / 'aec_erle.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    summ = {name: {'median_erle_db': float(np.nanmedian(er)), 'p10_erle_db': float(np.nanpercentile(er, 10))}
            for name, (tt, er) in cases.items()}
    tt, er = cases['sound check, then the echo path changes']
    summ['sound check, then the echo path changes']['median_before_change_db'] = float(np.nanmedian(er[tt < k / SR]))
    summ['sound check, then the echo path changes']['median_after_change_db'] = float(np.nanmedian(er[tt > k / SR + 0.5]))
    summ['soundcheck_erle_db'] = sc_erle
    return summ


def main():
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    ctx = sim.synthetic_context()
    h, info = sim.room_ir()
    res = {'echo_path': info, 'why_it_fails': why_it_fails(ctx, h, out), 'aec_erle': aec_erle(ctx, h, out)}
    (HERE / 'results' / 'figures_numbers.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
