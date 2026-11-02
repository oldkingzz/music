"""Tables and slide figures from the sweep results.

    .venv/bin/python echo/analyze.py

Reads echo/results/{main,stress}.jsonl (from sweep.py), writes echo/results/summary.json and
echo/figures/*.png.

Tolerance criterion (per mitigation, pooled over 4 scenarios x 3 echo paths = 12 runs per level):
  (1) pooled median |asynchrony| of played notes <= no-bleed median + 10 ms   (the brief)
  (2) notes played >= 90 % of the no-bleed count  (otherwise a system that goes silent "passes")
  (3) no more wrong notes (sung where she skipped) than with no bleed
The tolerated bleed is the highest level L such that every tested level <= L passes.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
FIG = HERE / 'figures'

METHODS = ['none', 'aec_cold', 'mask', 'aec', 'aec+mask']
LABEL = {'none': 'No guard', 'aec_cold': 'Echo canceller, no sound check', 'mask': 'Pitch mask',
         'aec': 'Echo canceller', 'aec+mask': 'Canceller + mask'}
COLOR = {'none': '#8a8985', 'aec_cold': '#eda100', 'mask': '#eb6834', 'aec': '#2a78d6', 'aec+mask': '#1baf7a'}
INK, INK2, GRID = '#1d1d1f', '#6e6e73', '#e5e5ea'


def load(name):
    p = RES / name
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=11, length=0)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def aggregate(rows):
    """-> {method: {level: stats}} pooled over scenarios and paths."""
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g[r['method']][r['level']].append(r)
    out = {}
    for m, by in g.items():
        out[m] = {}
        for L, rs in by.items():
            errs = np.abs(np.concatenate([np.array(r['errors_ms']) for r in rs if r['errors_ms']] or [np.array([])]))
            out[m][L] = {
                'runs': len(rs),
                'played': int(sum(r['played'] for r in rs)), 'plan': int(sum(r['n_plan'] for r in rs)),
                'held': int(sum(r['held'] for r in rs)), 'missed': int(sum(r['missed'] for r in rs)),
                'wrong': int(sum(r['wrong'] for r in rs)), 'false_entries': int(sum(r['false_entries'] for r in rs)),
                'median_abs_ms': float(np.median(errs)) if errs.size else None,
                'p90_abs_ms': float(np.percentile(errs, 90)) if errs.size else None,
                'on_time30': int(np.sum(errs <= 30)),
                'tracking_within100ms': float(np.mean([r['tracking_within100ms'] for r in rs])),
                'resets': int(sum(r['resets'] for r in rs)),
                'bleed_measured_db': (float(np.mean([r['bleed_db_measured'] for r in rs if r['bleed_db_measured'] is not None]))
                                      if any(r['bleed_db_measured'] is not None for r in rs) else None),
                'guard_ms_median': float(np.median([r['guard_ms_median'] for r in rs])),
                'guard_ms_p99': float(np.max([r['guard_ms_p99'] for r in rs])),
            }
    return out


def tolerance(agg, ref):
    """Highest bleed level each method tolerates (see module docstring)."""
    base = ref
    res = {}
    for m, by in agg.items():
        levels = sorted(L for L in by if L is not None)
        tol, fail_reason, checks = None, None, {}
        for L in levels:
            s = by[L]
            ok_t = s['median_abs_ms'] is not None and s['median_abs_ms'] <= base['median_abs_ms'] + 10.0
            ok_p = s['played'] >= 0.9 * base['played']
            ok_w = s['wrong'] <= base['wrong']
            checks[L] = {'timing': ok_t, 'played': ok_p, 'wrong': ok_w}
            if ok_t and ok_p and ok_w:
                if fail_reason is None:
                    tol = L
            elif fail_reason is None:
                fail_reason = {'level': L, 'timing_ok': ok_t, 'played_ok': ok_p, 'wrong_ok': ok_w,
                               'median_abs_ms': s['median_abs_ms'], 'played': s['played']}
        t_only = None
        for L in levels:
            if checks[L]['timing']:
                t_only = L
            else:
                break
        res[m] = {'tolerated_db': tol, 'tolerated_db_timing_only': t_only, 'first_failure': fail_reason,
                  'checks': checks}
    return res


def fig_sweep(agg, tol, base, fn):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'wspace': 0.28})
    for ax in axes:
        style(ax)
    order = ['none', 'aec_cold', 'mask', 'aec', 'aec+mask']
    for m in order:
        by = agg.get(m)
        if not by:
            continue
        Ls = sorted(L for L in by if L is not None)
        frac = [100 * by[L]['on_time30'] / base['plan'] for L in Ls]
        med = [by[L]['median_abs_ms'] if by[L]['played'] >= 0.9 * base['played'] else np.nan for L in Ls]
        lw = 2.6 if m in ('aec+mask', 'none') else 2.0
        ls = '--' if m == 'aec_cold' else '-'
        axes[0].plot(Ls, frac, ls, color=COLOR[m], lw=lw, marker='o', ms=5, label=LABEL[m])
        axes[1].plot(Ls, med, ls, color=COLOR[m], lw=lw, marker='o', ms=5, label=LABEL[m])
    axes[0].axhline(100 * base['on_time30'] / base['plan'], color=INK2, lw=1, ls=':')
    axes[0].text(-36.5, 100 * base['on_time30'] / base['plan'] + 2.5, 'no bleed', color=INK2, fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Harmony notes on time (≤ 30 ms), %', color=INK, fontsize=12)
    axes[1].axhline(base['median_abs_ms'] + 10, color=INK2, lw=1, ls=':')
    axes[1].text(-36.5, base['median_abs_ms'] + 10.8, 'no-bleed + 10 ms', color=INK2, fontsize=10)
    axes[1].set_ylabel('Median timing error of played notes, ms', color=INK, fontsize=12)
    axes[1].set_ylim(0, max(40, base['median_abs_ms'] + 25))
    for ax in axes:
        ax.set_xlabel('Speaker-to-mic bleed, dB relative to her voice', color=INK, fontsize=12)
        ax.set_xticks([-36, -30, -24, -18, -12, -6, 0, 6])
    axes[1].text(0.99, 0.02, 'omitted where < 90 % of notes are played', transform=axes[1].transAxes,
                 ha='right', color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=10, loc='lower left')
    fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_tolerance(tol, fn, phys=None):
    """Hero figure: tolerated bleed per mitigation as horizontal bars."""
    order = ['none', 'aec_cold', 'mask', 'aec', 'aec+mask']
    fig, ax = plt.subplots(figsize=(10, 4.2))
    style(ax)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.grid(axis='y', visible=False)
    lo = -42
    for i, m in enumerate(order[::-1]):
        t = tol.get(m, {}).get('tolerated_db')
        v = lo if t is None else t
        ax.barh(i, v - lo, left=lo, height=0.56, color=COLOR[m])
        txt = 'below −36 dB' if t is None else f'{t:+.0f} dB'.replace('-', '−')
        if t is not None and t >= 6:
            txt = '≥ +6 dB  (highest level tested)'
        ax.text(v + 0.6, i, txt, va='center', fontsize=13, color=INK, fontweight='bold')
        ax.text(lo - 0.8, i, LABEL[m], va='center', ha='right', fontsize=12, color=INK)
    ax.set_yticks([])
    ax.set_xlim(lo, 12)
    ax.set_xlabel('Highest bleed it tolerates (timing within 10 ms of no-bleed, ≥ 90 % notes played)',
                  color=INK, fontsize=11)
    fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_physical(phys, tol, fn):
    rows = phys['configs']
    fig, ax = plt.subplots(figsize=(11, 4.6))
    style(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, lw=0.8)
    names = [r['config'] for r in rows][::-1]
    vals = [r['bleed_db'] for r in rows][::-1]
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.barh(i, v - (-40), left=-40, height=0.55, color='#c7c7cc')
        ax.text(v + 0.8, i, f'{v:+.0f} dB', va='center', fontsize=12, color=INK, fontweight='bold')
        ax.text(-41, i, n, va='center', ha='right', fontsize=11, color=INK)
    for m, lab in (('none', 'limit, no guard'), ('aec+mask', 'canceller + mask: fine at least to here')):
        t = tol.get(m, {}).get('tolerated_db')
        if t is not None:
            ax.axvline(t, color=COLOR[m], lw=2.2)
            ax.text(t + 0.4, len(rows) - 0.45, lab, color=COLOR[m], fontsize=10, va='bottom')
    ax.set_yticks([])
    ax.set_xlim(-40, 26)
    ax.set_xlabel('Machine voice at the mic, dB relative to her voice (estimate)', color=INK, fontsize=11)
    fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_stress(stress, base_by_scen, fn):
    if not stress:
        return
    g = defaultdict(lambda: defaultdict(list))
    for r in stress:
        g[(r['stress'], r['method'])][r['level']].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, gridspec_kw={'wspace': 0.12})
    titles = {'thd3': 'Speaker distortion 3 % (non-linear echo)', 'pathchange': 'Echo path changes mid-song'}
    for ax, st in zip(axes, ('thd3', 'pathchange')):
        style(ax)
        for m in ('none', 'mask', 'aec', 'aec+mask'):
            by = g.get((st, m))
            if not by:
                continue
            Ls = sorted(L for L in by if L is not None)
            if not Ls:
                continue
            plan = sum(r['n_plan'] for r in by[Ls[0]])
            on = [100 * sum(int(np.sum(np.abs(r['errors_ms']) <= 30)) for r in by[L]) / plan for L in Ls]
            ax.plot(Ls, on, color=COLOR[m], lw=2.4 if m == 'aec+mask' else 2.0, marker='o', ms=5, label=LABEL[m])
        ax.set_title(titles[st], color=INK, fontsize=13, loc='left')
        ax.set_xlabel('Speaker-to-mic bleed, dB re her voice', color=INK, fontsize=12)
        ax.set_xticks([-36, -24, -12, 0, 6])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel('Harmony notes on time (≤ 30 ms), %', color=INK, fontsize=12)
    axes[0].legend(frameon=False, fontsize=10, loc='lower left')
    fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    rows = load('main.jsonl')
    stress = load('stress.jsonl')
    agg = aggregate(rows)
    base = agg['none'][None]
    tol = tolerance(agg, base)
    # per-scenario tolerance too (is one scenario carrying the result?)
    per_scen = {}
    for scen in sorted({r['scenario'] for r in rows}):
        a = aggregate([r for r in rows if r['scenario'] == scen])
        per_scen[scen] = {m: v['tolerated_db'] for m, v in tolerance(a, a['none'][None]).items()}
    stress_tol = {}
    for st in sorted({r['stress'] for r in stress}):
        a = aggregate([r for r in stress if r['stress'] == st])
        b = aggregate([r for r in rows if r['scenario'] in ('natural', 'pause') and r['path'] == 1])['none'][None]
        stress_tol[st] = {m: v['tolerated_db'] for m, v in tolerance(a, b).items()}
    phys = json.loads((RES / 'physical.json').read_text()) if (RES / 'physical.json').exists() else None
    summary = {
        'criterion': 'pooled median |async| <= no-bleed + 10 ms AND played >= 90% of no-bleed AND wrong <= no-bleed',
        'no_bleed_reference': base,
        'tolerated_db': {m: tol[m]['tolerated_db'] for m in tol},
        'tolerated_db_timing_only': {m: tol[m]['tolerated_db_timing_only'] for m in tol},
        'first_failure': {m: tol[m]['first_failure'] for m in tol},
        'tolerated_db_per_scenario': per_scen,
        'tolerated_db_stress': stress_tol,
        'table': {m: {str(L): agg[m][L] for L in sorted(agg[m], key=lambda v: -99 if v is None else v)} for m in agg},
        'guard_cost_ms_per_10ms_frame': {m: {'median': float(np.median([agg[m][L]['guard_ms_median'] for L in agg[m]])),
                                             'p99_max': float(np.max([agg[m][L]['guard_ms_p99'] for L in agg[m]]))}
                                         for m in agg},
        'runs': len(rows), 'stress_runs': len(stress),
    }
    (RES / 'summary.json').write_text(json.dumps(summary, indent=1))
    fig_sweep(agg, tol, base, FIG / 'bleed_sweep.png')
    fig_tolerance(tol, FIG / 'tolerance.png')
    if phys:
        fig_physical(phys, tol, FIG / 'mic_placement.png')
    fig_stress(stress, None, FIG / 'stress.png')
    print('no-bleed reference: median', round(base['median_abs_ms'], 1), 'ms, played', base['played'], '/', base['plan'])
    print(f"{'method':10s} tolerated   " + ' '.join(f'{L:>6}' for L in sorted(L for L in agg['none'] if L is not None)))
    for m in METHODS:
        if m not in agg:
            continue
        Ls = sorted(L for L in agg[m] if L is not None)
        cells = []
        for L in Ls:
            s = agg[m][L]
            cells.append(f"{(s['median_abs_ms'] or 0):4.0f}/{100 * s['played'] / base['played']:3.0f}%")
        print(f"{m:10s} {str(tol[m]['tolerated_db']):>9s}   " + ' '.join(f'{c:>6}' for c in cells))
    print('timing-only criterion:', json.dumps({m: tol[m]['tolerated_db_timing_only'] for m in tol}))
    print('per scenario:', json.dumps(per_scen))
    print('stress:', json.dumps(stress_tol))
    nb = {m: agg[m].get(None) for m in agg}
    print('no-bleed per method (guard must not hurt):',
          {m: (round(v['median_abs_ms'], 1), v['played']) for m, v in nb.items() if v})


if __name__ == '__main__':
    main()
