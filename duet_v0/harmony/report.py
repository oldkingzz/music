"""Figures + harmony/eval.md from harmony/results/eval_results.json and params/*.json.

    .venv/bin/python harmony/report.py
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COL = {'Rule (follow.plan_harmony)': '#eb6834', 'CRF (Bach style)': '#1baf7a',
       'CRF + duet constraints': '#2a78d6', 'Bach (reference)': '#8a8984'}
SHORT = {'Rule (follow.plan_harmony)': 'Rule', 'CRF (Bach style)': 'CRF',
         'CRF + duet constraints': 'CRF + duet', 'Bach (reference)': 'Bach'}

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
                     'axes.spines.top': False, 'axes.spines.right': False})


def fig_metrics(R):
    panels = [('alto_exact_%', 'Same note as Bach\'s alto (%)'), ('chord_tone_%', 'In Bach\'s chord (%)'),
              ('consonance_2v_%', 'Consonant with melody, 2-voice (%)'),
              ('repeated_note_%', 'Repeated harmony note (%)')]
    names = list(COL)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (key, title) in zip(axes, panels):
        vals = [R['test'][n][key] for n in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=[COL[n] for n in names], height=0.62)
        for n, yy, v in zip(names, y, vals):
            ci = R['test_ci'].get(n, {}).get(key)
            right = v
            if ci and n != 'Bach (reference)':
                ax.plot(ci, [yy, yy], color=INK, lw=1.2)
                right = max(v, ci[1])
            ax.text(right + 1.5, yy, f'{v:.0f}', va='center', fontsize=9, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[n] for n in names] if ax is axes[0] else [])
        ax.set_xlim(0, 112)
        ax.set_title(title, loc='left', fontsize=10)
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"Held-out Bach chorales (test, {R['split_sizes']['test']} chorales, tune-disjoint); "
                 'bars = pooled rate, black line = 95 % bootstrap CI over chorales', x=0.01, ha='left',
                 fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'test_metrics.png'), dpi=170)
    plt.close(fig)


def fig_sweep(R):
    sw = R['duet_sweep_val']
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = [r['consonance_2v_%'] for r in sw]
    y = [r['alto_exact_%'] for r in sw]
    rep = np.array([r['repeated_note_%'] for r in sw])
    ok = rep <= R['bach_val']['repeated_note_%']
    ax.scatter(np.array(x)[~ok], np.array(y)[~ok], s=36, facecolors='none', edgecolors='#b5b4ae',
               label='grid point, more static than Bach')
    ax.scatter(np.array(x)[ok], np.array(y)[ok], s=36, color='#b5b4ae', label='grid point, Bach-like motion')
    base = next(r for r in sw if r['lam'] == 0 and r['beta'] == 0)
    ch = next(r for r in sw if r['lam'] == R['lam'] and r['beta'] == R['beta'])
    ax.scatter([base['consonance_2v_%']], [base['alto_exact_%']], s=80, color=COL['CRF (Bach style)'],
               zorder=3, label='CRF, no duet penalties')
    ax.scatter([ch['consonance_2v_%']], [ch['alto_exact_%']], s=110, color=COL['CRF + duet constraints'],
               zorder=3, label=f"shipped: lambda={R['lam']}, beta={R['beta']}")
    rv = R['rule_val']
    ax.scatter([rv['consonance_2v_%']], [rv['alto_exact_%']], s=80, color=COL['Rule (follow.plan_harmony)'],
               marker='s', zorder=3, label='Rule (third/sixth below)')
    ax.axvline(95, color=INK2, lw=0.8, ls='--')
    ax.text(95.3, min(y) - 0.5, '95 %', color=INK2, fontsize=9)
    ax.set_xlabel('two-voice consonance with the melody (%)')
    ax.set_ylabel("same note as Bach's alto (%)")
    ax.set_title('Decoding trade-off on the validation split', loc='left')
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'duet_tradeoff_val.png'), dpi=170)
    plt.close(fig)


def fig_hist(R):
    names = ['Rule (follow.plan_harmony)', 'CRF + duet constraints', 'Bach (reference)']
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharey='row')
    lab = ['uni', 'm2', 'M2', 'm3', 'M3', 'P4', 'TT', 'P5', 'm6', 'M6', 'm7', 'M7', '8ve']
    for j, n in enumerate(names):
        h = np.array(R['interval_hist'][n]['interval'])
        axes[0, j].bar(np.arange(13), 100 * h / h.sum(), color=COL[n], width=0.75)
        axes[0, j].set_xticks(np.arange(13))
        axes[0, j].set_xticklabels(lab, fontsize=7)
        axes[0, j].set_title(SHORT[n] + ': interval below the melody', loc='left', fontsize=10)
        m = np.array(R['interval_hist'][n]['motion'])[5:20]
        axes[1, j].bar(np.arange(-7, 8), 100 * m / m.sum(), color=COL[n], width=0.75)
        axes[1, j].set_title(SHORT[n] + ': harmony step (semitones)', loc='left', fontsize=10)
        for ax in axes[:, j]:
            ax.grid(axis='y', color=GRID, lw=0.6)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel('% of notes')
    axes[1, 0].set_ylabel('% of moves')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'interval_motion_hist.png'), dpi=170)
    plt.close(fig)


def fig_ablation(A):
    best = {}
    for r in A:
        if r['set'] not in best or r['val_acc'] > best[r['set']]['val_acc']:
            best[r['set']] = r
    rows = sorted(best.values(), key=lambda r: r['val_acc'])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(rows))
    ax.barh(y, [100 * r['val_acc'] for r in rows], color='#2a78d6', height=0.6)
    for yy, r in zip(y, rows):
        ax.text(100 * r['val_acc'] + 0.3, yy, f"{100 * r['val_acc']:.1f}  (W={r['W']}, sigma={r['sigma']})",
                va='center', fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r['set'] for r in rows], fontsize=9)
    ax.set_xlim(30, 46)
    ax.set_xlabel("validation: same note as Bach's alto (%)")
    ax.set_title('Feature ablation (CRF, Viterbi, no duet penalties)', loc='left', fontsize=10)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'ablation_val.png'), dpi=170)
    plt.close(fig)
    return best


def main():
    with open(os.path.join(HERE, 'results', 'eval_results.json')) as fh:
        R = json.load(fh)
    os.makedirs(FIG, exist_ok=True)
    fig_metrics(R)
    fig_sweep(R)
    fig_hist(R)
    A = None
    p = os.path.join(HERE, 'params', 'ablation_alto.json')
    if os.path.exists(p):
        with open(p) as fh:
            A = json.load(fh)
        fig_ablation(A)
    print('figures written to', FIG)


if __name__ == '__main__':
    main()
