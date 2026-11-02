"""Verifier (run v5rt_1101): is rt:r2_copyhmbold_v1's gain over reactive:r2_mix_v2 broad across val songs or carried by a
few? Per song, bench_rt.simulate + bench_rt.score_song (unchanged) give hit / harsh / counted time, and attack_metric's
song_counts (unchanged) gives fresh clashes. Then: share of songs where the champion's per-song rt_nd beats reactive's,
and a song bootstrap (resample songs within each dataset, 2000 reps, numpy seed 20261101) of the headline deltas.
L = 0.1, seed 1, val only (tuned on val; no test split). Writes one new JSON (mode 'x') under runs_mini/v5rt_1101/verify/.

  .venv/bin/python runs_mini/v5rt_1101/verify/song_delta_v1.py --out runs_mini/v5rt_1101/verify/<new>.json
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402

_s = importlib.util.spec_from_file_location('attack_metric', os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'tools',
                                                                          'attack_metric.py'))
AM = importlib.util.module_from_spec(_s)
_s.loader.exec_module(AM)
DS = ('bach', 'folk', 'pop', 'pop909')
L = 0.1


def per_song(spec):
    B.bench3.B2.heard = B.HEARD1
    obj = B.load(spec, B.bench3.train_sets())
    out = {}
    for ds, songs in B.bench3.eval_sets('val').items():
        rows = []
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, _, _ = B.simulate(obj, meta, notes, L)
            acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                                 'on_time_right', 'changes', 'late_entry'), 0.0)
            B.score_song(s, T, H, L, acc)
            aa = dict.fromkeys(('notes', 'attacks', 'fresh_clash', 'attack_miss', 'd', 'harsh_fresh', 'harsh_bleed'), 0.0)
            AM.song_counts(s, T, H, L, aa)
            rows.append((acc['d'], acc['hit'], acc['harsh'], aa['notes'], aa['fresh_clash']))
        out[ds] = np.array(rows)
    return out


def head(c, r, idx):
    """Headline deltas (champion - reactive) for song index arrays idx[ds]."""
    nd = hr = fc = 0.0
    for ds in DS:
        a, b = c[ds][idx[ds]], r[ds][idx[ds]]
        nd += a[:, 1].sum() / a[:, 0].sum() - b[:, 1].sum() / b[:, 0].sum()
        hr += a[:, 2].sum() / a[:, 0].sum() - b[:, 2].sum() / b[:, 0].sum()
        fc += 100 * (a[:, 4].sum() / max(a[:, 3].sum(), 1) - b[:, 4].sum() / max(b[:, 3].sum(), 1))
    return nd / 4, hr / 4, fc / 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'verify') + os.sep) or os.path.exists(path):
        sys.exit('out must be a new file under runs_mini/v5rt_1101/verify/')
    c = per_song('rt:r2_copyhmbold_v1')
    r = per_song('reactive:r2_mix_v2')
    res = {'L': L, 'seed': 'seed1', 'split': 'val', 'rows': {}}
    for ds in DS:
        a_, b_ = c[ds], r[ds]
        ok = (a_[:, 0] > 0) & (b_[:, 0] > 0)
        nd_a = a_[ok, 1] / a_[ok, 0]
        nd_b = b_[ok, 1] / b_[ok, 0]
        res['rows'][ds] = {
            'songs': int(len(a_)), 'songs_counted': int(ok.sum()),
            'champion_better': int((nd_a > nd_b + 1e-12).sum()), 'equal': int((abs(nd_a - nd_b) <= 1e-12).sum()),
            'reactive_better': int((nd_b > nd_a + 1e-12).sum()),
            'median_song_delta_nd': float(np.median(nd_a - nd_b)),
            'rt_nd': [float(a_[:, 1].sum() / a_[:, 0].sum()), float(b_[:, 1].sum() / b_[:, 0].sum())],
            'fresh_clashes': [int(a_[:, 4].sum()), int(b_[:, 4].sum())], 'notes': int(a_[:, 3].sum()),
            'songs_more_fresh': int((a_[:, 4] > b_[:, 4]).sum()), 'songs_fewer_fresh': int((a_[:, 4] < b_[:, 4]).sum())}
    full = {ds: np.arange(len(c[ds])) for ds in DS}
    res['headline_delta'] = dict(zip(('rt_nd', 'harsh', 'fresh_per100'), head(c, r, full)))
    rng = np.random.default_rng(20261101)
    reps = []
    for _ in range(2000):
        idx = {ds: rng.integers(0, len(c[ds]), len(c[ds])) for ds in DS}
        reps.append(head(c, r, idx))
    reps = np.array(reps)
    res['bootstrap'] = {k: {'p2.5': float(np.percentile(reps[:, i], 2.5)), 'p50': float(np.percentile(reps[:, i], 50)),
                            'p97.5': float(np.percentile(reps[:, i], 97.5)),
                            'share_rt_nd_delta_ge_0.005' if k == 'rt_nd' else 'share_le_0':
                                float((reps[:, i] >= 0.005).mean() if k == 'rt_nd' else (reps[:, i] <= 0).mean())}
                        for i, k in enumerate(('rt_nd', 'harsh', 'fresh_per100'))}
    bach = []
    for _ in range(2000):
        i = rng.integers(0, len(c['bach']), len(c['bach']))
        a_, b_ = c['bach'][i], r['bach'][i]
        bach.append(100 * (a_[:, 4].sum() / a_[:, 3].sum() - b_[:, 4].sum() / b_[:, 3].sum()))
    res['bootstrap']['bach_fresh_per100_delta'] = {'p2.5': float(np.percentile(bach, 2.5)),
                                                   'p97.5': float(np.percentile(bach, 97.5)),
                                                   'share_gt_1.0': float((np.array(bach) > 1.0).mean())}
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
