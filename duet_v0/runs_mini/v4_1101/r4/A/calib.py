"""One-off E4 diagnostic, v4 round 4 implementer A (not a system file, not a scored variant). (1) Calibration: at beat
boundaries that one of her (true) notes is held across, how often does the labelled chord change, and how often do the
style HMMs' chord-change tables say it changes (per val dataset; also counted on train)? (2) Exploration: r4_cover_fit_v1 with
'held-note' tables (chord changes counted on train only at boundaries her note is held across, backed off to the style's own
table with weight 5 as m1's order-2 table) on ALL val songs (tuned on val). Writes r4/A/calib_<time>.json (new file, mode 'x')."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_copy import CHND, DS, OUTDIR, bench3, compare, nd, run_all  # noqa: E402
from explore import PARTS, parts  # noqa: E402
from systems import r2_style_hmm_v1 as sty  # noqa: E402
from systems import r3_style_mix_v3, r4_cover_fit_v1  # noqa: E402
from systems.m1_filter_v5lo import CTM, S, tkey  # noqa: E402

BACK = 5.0


def held_counts(songs, wd, C):
    for s in songs:
        M, z = s['M'], [b['z'] for b in s['beats']]
        for on, off, m in s['sop']:
            for k in range(int(on) + 1, int(np.ceil(off))):
                if k >= len(z):
                    break
                if z[k - 1] >= 0 and z[k] >= 0:
                    C[tkey(s['beats'][k]['pos'], M), z[k - 1], z[k]] += wd
    return C


def held_tables(train, models):
    nmax = max(len(v) for v in train.values())
    groups = [[(ds, nmax / len(v)) for ds, v in train.items()]] + [[(ds, 1.0) for ds in hmm] for hmm, _, _ in sty.STYLES[1:]]
    out = []
    for g, mo in zip(groups, models):
        C = np.zeros((6, S, S))
        for ds, wd in g:
            held_counts(train[ds], wd, C)
        out.append((C + BACK * mo.A) / (C.sum(-1, keepdims=True) + BACK))
    return out


def change_rates(songs, tables):
    """Per boundary kind (held / all): actual change rate, and each table's implied mean P(change | previous chord)."""
    res = {}
    for kind in ('held', 'all'):
        n = ch = 0
        imp = np.zeros(len(tables))
        for s in songs:
            M, z = s['M'], [b['z'] for b in s['beats']]
            ks = set()
            if kind == 'held':
                for on, off, m in s['sop']:
                    ks.update(range(int(on) + 1, min(int(np.ceil(off)), len(z))))
            else:
                ks = set(range(1, len(z)))
            for k in ks:
                if z[k - 1] < 0 or z[k] < 0:
                    continue
                key = tkey(s['beats'][k]['pos'], M)
                n += 1
                ch += z[k] != z[k - 1]
                imp += np.array([1.0 - T[key, z[k - 1], z[k - 1]] for T in tables])
        res[kind] = {'boundaries': n, 'actual_change': ch / max(n, 1), 'implied_change': list(imp / max(n, 1))}
    return res


class HeldFc(r4_cover_fit_v1.CoverStyle):
    """r4_cover_fit_v1 with each style's forecast made by its held-note table (self.held, set by this script)."""

    def _pfit(self, k, onset):
        v = self._beat_v(k)
        bs = np.array([self._step(j, k, v) for j in range(len(self.models))])
        wt = np.exp(self.logw - self.logw.max())
        wt = wt / wt.sum()
        bg = np.maximum(np.asarray(self._p, dtype=np.float64), r4_cover_fit_v1.FLOOR)
        b0 = self._mixed(wt @ bs, bg)
        sh = self._shares(k, onset)
        if len(sh) == 1:
            return b0 @ CTM
        fc, hmm = bs, [wt @ bs]
        for j in range(1, len(sh)):
            key = tkey(self._pos(k + j), self.M)
            fc = np.array([fc[s] @ self.held[s][key] for s in range(len(self.models))])
            hmm.append(wt @ fc)
        if self.mix == 'cover':
            q = sh[0] * hmm[0]
            for j in range(1, len(sh)):
                q = q + sh[j] * hmm[j]
            return self._mixed(q, bg) @ CTM
        q = sh[0] * b0
        for j in range(1, len(sh)):
            q = q + sh[j] * self._mixed(hmm[j], bg)
        return q @ CTM


CONFIGS = [
    ('held_beat_p32w128_a8', dict(mix='beat', ndur=32, window=128, ahead=8)),
    ('held_cover_p32w128_a8', dict(mix='cover', ndur=32, window=128, ahead=8)),
]


def main():
    t0 = time.time()
    train = bench3.train_sets()
    every = bench3.eval_sets('val')
    built = r4_cover_fit_v1.build(train)
    held = held_tables(train, built.models)
    tables = [mo.A for mo in built.models] + held
    names = ['own_pooled', 'own_pop', 'own_pop909', 'held_pooled', 'held_pop', 'held_pop909']
    res = {'what': 'chord-change calibration at held-note boundaries + held-table E4 (all val songs, tuned on val)',
           'tables': names, 'calibration_val': {}, 'calibration_train': {}, 'configs': {}}
    for ds in DS:
        res['calibration_val'][ds] = change_rates(every[ds], tables)
        res['calibration_train'][ds] = change_rates(train[ds], tables)
        for where in ('calibration_val', 'calibration_train'):
            c = res[where][ds]
            print(f"{where[12:]:5s} {ds:7s} held: actual {c['held']['actual_change']:.3f} implied " +
                  ' '.join(f"{nm} {x:.3f}" for nm, x in zip(names, c['held']['implied_change'])) +
                  f" | all: actual {c['all']['actual_change']:.3f} n_held {c['held']['boundaries']}", flush=True)
    with open(CHND) as fh:
        rec = json.load(fh)
    ref = run_all(r3_style_mix_v3.build(train), every)
    base_parts = parts(ref, every)
    mine = HeldFc(built.models, built.lams, built.rh, built.mprior, built.ens)
    mine.held = held
    for name, cfg in CONFIGS:
        for a, v in cfg.items():
            setattr(mine, a, v)
        t1 = time.time()
        outs = run_all(mine, every)
        got = nd(outs, every)
        c = compare(ref, outs)
        pp = parts(outs, every)
        dparts = {d: {p: pp[d][p] - base_parts[d][p] for p in PARTS} for d in DS}
        res['configs'][name] = {'cfg': cfg, 'nd': got, 'notes_changed': c['notes_differing'],
                                'delta': {d: got[d]['music_nd'] - rec[d]['music_nd'] for d in DS},
                                'delta_headline': got['headline_nd'] - rec['headline_nd'], 'delta_parts': dparts}
        print(f"{name:24s} {got['headline_nd']:.5f} (d {got['headline_nd'] - rec['headline_nd']:+.5f}) " +
              ' '.join(f"{d} {got[d]['music_nd'] - rec[d]['music_nd']:+.4f}" for d in DS) +
              ' dbl ' + ' '.join(f"{got[d]['double'] - rec[d]['double']:+.4f}" for d in DS) +
              ' parts ' + ' '.join(f"{d[:2]} " + '/'.join(f"{dparts[d][p]:+.4f}" for p in PARTS) for d in DS) +
              f" chg {c['notes_differing']} {time.time() - t1:.0f}s", flush=True)
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"calib_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', path, 'secs', res['secs'])


if __name__ == '__main__':
    main()
