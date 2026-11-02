"""One-off E4 exploration, v4 round 4 implementer A (not a system file, not a scored variant): r4_cover_fit_v1's class under a
few settings on ALL val songs (tuned on val), against r3_style_mix_v3; music_nd also split by where her note is (onset beat /
later beat with the onset beat's chord / later beat whose chord changed). Writes r4/A/explore_<time>.json (new file, mode 'x')."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_copy import CHND, DS, OUTDIR, bench3, compare, nd, run_all  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402
from systems import r3_style_mix_v3, r4_cover_fit_v1  # noqa: E402
from systems.m1_filter_v5lo import CTM, tkey  # noqa: E402

PARTS = ('onset_beat', 'later_same_chord', 'later_chord_changed')


class PooledFc(r4_cover_fit_v1.CoverStyle):
    """Ablation: every style's belief is forecast with the POOLED style's table (models[0].A) instead of its own."""

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
        fc, q = bs, sh[0] * b0
        for j in range(1, len(sh)):
            fc = fc @ self.models[0].A[tkey(self._pos(k + j), self.M)]
            q = q + sh[j] * self._mixed(wt @ fc, bg)
        return q @ CTM


class AnyLen(r4_cover_fit_v1.CoverStyle):
    """Length estimate: her last ndur finished notes of any kind (round 3's v4 estimate)."""

    def _lengths(self):
        durs = [n[1] - n[0] for n in self.notes[-(self.ndur + 1):-1] if n[1] is not None]
        return durs or [r4_cover_fit_v1.FIRST_LEN]


CONFIGS = [
    ('beat_p32w128_a8', 'cover', dict(mix='beat', ndur=32, window=128, ahead=8)),
    ('cover_p32w128_a8', 'cover', dict(mix='cover', ndur=32, window=128, ahead=8)),
    ('beat_p32w128_a2', 'cover', dict(mix='beat', ndur=32, window=128, ahead=2)),
    ('beat_p16w64_a8', 'cover', dict(mix='beat', ndur=16, window=64, ahead=8)),
    ('beat_any16_a8', 'any', dict(mix='beat', ndur=16, ahead=8)),
    ('beat_pooledA_p32w128_a8', 'pooled', dict(mix='beat', ndur=32, window=128, ahead=8)),
]


def parts(outs, sets):
    res = {}
    for ds, songs in sets.items():
        den = 0.0
        num = dict.fromkeys(PARTS, 0.0)
        for s, hs in zip(songs, outs[ds]):
            true = sorted(s['sop'], key=lambda n: n[0])
            z = [b['z'] for b in s['beats']]
            t = s['tonic']
            for i, (on, off, m) in enumerate(true):
                h = hs[i]
                k0 = int(on)
                for k in range(max(bench3.J, k0), int(off) + 1):
                    ov = min(off, k + 1) - max(on, k)
                    if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                        continue
                    p = PARTS[0] if k == k0 else (PARTS[1] if k0 < len(z) and z[k0] == z[k] else PARTS[2])
                    den += ov
                    if h >= 0:
                        iv = abs(m - h) % 12
                        num[p] += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])
        res[ds] = {p: num[p] / den for p in PARTS}
    return res


def main():
    t0 = time.time()
    train = bench3.train_sets()
    every = bench3.eval_sets('val')
    with open(CHND) as fh:
        rec = json.load(fh)
    ref = run_all(r3_style_mix_v3.build(train), every)
    base_parts = parts(ref, every)
    built = r4_cover_fit_v1.build(train)
    objs = {'cover': built,
            'any': AnyLen(built.models, built.lams, built.rh, built.mprior, built.ens),
            'pooled': PooledFc(built.models, built.lams, built.rh, built.mprior, built.ens)}
    res = {'what': 'E4 settings on all val songs (tuned on val), vs r3_style_mix_v3', 'champion_nd': rec,
           'champion_parts': base_parts, 'configs': {}}
    for name, which, cfg in CONFIGS:
        mine = objs[which]
        for a, v in cfg.items():
            setattr(mine, a, v)
        t1 = time.time()
        outs = run_all(mine, every)
        got = nd(outs, every)
        c = compare(ref, outs)
        pp = parts(outs, every)
        dparts = {d: {p: pp[d][p] - base_parts[d][p] for p in PARTS} for d in DS}
        res['configs'][name] = {'cls': which, 'cfg': cfg, 'nd': got, 'notes_changed': c['notes_differing'],
                                'delta': {d: got[d]['music_nd'] - rec[d]['music_nd'] for d in DS},
                                'delta_headline': got['headline_nd'] - rec['headline_nd'], 'delta_parts': dparts}
        print(f"{name:24s} {got['headline_nd']:.5f} (d {got['headline_nd'] - rec['headline_nd']:+.5f}) " +
              ' '.join(f"{d} {got[d]['music_nd'] - rec[d]['music_nd']:+.4f}" for d in DS) +
              ' dbl ' + ' '.join(f"{got[d]['double'] - rec[d]['double']:+.4f}" for d in DS) +
              ' parts ' + ' '.join(f"{d[:2]} " + '/'.join(f"{dparts[d][p]:+.4f}" for p in PARTS) for d in DS) +
              f" chg {c['notes_differing']} {time.time() - t1:.0f}s", flush=True)
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"explore_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('deltas vs r3_style_mix_v3 (music_nd, double, parts onset/later-same/later-changed); wrote', path,
          'secs', res['secs'])


if __name__ == '__main__':
    main()
