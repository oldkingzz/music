"""One-off E4 diagnostic, v4 round 3 implementer A (not a system, not a scored variant): where does music_nd change? Each
scored stretch of her note is put in one of three places: her onset beat; a later beat with the same chord label as the onset
beat; a later beat whose chord differs (the chord changed under her held note). music_nd = sum of the three parts. All val
songs (tuned on val). Writes runs_mini/v4_1101/r3/A/where_<time>.json (new file, mode 'x')."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_copy import DS, OUTDIR, bench3, run_all  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402
from systems import r2_mix_v2, r3_cover_fit_v1, r3_cover_fit_v3, r3_cover_fit_v4  # noqa: E402

PARTS = ('onset_beat', 'later_same_chord', 'later_chord_changed')


def parts(outs, sets):
    res = {}
    for ds, songs in sets.items():
        den = 0.0
        num = dict.fromkeys(PARTS, 0.0)
        tim = dict.fromkeys(PARTS, 0.0)
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
                    tim[p] += ov
                    if h >= 0:
                        iv = abs(m - h) % 12
                        num[p] += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])
        res[ds] = {'music_nd_part': {p: num[p] / den for p in PARTS}, 'time_share': {p: tim[p] / den for p in PARTS}}
    return res


def main():
    t0 = time.time()
    train = bench3.train_sets()
    every = bench3.eval_sets('val')
    res = {'what': 'music_nd split by where her note is (all val songs, tuned on val)', 'systems': {}}
    base = None
    for name, mod in (('r2_mix_v2', r2_mix_v2), ('r3_cover_fit_v1', r3_cover_fit_v1), ('r3_cover_fit_v3', r3_cover_fit_v3),
                      ('r3_cover_fit_v4', r3_cover_fit_v4)):
        got = parts(run_all(mod.build(train), every), every)
        res['systems'][name] = got
        if base is None:
            base = got
            print('time share  ' + '  '.join(f"{d} " + '/'.join(f"{got[d]['time_share'][p]:.3f}" for p in PARTS) for d in DS))
        print(f"{name:16s}" + '  '.join(
            f"{d} " + '/'.join(f"{got[d]['music_nd_part'][p] - (0 if got is base else base[d]['music_nd_part'][p]):+.4f}"
                               for p in PARTS) for d in DS), flush=True)
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"where_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('rows after the first: change vs r2_mix_v2 in onset_beat / later_same_chord / later_chord_changed; wrote', path,
          'secs', res['secs'])


if __name__ == '__main__':
    main()
