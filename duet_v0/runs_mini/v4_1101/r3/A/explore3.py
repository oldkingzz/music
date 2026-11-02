"""One-off E4 exploration, part 3, v4 round 3 implementer A (not a system file, not a scored variant): the on/off-beat length
estimate with a longer history, and the HMM-only forecast with 32 lengths. All val songs (tuned on val).
Writes runs_mini/v4_1101/r3/A/explore3_<time>.json (new file, mode 'x')."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_copy import CHND, DS, OUTDIR, bench3, compare, nd, run_all  # noqa: E402
from explore2 import Phase  # noqa: E402
from systems import m1_filter_v5lo, r2_mix_v2, r3_cover_fit_v1  # noqa: E402
from systems.m3_seq_v4 import V3, Ens, _load  # noqa: E402

CONFIGS = [
    ('hmm_emp32_a8', 'cover', dict(side='hmm', length='emp', ndur=32, ahead=8, evid=0)),
    ('mix_phase16_w64_a8', 'phase', dict(side='mix', length='emp', ndur=16, window=64, ahead=8, evid=0)),
    ('mix_phase32_w128_a8', 'phase', dict(side='mix', length='emp', ndur=32, window=128, ahead=8, evid=0)),
    ('hmm_phase16_w64_a8', 'phase', dict(side='hmm', length='emp', ndur=16, window=64, ahead=8, evid=0)),
]


def main():
    t0 = time.time()
    train = bench3.train_sets()
    every = bench3.eval_sets('val')
    with open(CHND) as fh:
        rec = json.load(fh)
    ref = run_all(r2_mix_v2.build(train), every)
    cover = r3_cover_fit_v1.build(train)
    phase = Phase(m1_filter_v5lo.build(train).m, Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0,
                                                     dict(V3, step=0.01)))
    res = {'what': 'E4 settings part 3 on all val songs (tuned on val)', 'champion_nd': rec, 'configs': {}}
    for name, which, cfg in CONFIGS:
        mine = cover if which == 'cover' else phase
        for a, v in cfg.items():
            setattr(mine, a, v)
        t1 = time.time()
        outs = run_all(mine, every)
        got = nd(outs, every)
        c = compare(ref, outs)
        res['configs'][name] = {'cls': which, 'cfg': cfg, 'nd': got, 'notes_changed': c['notes_differing'],
                                'delta': {d: got[d]['music_nd'] - rec[d]['music_nd'] for d in DS},
                                'delta_headline': got['headline_nd'] - rec['headline_nd']}
        print(f"{name:20s} {got['headline_nd']:.5f} (d {got['headline_nd'] - rec['headline_nd']:+.5f}) " +
              ' '.join(f"{d} {got[d]['music_nd']:.4f}({got[d]['music_nd'] - rec[d]['music_nd']:+.4f})" for d in DS) +
              ' dbl ' + ' '.join(f"{got[d]['double']:.4f}" for d in DS) +
              f" changed {c['notes_differing']}/{c['notes']} {time.time() - t1:.0f}s", flush=True)
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"explore3_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', path, 'secs', res['secs'])


if __name__ == '__main__':
    main()
