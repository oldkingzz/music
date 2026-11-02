"""One-off E4 exploration for v4 round 3, implementer A (not a system file, not a scored variant): music_nd / double of
r3_cover_fit_v1's class under a few settings (instance attributes on one object; class and module untouched), on ALL val
songs (tuned on val: the bundle has no other held-out songs), plus how many of Duet's notes change vs r2_mix_v2.
Writes runs_mini/v4_1101/r3/A/explore_<time>.json (new file, mode 'x')."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_copy import CHND, DS, OUTDIR, bench3, compare, nd, run_all  # noqa: E402
from systems import r2_mix_v2, r3_cover_fit_v1  # noqa: E402

CONFIGS = [
    ('mix_emp8_a8', dict(side='mix', length='emp', ndur=8, ahead=8, evid=0)),
    ('mix_med8_a8', dict(side='mix', length='med', ndur=8, ahead=8, evid=0)),
    ('mix_emp8_a1', dict(side='mix', length='emp', ndur=8, ahead=1, evid=0)),
    ('hmm_emp8_a8', dict(side='hmm', length='emp', ndur=8, ahead=8, evid=0)),
    ('mix_emp8_a8_ev', dict(side='mix', length='emp', ndur=8, ahead=8, evid=1)),
    ('hmm_emp8_a8_ev', dict(side='hmm', length='emp', ndur=8, ahead=8, evid=1)),
]


def main():
    t0 = time.time()
    train = bench3.train_sets()
    every = bench3.eval_sets('val')
    with open(CHND) as fh:
        rec = json.load(fh)
    ref = run_all(r2_mix_v2.build(train), every)
    mine = r3_cover_fit_v1.build(train)
    res = {'what': 'E4 settings on all val songs (tuned on val)', 'champion_nd': rec, 'configs': {}}
    print(f"champion r2_mix_v2 {rec['headline_nd']:.5f} " + ' '.join(f"{d} {rec[d]['music_nd']:.4f}" for d in DS), flush=True)
    for name, cfg in CONFIGS:
        for a, v in cfg.items():
            setattr(mine, a, v)
        t1 = time.time()
        outs = run_all(mine, every)
        got = nd(outs, every)
        c = compare(ref, outs)
        res['configs'][name] = {'cfg': cfg, 'nd': got, 'notes_changed': c['notes_differing'], 'notes': c['notes'],
                                'delta': {d: got[d]['music_nd'] - rec[d]['music_nd'] for d in DS},
                                'delta_headline': got['headline_nd'] - rec['headline_nd'], 'secs': round(time.time() - t1, 1)}
        print(f"{name:16s} {got['headline_nd']:.5f} (d {got['headline_nd'] - rec['headline_nd']:+.5f}) " +
              ' '.join(f"{d} {got[d]['music_nd']:.4f}({got[d]['music_nd'] - rec[d]['music_nd']:+.4f})" for d in DS) +
              ' dbl ' + ' '.join(f"{got[d]['double']:.4f}" for d in DS) +
              f" changed {c['notes_differing']}/{c['notes']} {time.time() - t1:.0f}s", flush=True)
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"explore_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', path, 'secs', res['secs'])


if __name__ == '__main__':
    main()
