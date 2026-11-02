"""One-off E4 exploration, part 2, v4 round 3 implementer A (not a system file, not a scored variant): longer length history,
two beats ahead, and a length estimate from her recent notes that started at the same kind of place in the beat (on or off
the beat). All val songs (tuned on val). Writes runs_mini/v4_1101/r3/A/explore2_<time>.json (new file, mode 'x')."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_copy import CHND, DS, OUTDIR, bench3, compare, nd, run_all  # noqa: E402
from systems import m1_filter_v5lo, r2_mix_v2, r3_cover_fit_v1  # noqa: E402
from systems.m3_seq_v4 import V3, Ens, _load  # noqa: E402


class Phase(r3_cover_fit_v1.Cover):
    """Length estimate: her last ndur finished notes whose onset was on the beat (or off it) like the current one."""
    window = 32

    def _lengths(self):
        cur = self.notes[-1][0]
        off_beat = (cur - int(cur)) > 1e-6
        prev = [n for n in self.notes[-(self.window + 1):-1] if n[1] is not None]
        same = [n[1] - n[0] for n in prev if ((n[0] - int(n[0])) > 1e-6) == off_beat][-self.ndur:]
        if len(same) >= 2:
            return same
        durs = [n[1] - n[0] for n in prev][-self.ndur:]
        return durs or [r3_cover_fit_v1.FIRST_LEN]


CONFIGS = [
    ('mix_emp16_a8', 'cover', dict(side='mix', length='emp', ndur=16, ahead=8, evid=0)),
    ('mix_emp32_a8', 'cover', dict(side='mix', length='emp', ndur=32, ahead=8, evid=0)),
    ('mix_emp8_a2', 'cover', dict(side='mix', length='emp', ndur=8, ahead=2, evid=0)),
    ('hmm_emp16_a8', 'cover', dict(side='hmm', length='emp', ndur=16, ahead=8, evid=0)),
    ('mix_phase8_a8', 'phase', dict(side='mix', length='emp', ndur=8, ahead=8, evid=0)),
    ('hmm_phase8_a8', 'phase', dict(side='hmm', length='emp', ndur=8, ahead=8, evid=0)),
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
    res = {'what': 'E4 settings part 2 on all val songs (tuned on val)', 'champion_nd': rec, 'configs': {}}
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
        print(f"{name:16s} {got['headline_nd']:.5f} (d {got['headline_nd'] - rec['headline_nd']:+.5f}) " +
              ' '.join(f"{d} {got[d]['music_nd']:.4f}({got[d]['music_nd'] - rec[d]['music_nd']:+.4f})" for d in DS) +
              ' dbl ' + ' '.join(f"{got[d]['double']:.4f}" for d in DS) +
              f" changed {c['notes_differing']}/{c['notes']} {time.time() - t1:.0f}s", flush=True)
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"explore2_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', path, 'secs', res['secs'])


if __name__ == '__main__':
    main()
