"""One-off reproduction check, v4 round 3, implementer B (E5); not a system, not a scored variant; ALL val songs (val, tuned on val).
(1) r3_true_cons_v1 with E5 switched off (instance attribute mode = 'off') on every val song: music_nd and double, recomputed with
    rescore_nd.py's exact loops and order, must equal the champion's recorded nd_r2_mix_v2.json bit for bit;
(2) each scored r3_true_cons_v<k> as built must reproduce its own recorded nd_<name>.json (determinism of the scored numbers).
Writes runs_mini/v4_1101/r3/B/check_repro_<time>.json (new file, mode 'x'); exits 1 on any difference.
"""
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V3DIR = os.path.join(ABS, 'futures', 'v3_nolatency')
RUN = os.path.join(ABS, 'runs_mini', 'v4_1101')
sys.path.insert(0, V3DIR)
import bench3  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402
from systems import r3_true_cons_v1, r3_true_cons_v2, r3_true_cons_v3, r3_true_cons_v4  # noqa: E402


def nd(sysm, sets):
    """rescore_nd.py's computation, same loops, same order."""
    out = {}
    for ds, songs in sets.items():
        num = den = dbl = 0.0
        for s in songs:
            hs = bench3.run_song(sysm, s)
            true = sorted(s['sop'], key=lambda n: n[0]); z = [b['z'] for b in s['beats']]; t = s['tonic']
            for i, (on, off, m) in enumerate(true):
                h = hs[i]
                for k in range(max(bench3.J, int(on)), int(off) + 1):
                    ov = min(off, k + 1) - max(on, k)
                    if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                        continue
                    den += ov
                    if h >= 0:
                        iv = abs(m - h) % 12
                        num += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])
                        dbl += ov * (iv == 0)
        out[ds] = {'music_nd': num / den, 'double': dbl / den}
    out['headline_nd'] = sum(v['music_nd'] for v in out.values()) / 4
    return out


def recorded(name):
    with open(os.path.join(RUN, 'nd', name, 'results', f'nd_{name}.json')) as fh:
        return json.load(fh)


def same(a, b):
    return a == b


def main():
    t0 = time.time()
    train = bench3.train_sets()
    sets = bench3.eval_sets('val')
    res = {'songs': {d: len(v) for d, v in sets.items()}}
    off = r3_true_cons_v1.build(train)
    off.mode = 'off'
    got = nd(off, sets)
    rec = recorded('r2_mix_v2')
    res['v1_E5_off_vs_recorded_r2_mix_v2'] = {'got': got, 'recorded': rec, 'exact': same(got, rec)}
    ok = same(got, rec)
    print('v1 with E5 off reproduces r2_mix_v2 nd exactly:', same(got, rec), round(got['headline_nd'], 6), flush=True)
    for name, mod in (('r3_true_cons_v1', r3_true_cons_v1), ('r3_true_cons_v2', r3_true_cons_v2),
                      ('r3_true_cons_v3', r3_true_cons_v3), ('r3_true_cons_v4', r3_true_cons_v4)):
        g = nd(mod.build(train), sets)
        r = recorded(name)
        res[name + '_vs_own_recorded'] = {'got_headline_nd': g['headline_nd'], 'recorded_headline_nd': r['headline_nd'],
                                          'exact': same(g, r)}
        ok = ok and same(g, r)
        print(name, 'reproduces its recorded nd exactly:', same(g, r), round(g['headline_nd'], 6), flush=True)
    res['all_ok'] = ok
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(RUN, 'r3', 'B', f"check_repro_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('ALL_OK', ok, 'secs', res['secs'], 'wrote', path)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
