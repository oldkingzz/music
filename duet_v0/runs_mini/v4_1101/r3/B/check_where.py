"""One-off diagnostic, v4 round 3, implementer B (E5); not a system, not a scored variant; ALL val songs (val, tuned on val).
For r3_true_cons_v1 and r3_true_cons_v4 vs r2_mix_v2, note by note: music_nd time won / lost (rescore_nd.py's rule), split into
DIRECT changes (same previous Duet note) by her note misheard or heard right x heard pc in / out of the major scale, and CASCADE
changes (an earlier change moved Duet's voice). Shares are of each dataset's scored time. Writes check_where_<time>.json (mode 'x').
"""
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V3DIR = os.path.join(ABS, 'futures', 'v3_nolatency')
OUTDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'r3', 'B')
sys.path.insert(0, V3DIR)
import bench3  # noqa: E402
from theory import CONSONANT, CT, MAJOR  # noqa: E402
from systems import r2_mix_v2, r3_true_cons_v1, r3_true_cons_v4  # noqa: E402


def credit(song, hs):
    true = sorted(song['sop'], key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    t = song['tonic']
    num, den = [0.0] * len(true), [0.0] * len(true)
    for i, (on, off, m) in enumerate(true):
        h = hs[i]
        for k in range(max(bench3.J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            den[i] += ov
            if h >= 0:
                iv = abs(m - h) % 12
                num[i] += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])
    return num, den


def main():
    t0 = time.time()
    train = bench3.train_sets()
    sets = bench3.eval_sets('val')
    champ = r2_mix_v2.build(train)
    ref = {ds: [bench3.run_song(champ, s) for s in songs] for ds, songs in sets.items()}
    res = {'what': __doc__.splitlines()[0], 'variants': {}}
    for name, mod in (('r3_true_cons_v1', r3_true_cons_v1), ('r3_true_cons_v4', r3_true_cons_v4)):
        system = mod.build(train)
        out = {}
        for ds, songs in sets.items():
            acc, den_all, notes, oos_notes, mis_notes = {}, 0.0, 0, 0, 0
            for s, q in zip(songs, ref[ds]):
                o = bench3.run_song(system, s)
                heard = sorted(bench3.B2.heard(s)['sop'], key=lambda n: n[0])
                true = sorted(s['sop'], key=lambda n: n[0])
                n1, d1 = credit(s, o)
                n0, _ = credit(s, q)
                den_all += sum(d1)
                for i in range(len(o)):
                    mis = heard[i][2] != true[i][2]
                    oos = (heard[i][2] - s['tonic']) % 12 not in MAJOR
                    notes += 1
                    oos_notes += oos
                    mis_notes += mis
                    if o[i] == q[i]:
                        continue
                    if i == 0 or o[i - 1] == q[i - 1]:
                        key = ('misheard' if mis else 'heard_right') + '|' + ('out_of_scale' if oos else 'in_scale')
                    else:
                        key = 'cascade'
                    e = acc.setdefault(key, [0, 0.0, 0.0])
                    dlt = n1[i] - n0[i]
                    e[0] += 1
                    e[1] += max(dlt, 0.0)
                    e[2] += max(-dlt, 0.0)
            out[ds] = {'notes': notes, 'heard_out_of_scale_share': round(oos_notes / notes, 4),
                       'misheard_share': round(mis_notes / notes, 4)}
            for key, (n, w, l) in sorted(acc.items()):
                out[ds][key] = {'changed_notes': n, 'won': round(w / den_all, 5), 'lost': round(l / den_all, 5),
                                'net': round((w - l) / den_all, 5)}
            out[ds]['total_net'] = round(sum(v['net'] for k, v in out[ds].items() if isinstance(v, dict)), 5)
            print(name, ds, {k: (v['net'] if isinstance(v, dict) else v) for k, v in out[ds].items()}, flush=True)
        res['variants'][name] = out
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"check_where_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('secs', res['secs'], 'wrote', path)


if __name__ == '__main__':
    main()
