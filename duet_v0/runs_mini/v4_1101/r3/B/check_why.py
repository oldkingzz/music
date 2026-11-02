"""One-off diagnostic, v4 round 3, implementer B (E5); not a system, not a scored variant; val songs, tuned on val.
On the first 40 val songs per dataset, compares r3_true_cons_v1 (and v3) with r2_mix_v2 note by note and classifies the DIRECT
changes (same previous Duet note, so only E5 differs) by: her note misheard or not, E5's note consonant with the HEARD note or
not, the model's P(misheard), and whether the belief's top chord was the true chord; reports music_nd time won / lost per class.
Writes runs_mini/v4_1101/r3/B/check_why_<time>.json (new file, mode 'x').
"""
import json
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V3DIR = os.path.join(ABS, 'futures', 'v3_nolatency')
OUTDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'r3', 'B')
sys.path.insert(0, V3DIR)
import bench3  # noqa: E402
from theory import CONSONANT, CT, MAJOR  # noqa: E402
from systems import r2_mix_v2, r3_true_cons_v1, r3_true_cons_v3  # noqa: E402

N_FIRST = 40


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


def recorder(system):
    log = []
    cls = type(system)

    def rec(k):
        b = cls.belief(system, k)
        log.append(np.asarray(b, dtype=np.float64))
        return b
    system.belief = rec                     # instance attribute; the class is untouched
    return log


def add(d, key, dlt):
    e = d.setdefault(key, {'n': 0, 'won': 0.0, 'lost': 0.0})
    e['n'] += 1
    e['won'] += max(dlt, 0.0)
    e['lost'] += max(-dlt, 0.0)


def main():
    t0 = time.time()
    train = bench3.train_sets()
    first = {ds: songs[:N_FIRST] for ds, songs in bench3.eval_sets('val').items()}
    champ = r2_mix_v2.build(train)
    res = {'what': __doc__.splitlines()[0], 'variants': {}}
    for name, mod in (('r3_true_cons_v1', r3_true_cons_v1), ('r3_true_cons_v3', r3_true_cons_v3)):
        system = mod.build(train)
        blog = recorder(system)
        Bc, N = system.m.Bc, system.m.N
        out = {}
        for ds, songs in first.items():
            cls_ = {}
            for s in songs:
                q = bench3.run_song(champ, s)
                del blog[:]
                o = bench3.run_song(system, s)
                heard = sorted(bench3.B2.heard(s)['sop'], key=lambda n: n[0])
                true = sorted(s['sop'], key=lambda n: n[0])
                n1, _ = credit(s, o)
                n0, _ = credit(s, q)
                t = s['tonic']
                zt = [b['z'] for b in s['beats']]
                for i in range(len(o)):
                    if o[i] == q[i]:
                        continue
                    direct = i == 0 or o[i - 1] == q[i - 1]
                    dlt = n1[i] - n0[i]
                    if not direct:
                        add(cls_, 'cascade', dlt)
                        continue
                    hm = heard[i][2]
                    hp = (hm - t) % 12
                    mis = hm != true[i][2]
                    b = blog[i]
                    pt = Bc * N[:, hp][None, :]
                    pt = pt / pt.sum(1, keepdims=True)
                    pmis = 1.0 - float(b @ pt[:, hp])
                    k = int(heard[i][0])
                    top_ok = k < len(zt) and zt[k] >= 0 and int(np.argmax(b)) == zt[k]
                    e5_cons_heard = (abs(hm - o[i]) % 12) in CONSONANT
                    key = '|'.join(['misheard' if mis else 'heard_right',
                                    'in_scale' if hp in MAJOR else 'out_of_scale',
                                    'e5_cons_heard' if e5_cons_heard else 'e5_DISS_heard',
                                    'pmis<0.3' if pmis < 0.3 else 'pmis>=0.3',
                                    'top_chord_right' if top_ok else 'top_chord_wrong'])
                    add(cls_, key, dlt)
            out[ds] = {k: {'n': v['n'], 'won': round(v['won'], 2), 'lost': round(v['lost'], 2),
                           'net': round(v['won'] - v['lost'], 2)} for k, v in sorted(cls_.items())}
        res['variants'][name] = out
        for ds, v in out.items():
            print(name, ds)
            for k, e in v.items():
                print('   ', k, e)
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"check_why_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('secs', res['secs'], 'wrote', path)


if __name__ == '__main__':
    main()
