"""One-off diagnostic, v4 round 4, implementer B (E5); not a system, not a scored variant; ALL val songs (val, tuned on val).
Does leaving a suspect heard note out of the chord belief make the REAL-TIME chord guess better? For each setting, at every note:
top-1 accuracy of the belief for beat k (= int(onset)) against the true chord label, and mean P(true chord), split into notes
that are suspect (r3_true_cons_v4's rule) and misheard / heard right, and all other notes. Settings: the champion copy, r4 v1's
belief side ('cur'), 'all' (suspect notes never enter the HMM), and an ORACLE (belief fed her TRUE notes; diagnosis only).
Writes runs_mini/v4_1101/r4/B/check_belief_<time>.json (new file, mode 'x').
"""
import json
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V3DIR = os.path.join(ABS, 'futures', 'v3_nolatency')
OUTDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'r4', 'B')
sys.path.insert(0, V3DIR)
sys.path.insert(0, OUTDIR)
import bench3  # noqa: E402
from systems import r4_true_cons_v1 as R  # noqa: E402
from check_why import OracleBelief, run_clean  # noqa: E402


def recorder(system):
    log = []
    cls = type(system)

    def rec(k):
        b = cls.belief(system, k)
        log.append((k, np.asarray(b, dtype=np.float64)))
        return b
    system.belief = rec                     # instance attribute; the class is untouched
    return log


def main():
    t0 = time.time()
    train = bench3.train_sets()
    sets = bench3.eval_sets('val')
    base = R.build(train)
    orc = OracleBelief(base.models, base.lams, base.rh, base.mprior, base.ens)
    orc.belief_side, orc.choice_side = 'off', 'off'
    res = {'what': __doc__.splitlines()[0], 'rows': {}}
    settings = (('champion_copy', base, 'off'), ('cur_prev', base, 'cur'), ('all_prev', base, 'all'), ('oracle_true_notes', orc, 'off'))
    for sname, system, bside in settings:
        system.belief_side = bside
        log = recorder(system)
        out = {}
        for ds, songs in sets.items():
            acc = {}
            for s in songs:
                heard = sorted(bench3.B2.heard(s)['sop'], key=lambda n: n[0])
                true = sorted(s['sop'], key=lambda n: n[0])
                z = [b['z'] for b in s['beats']]
                # which notes are suspect in the real (heard) setting: from a run of the champion-side rule
                del log[:]
                if system is orc:
                    orc.true_m = [int(n[2]) for n in true]
                bench3.run_song(system, s)
                sus = list(base._sus) if system is base else None
                if system is orc:
                    base.belief_side = 'off'
                    bench3.run_song(base, s)        # only to get the suspect flags of the heard notes
                    sus = list(base._sus)
                    blog = list(log[-len(true) * 2:-len(true)]) if len(log) >= 2 * len(true) else list(log[:len(true)])
                else:
                    blog = list(log)
                for i, (k, b) in enumerate(blog[:len(true)]):
                    if k >= len(z) or z[k] < 0 or k < bench3.J:
                        continue
                    mis = heard[i][2] != true[i][2]
                    key = ('suspect' if sus[i] else 'not_suspect') + '|' + ('misheard' if mis else 'heard_right')
                    e = acc.setdefault(key, [0, 0.0, 0.0])
                    e[0] += 1
                    e[1] += float(np.argmax(b) == z[k])
                    e[2] += float(b[z[k]])
            tot = [sum(v[j] for v in acc.values()) for j in range(3)]
            out[ds] = {key: {'notes': v[0], 'top1': round(v[1] / v[0], 4), 'p_true': round(v[2] / v[0], 4)} for key, v in sorted(acc.items())}
            out[ds]['all'] = {'notes': tot[0], 'top1': round(tot[1] / tot[0], 4), 'p_true': round(tot[2] / tot[0], 4)}
            print(sname, ds, out[ds], flush=True)
        del system.belief
        res['rows'][sname] = out
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"check_belief_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('secs', res['secs'], 'wrote', path)


if __name__ == '__main__':
    main()
