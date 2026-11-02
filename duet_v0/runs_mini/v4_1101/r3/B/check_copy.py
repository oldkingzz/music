"""One-off faithfulness check, v4 round 3, implementer B (E5 r3_true_cons); not a system, not a scored variant; val songs, tuned on val.
(1) Every r3_true_cons_v<k> file present, with E5 switched off (instance attribute mode = 'off'; class and modules untouched),
    must sing exactly r2_mix_v2's notes through bench3.run_song on the first 40 val songs of each dataset (bach has 17), and its
    3-GRU ensemble must advance exactly once per note.
(2) Diagnostic with each file's own MODE on the same songs: notes changed vs the champion and where (her heard note misheard or
    not; heard pc in or out of the major scale), and the music_nd time won / lost on the changed notes (rescore_nd.py's rule).
Writes runs_mini/v4_1101/r3/B/check_copy_<time>.json (new file, mode 'x'); exits 1 if (1) fails anywhere.
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
from systems import r2_mix_v2  # noqa: E402

MODS = {}
from systems import r3_true_cons_v1  # noqa: E402
MODS['r3_true_cons_v1'] = r3_true_cons_v1
try:
    from systems import r3_true_cons_v2  # noqa: E402
    MODS['r3_true_cons_v2'] = r3_true_cons_v2
except ImportError:
    pass
try:
    from systems import r3_true_cons_v3  # noqa: E402
    MODS['r3_true_cons_v3'] = r3_true_cons_v3
except ImportError:
    pass
try:
    from systems import r3_true_cons_v4  # noqa: E402
    MODS['r3_true_cons_v4'] = r3_true_cons_v4
except ImportError:
    pass

N_FIRST = 40


def counter(system):
    calls = [0]
    orig = system.ens.chord_probs

    def counted(onset, midi, prev_dur):
        calls[0] += 1
        return orig(onset, midi, prev_dur)
    system.ens.chord_probs = counted          # instance attribute on this one Ens object
    return calls


def credit(song, hs):
    """Per her note i: (music_nd time, scored time), the same rule and loops as rescore_nd.py."""
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
    first = {ds: songs[:N_FIRST] for ds, songs in bench3.eval_sets('val').items()}
    champ = r2_mix_v2.build(train)
    ref = {ds: [bench3.run_song(champ, s) for s in songs] for ds, songs in first.items()}
    res = {'check': 'r3 B: r3_true_cons_v<k> with E5 off vs r2_mix_v2 (first 40 val songs per dataset; val, tuned on val)',
           'songs': {d: len(v) for d, v in first.items()}, 'variants': {}}
    ok = True
    for name, mod in MODS.items():
        system = mod.build(train)
        calls = counter(system)
        own_mode = system.mode
        system.mode = 'off'
        r = {'file_MODE': own_mode, 'off': {}, 'own': {}}
        for ds, songs in first.items():
            c0 = calls[0]
            outs = [bench3.run_song(system, s) for s in songs]
            notes = sum(len(o) for o in outs)
            diff = sum(sum(a != b for a, b in zip(o, q)) + abs(len(o) - len(q)) for o, q in zip(outs, ref[ds]))
            r['off'][ds] = {'notes': notes, 'notes_differing_vs_champion': diff, 'gru_calls': calls[0] - c0}
            ok = ok and diff == 0 and calls[0] - c0 == notes and notes > 0
        del system.mode                       # back to the file's own MODE (class attribute)
        for ds, songs in first.items():
            c0 = calls[0]
            g = dict(notes=0, changed=0, changed_misheard=0, changed_heard_out_of_scale=0, misheard=0,
                     heard_out_of_scale=0, nd_time=0.0, nd_time_champion=0.0, scored_time=0.0,
                     won_misheard=0.0, lost_misheard=0.0, won_heard_right=0.0, lost_heard_right=0.0,
                     won_out_of_scale=0.0, lost_out_of_scale=0.0)
            for s, q in zip(songs, ref[ds]):
                o = bench3.run_song(system, s)
                heard = sorted(bench3.B2.heard(s)['sop'], key=lambda n: n[0])
                true = sorted(s['sop'], key=lambda n: n[0])
                n1, d1 = credit(s, o)
                n0, _ = credit(s, q)
                for i in range(len(o)):
                    mis = heard[i][2] != true[i][2]
                    oos = (heard[i][2] - s['tonic']) % 12 not in MAJOR
                    g['notes'] += 1
                    g['misheard'] += mis
                    g['heard_out_of_scale'] += oos
                    g['nd_time'] += n1[i]
                    g['nd_time_champion'] += n0[i]
                    g['scored_time'] += d1[i]
                    if o[i] != q[i]:
                        g['changed'] += 1
                        g['changed_misheard'] += mis
                        g['changed_heard_out_of_scale'] += oos
                        dlt = n1[i] - n0[i]
                        key = 'misheard' if mis else 'heard_right'
                        g['won_' + key] += max(dlt, 0.0)
                        g['lost_' + key] += max(-dlt, 0.0)
                        if oos:
                            g['won_out_of_scale'] += max(dlt, 0.0)
                            g['lost_out_of_scale'] += max(-dlt, 0.0)
            g['gru_calls'] = calls[0] - c0
            g['music_nd_subset'] = round(g['nd_time'] / g['scored_time'], 5)
            g['music_nd_subset_champion'] = round(g['nd_time_champion'] / g['scored_time'], 5)
            g['delta_subset'] = round((g['nd_time'] - g['nd_time_champion']) / g['scored_time'], 5)
            ok = ok and g['gru_calls'] == g['notes']
            r['own'][ds] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()}
        res['variants'][name] = r
        print(name, 'E5 off identical to r2_mix_v2:',
              all(v['notes_differing_vs_champion'] == 0 for v in r['off'].values()),
              '| own MODE', own_mode, 'delta on subset:', {d: v['delta_subset'] for d, v in r['own'].items()},
              '| changed', {d: v['changed'] for d, v in r['own'].items()}, flush=True)
    res['all_ok'] = ok
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"check_copy_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('ALL_OK', ok, 'secs', res['secs'], 'wrote', path)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
