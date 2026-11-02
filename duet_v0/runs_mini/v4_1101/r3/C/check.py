"""One-off faithfulness check for v4 round 3, implementer C (r3_style_mix, idea E2 x E3); not a system, not scored.

(1) First 40 val songs of each dataset (bach has 17): r3_style_mix_v<k> with a = 0 (instance attribute on that one object;
    class and module untouched) must sing exactly the notes of the r2_style_hmm file it subclasses, with a = 1 exactly
    r1_gru_chooser_v1's; the beliefs the chooser uses are recorded and compared; the GRU ensemble must advance exactly
    once per note and belief(k) must be called exactly once per note.
(2) Reproduction on ALL val songs: at a = 0 the notes equal the r2_style_hmm file's and music_nd / double recomputed exactly
    as rescore_nd.py equal its recorded nd json; at a = 1 the same against r1_gru_chooser_v1.
(3) Sanity: with the file's own ALPHA the notes do differ from both ends (count); m1_filter_v5lo.P is unchanged.
Val only, tuned on val. Writes runs_mini/v4_1101/r3/C/check_<time>.json (mode 'x'); exits 1 if anything differs.
"""
import json
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V3DIR = os.path.join(ABS, 'futures', 'v3_nolatency')
OUTDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'r3', 'C')
NDDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'nd')
sys.path.insert(0, V3DIR)
import bench3  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402
from systems import m1_filter_v5lo, r1_gru_chooser_v1  # noqa: E402
from systems import r2_style_hmm_v1, r2_style_hmm_v3, r3_style_mix_v1, r3_style_mix_v2  # noqa: E402

N_FIRST = 40
DS = ('bach', 'folk', 'pop', 'pop909')
PAIRS = (('r3_style_mix_v1', r3_style_mix_v1, 'r2_style_hmm_v1', r2_style_hmm_v1),
         ('r3_style_mix_v2', r3_style_mix_v2, 'r2_style_hmm_v3', r2_style_hmm_v3))


def recorder(obj):
    """Shadow obj.belief on the instance: every belief the chooser uses is logged (normalized, float64)."""
    method = type(obj).belief
    log = []

    def rec(k):
        b = method(obj, k)
        x = np.asarray(b, dtype=np.float64)
        log.append(x / x.sum())
        return b
    obj.belief = rec
    return log


def counter(mix):
    """Shadow mix.ens.chord_probs on the instance to count GRU advances."""
    orig = mix.ens.chord_probs
    n = [0]

    def counted(onset, midi, prev_dur):
        n[0] += 1
        return orig(onset, midi, prev_dur)
    mix.ens.chord_probs = counted
    return n


def run(system, songs, log=None):
    """Notes per song, and (if log) the beliefs per song."""
    notes, bels = [], []
    for s in songs:
        if log is not None:
            del log[:]
        notes.append(bench3.run_song(system, s))
        if log is not None:
            bels.append(list(log))
    return notes, bels


def nd(outs, sets):
    """music_nd and double per dataset, computed exactly as rescore_nd.py does (same loops, same order)."""
    out = {}
    for ds, songs in sets.items():
        num = den = dbl = 0.0
        for s, hs in zip(songs, outs[ds]):
            true = sorted(s['sop'], key=lambda n: n[0])
            z = [b['z'] for b in s['beats']]
            t = s['tonic']
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
    with open(os.path.join(NDDIR, name, 'results', f'nd_{name}.json')) as fh:
        return json.load(fh)


def same_nd(got, rec):
    diff = {d: got[d]['music_nd'] - rec[d]['music_nd'] for d in DS}
    diff['headline_nd'] = got['headline_nd'] - rec['headline_nd']
    ddbl = {d: got[d]['double'] - rec[d]['double'] for d in DS}
    return {'music_nd_diff': diff, 'double_diff': ddbl,
            'exact': all(v == 0.0 for v in diff.values()) and all(v == 0.0 for v in ddbl.values())}


def n_diff(a, b):
    return sum(sum(p != q for p, q in zip(x, y)) + abs(len(x) - len(y)) for x, y in zip(a, b))


def max_bdiff(a, b):
    m = 0.0
    for x, y in zip(a, b):
        if len(x) != len(y):
            return float('inf')
        if x:
            m = max(m, float(np.abs(np.array(x) - np.array(y)).max()))
    return m


def main():
    t0 = time.time()
    p_before = dict(m1_filter_v5lo.P)
    train = bench3.train_sets()
    every = bench3.eval_sets('val')
    first = {ds: songs[:N_FIRST] for ds, songs in every.items()}
    res = {'check': 'r3 C faithfulness: r3_style_mix_v<k> at a = 0 vs its r2_style_hmm file, at a = 1 vs r1_gru_chooser_v1 '
                    '(val, tuned on val)',
           'songs_per_dataset_first40': {d: len(v) for d, v in first.items()},
           'songs_per_dataset_all': {d: len(v) for d, v in every.items()}, 'variants': {}}
    ok = True
    r1 = r1_gru_chooser_v1.build(train)
    L_r1 = recorder(r1)
    r1_first = {ds: run(r1, first[ds], L_r1) for ds in DS}
    r1_all = {ds: run(r1, every[ds])[0] for ds in DS}
    for name, mod, sname, smod in PAIRS:
        ref = smod.build(train)
        L_ref = recorder(ref)
        mix = mod.build(train)
        L_mix = recorder(mix)
        calls = counter(mix)
        v = {'file_ALPHA': mod.ALPHA, 'subclasses': sname, 'is_subclass': issubclass(type(mix), smod.StyleFilter),
             'first40': {}}
        for ds in DS:
            songs = first[ds]
            n_notes = sum(len(x) for x in r1_first[ds][0])
            ref_notes, ref_bels = run(ref, songs, L_ref)
            mix.a = 0.0
            c0 = calls[0]
            a0_notes, a0_bels = run(mix, songs, L_mix)
            gru_a0 = calls[0] - c0
            mix.a = 1.0
            a1_notes, a1_bels = run(mix, songs, L_mix)
            mix.a = mod.ALPHA                          # back to the file's own ALPHA
            own_notes, _ = run(mix, songs, L_mix)
            r = {'songs': len(songs), 'notes': n_notes,
                 'a0_notes_differing_vs_style': n_diff(a0_notes, ref_notes),
                 'a1_notes_differing_vs_r1': n_diff(a1_notes, r1_first[ds][0]),
                 'a0_max_belief_diff_vs_style': max_bdiff(a0_bels, ref_bels),
                 'a1_max_belief_diff_vs_r1': max_bdiff(a1_bels, r1_first[ds][1]),
                 'a0_belief_calls': sum(len(x) for x in a0_bels), 'a0_gru_calls': gru_a0,
                 'own_alpha_notes_differing_vs_style': n_diff(own_notes, ref_notes),
                 'own_alpha_notes_differing_vs_r1': n_diff(own_notes, r1_first[ds][0])}
            r['ok'] = (r['a0_notes_differing_vs_style'] == 0 and r['a1_notes_differing_vs_r1'] == 0
                       and r['a0_belief_calls'] == n_notes and r['a0_gru_calls'] == n_notes
                       and r['a0_max_belief_diff_vs_style'] < 1e-12 and r['a1_max_belief_diff_vs_r1'] < 1e-12)
            ok = ok and r['ok']
            v['first40'][ds] = r
            print(name, ds, json.dumps(r), flush=True)
        # (2) reproduction on all val songs
        mix.a = 0.0
        a0_all = {ds: run(mix, every[ds])[0] for ds in DS}
        mix.a = 1.0
        a1_all = {ds: run(mix, every[ds])[0] for ds in DS}
        mix.a = mod.ALPHA
        ref_all = {ds: run(ref, every[ds])[0] for ds in DS}
        rep0 = same_nd(nd(a0_all, every), recorded(sname))
        rep0['notes_differing_vs_style'] = sum(n_diff(a0_all[ds], ref_all[ds]) for ds in DS)
        rep1 = same_nd(nd(a1_all, every), recorded('r1_gru_chooser_v1'))
        rep1['notes_differing_vs_r1'] = sum(n_diff(a1_all[ds], r1_all[ds]) for ds in DS)
        v['all_a0_vs_' + sname] = rep0
        v['all_a1_vs_r1_gru_chooser_v1'] = rep1
        ok = ok and rep0['exact'] and rep1['exact'] and rep0['notes_differing_vs_style'] == 0 \
            and rep1['notes_differing_vs_r1'] == 0 and v['is_subclass']
        print(name, 'all val: a0 exact', rep0['exact'], rep0['notes_differing_vs_style'], '| a1 exact', rep1['exact'],
              rep1['notes_differing_vs_r1'], flush=True)
        res['variants'][name] = v
    res['m1_P_unchanged'] = dict(m1_filter_v5lo.P) == p_before
    ok = ok and res['m1_P_unchanged']
    res['all_ok'] = ok
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"check_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('ALL_OK', ok, 'secs', res['secs'], 'wrote', path)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
