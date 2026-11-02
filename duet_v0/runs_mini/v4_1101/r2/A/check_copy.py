"""One-off faithfulness check for v4 round 2, implementer A (not a system file, not a scored variant).

(1) Spec check: each r2_gru_chooser_v<k> (k = 1, 2, 3) with its leap penalty set back to 0.4 (an instance attribute
    w_big on that one object; class and module untouched) must give exactly r1_gru_chooser_v1's notes, through
    bench3.run_song, on the first 40 val songs of each dataset (bach has 17).
(2) Reproduction: r2_gru_chooser_v1 with w_big 0.4 on ALL val songs: notes identical to r1_gru_chooser_v1, and music_nd /
    double recomputed exactly as rescore_nd.py equal r1_gru_chooser_v1's recorded nd json.
(3) Sanity: with each file's own W_BIG the notes do change (count), and bench3's 'big' share on the 40-song subset.
Writes runs_mini/v4_1101/r2/A/check_copy_<time>.json (new file, mode 'x'); exits 1 if any output differs.
"""
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V3DIR = os.path.join(ABS, 'futures', 'v3_nolatency')
OUTDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'r2', 'A')
R1ND = os.path.join(ABS, 'runs_mini', 'v4_1101', 'nd', 'r1_gru_chooser_v1', 'results', 'nd_r1_gru_chooser_v1.json')
sys.path.insert(0, V3DIR)
import bench3  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402
from systems import m1_filter_v5lo, r1_gru_chooser_v1  # noqa: E402
from systems import r2_gru_chooser_v1, r2_gru_chooser_v2, r2_gru_chooser_v3  # noqa: E402

N_FIRST = 40
DS = ('bach', 'folk', 'pop', 'pop909')
VARIANTS = (('r2_gru_chooser_v1', r2_gru_chooser_v1), ('r2_gru_chooser_v2', r2_gru_chooser_v2),
            ('r2_gru_chooser_v3', r2_gru_chooser_v3))


def run_all(system, sets):
    return {ds: [bench3.run_song(system, s) for s in songs] for ds, songs in sets.items()}


def compare(a, b):
    songs = notes = bad_songs = bad_notes = 0
    first = []
    for ds in a:
        for j, (x, y) in enumerate(zip(a[ds], b[ds])):
            songs += 1
            notes += len(x)
            if x != y:
                bad_songs += 1
                bad_notes += sum(p != q for p, q in zip(x, y)) + abs(len(x) - len(y))
                if len(first) < 5:
                    first.append({'ds': ds, 'song_index': j})
    return {'songs': songs, 'notes': notes, 'songs_differing': bad_songs, 'notes_differing': bad_notes,
            'identical': bad_songs == 0 and songs > 0, 'first_differences': first}


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


def big_share(sets, outs):
    res = {}
    for ds, songs in sets.items():
        big = moves = 0
        for s, hs in zip(songs, outs[ds]):
            a = bench3.song_stats(s, hs)
            big += a['big']
            moves += a['moves']
        res[ds] = round(big / max(moves, 1), 5)
    return res


def main():
    t0 = time.time()
    train = bench3.train_sets()
    every = bench3.eval_sets('val')
    first = {ds: songs[:N_FIRST] for ds, songs in every.items()}
    res = {'check': 'r2 A faithfulness: r2_gru_chooser_v<k> with w_big 0.4 vs r1_gru_chooser_v1 (val, tuned on val)',
           'm1_P_w_big': m1_filter_v5lo.P['w_big'], 'songs_per_dataset_first40': {d: len(v) for d, v in first.items()},
           'variants': {}}
    ok = True

    # (1) spec check on the first 40 songs per dataset
    ref_sys = r1_gru_chooser_v1.build(train)
    ref = run_all(ref_sys, first)
    res['r1_big_share_first40'] = big_share(first, ref)
    for name, mod in VARIANTS:
        system = mod.build(train)
        system.w_big = 0.4
        c = compare(ref, run_all(system, first))
        del system.w_big                      # back to the file's own W_BIG (class attribute)
        own = run_all(system, first)
        res['variants'][name] = {'file_W_BIG': mod.W_BIG, 'class_w_big': type(system).w_big,
                                 'w_big_0.4_vs_r1': c,
                                 'own_W_BIG_notes_changed_vs_r1': compare(ref, own)['notes_differing'],
                                 'own_W_BIG_big_share_first40': big_share(first, own)}
        ok = ok and c['identical']
        print(name, 'W_BIG', mod.W_BIG, '| with 0.4 identical to r1:', c['identical'], f"({c['songs']} songs, "
              f"{c['notes']} notes, {c['notes_differing']} differ) | own W_BIG changes "
              f"{res['variants'][name]['own_W_BIG_notes_changed_vs_r1']} notes", flush=True)

    # (2) reproduction on all val songs
    ref_all = run_all(ref_sys, every)
    v1 = r2_gru_chooser_v1.build(train)
    v1.w_big = 0.4
    mine_all = run_all(v1, every)
    c_all = compare(ref_all, mine_all)
    got = nd(mine_all, every)
    with open(R1ND) as fh:
        rec = json.load(fh)
    diff = {d: got[d]['music_nd'] - rec[d]['music_nd'] for d in DS}
    diff['headline_nd'] = got['headline_nd'] - rec['headline_nd']
    ddbl = {d: got[d]['double'] - rec[d]['double'] for d in DS}
    repro = {'all_val_vs_r1': c_all, 'nd_recomputed': got, 'nd_recorded_r1': rec, 'music_nd_diff': diff,
             'double_diff': ddbl, 'exact': all(v == 0.0 for v in diff.values()) and all(v == 0.0 for v in ddbl.values())}
    res['reproduction_v1_w_big_0.4'] = repro
    ok = ok and c_all['identical'] and repro['exact']
    print('reproduction (all val songs): identical', c_all['identical'], f"({c_all['songs']} songs, "
          f"{c_all['notes']} notes) | headline_nd recomputed {got['headline_nd']:.6f} recorded "
          f"{rec['headline_nd']:.6f} exact {repro['exact']}", flush=True)

    res['all_ok'] = ok
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"check_copy_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('ALL_OK', ok, 'secs', res['secs'], 'wrote', path)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
