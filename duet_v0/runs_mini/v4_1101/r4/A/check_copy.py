"""One-off faithfulness check for v4 round 4, implementer A (not a system file, not a scored variant).

(1) r4_cover_fit_v1's class with the forecast switched off (instance attribute ahead = 0 on that one object; class and module
    untouched) must give exactly r3_style_mix_v3's notes through bench3.run_song on the first 40 val songs of each dataset
    (bach has 17), on both mix paths ('beat' and 'cover').
(2) Reproduction: the same object (ahead 0) on ALL val songs: notes identical to r3_style_mix_v3, and music_nd / double
    recomputed exactly as rescore_nd.py equal r3_style_mix_v3's recorded nd json.
Writes runs_mini/v4_1101/r4/A/check_copy_<time>.json (new file, mode 'x'); exits 1 if any output differs.
"""
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V3DIR = os.path.join(ABS, 'futures', 'v3_nolatency')
OUTDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'r4', 'A')
CHND = os.path.join(ABS, 'runs_mini', 'v4_1101', 'nd', 'r3_style_mix_v3', 'results', 'nd_r3_style_mix_v3.json')
sys.path.insert(0, V3DIR)
import bench3  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402
from systems import r3_style_mix_v3, r4_cover_fit_v1  # noqa: E402

N_FIRST = 40
DS = ('bach', 'folk', 'pop', 'pop909')


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


def main():
    t0 = time.time()
    train = bench3.train_sets()
    every = bench3.eval_sets('val')
    first = {ds: songs[:N_FIRST] for ds, songs in every.items()}
    res = {'check': 'r4 A faithfulness: r4_cover_fit_v1 with ahead 0 vs r3_style_mix_v3 (val, tuned on val)',
           'songs_per_dataset_first40': {d: len(v) for d, v in first.items()}, 'paths': {}}
    ok = True
    champ = r3_style_mix_v3.build(train)
    ref = run_all(champ, first)
    mine = r4_cover_fit_v1.build(train)
    for mix in ('beat', 'cover'):
        mine.ahead = 0
        mine.mix = mix
        c = compare(ref, run_all(mine, first))
        res['paths'][mix] = c
        ok = ok and c['identical']
        print('mix', mix, 'ahead 0 identical to r3_style_mix_v3:', c['identical'], f"({c['songs']} songs, {c['notes']} notes, "
              f"{c['notes_differing']} differ)", flush=True)
    del mine.mix                                     # back to the file's own MIX (class attribute)
    del mine.ahead                                   # back to the file's own AHEAD (class attribute)
    own = compare(ref, run_all(mine, first))
    res['own_config_notes_changed_first40'] = own['notes_differing']
    print('own config (AHEAD', r4_cover_fit_v1.AHEAD, 'MIX', r4_cover_fit_v1.MIX, ') changes', own['notes_differing'], 'of',
          own['notes'], 'notes', flush=True)

    ref_all = run_all(champ, every)
    mine.ahead = 0
    mine_all = run_all(mine, every)
    c_all = compare(ref_all, mine_all)
    got = nd(mine_all, every)
    with open(CHND) as fh:
        rec = json.load(fh)
    diff = {d: got[d]['music_nd'] - rec[d]['music_nd'] for d in DS}
    diff['headline_nd'] = got['headline_nd'] - rec['headline_nd']
    ddbl = {d: got[d]['double'] - rec[d]['double'] for d in DS}
    repro = {'all_val_vs_r3_style_mix_v3': c_all, 'nd_recomputed': got, 'nd_recorded_r3_style_mix_v3': rec,
             'music_nd_diff': diff, 'double_diff': ddbl,
             'exact': all(v == 0.0 for v in diff.values()) and all(v == 0.0 for v in ddbl.values())}
    res['reproduction_ahead0_all_val'] = repro
    ok = ok and c_all['identical'] and repro['exact']
    print('reproduction (all val songs): identical', c_all['identical'], f"({c_all['songs']} songs, {c_all['notes']} notes)"
          f" | headline_nd recomputed {got['headline_nd']:.6f} recorded {rec['headline_nd']:.6f} exact {repro['exact']}",
          flush=True)
    res['all_ok'] = ok
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"check_copy_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('ALL_OK', ok, 'secs', res['secs'], 'wrote', path)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
