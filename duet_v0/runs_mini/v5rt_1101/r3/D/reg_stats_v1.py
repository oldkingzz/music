"""Register statistics of her melodies (run v5rt_1101, round 3, idea D: register adapter). Read-only on the data.

Anything a system will use is FITTED ON TRAIN ONLY (bench3.train_sets(): the soprano register the systems were trained
on). Val numbers are printed for reference only (to check that a rule is a no-op on bench_rt's songs; tuned on val, this
bundle has no test split). Also the low-voice cases: val Bach - 12 (the live bench's 'sung_low'), Twinkle and Twinkle
- 12, and the heard notes of the real low voice v11 (from the phase-1 reactive phone_v11 run JSON).

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/D/reg_stats_v1.py --out runs_mini/v5rt_1101/r3/D/<new>.json
"""
import argparse
import json
import os
import sys

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402
from songs import twinkle  # noqa: E402  (duet_beat on sys.path via bench3)

V11 = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'live_bench', 'L_reactive_r2_mix_v2_phone_v11.json')
NS = (1, 2, 3, 4, 5, 6, 8)
Q = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def q(a):
    a = np.asarray(a, float)
    return {str(p): round(float(np.percentile(a, p)), 2) for p in Q} if len(a) else None


def song_rows(songs):
    rows = []
    for s in songs:
        sop = sorted(s['sop'], key=lambda n: n[0])
        p = np.array([m for _, _, m in sop], float)
        on = np.array([a for a, _, _ in sop], float)
        d = np.array([b - a for a, b, _ in sop], float)
        r = {'id': s.get('id'), 'n': len(p), 'med': float(np.median(p)), 'tmed': float(np.average(p, weights=d)),
             'lo': float(p.min()), 'hi': float(p.max()), 'first': [float(np.median(p[:n])) for n in NS],
             'on_n': [float(on[min(n, len(on)) - 1]) for n in NS]}
        rows.append(r)
    return rows


def summarize(rows, allp):
    return {'songs': len(rows), 'note_pitch': q(allp), 'song_median': q([r['med'] for r in rows]),
            'song_min': q([r['lo'] for r in rows]), 'song_max': q([r['hi'] for r in rows]),
            'first_n_median': {str(n): q([r['first'][j] for r in rows]) for j, n in enumerate(NS)},
            'onset_of_note_n_beats': {str(n): q([r['on_n'][j] for r in rows]) for j, n in enumerate(NS)}}


def shifted(med, lo_up):
    """+12 per octave the median lies below lo_up (0 when med >= lo_up)."""
    k = 0
    while med + 12 * k < lo_up:
        k += 1
    return 12 * k


def rule_table(groups):
    out = {}
    for lo_up in (56, 58, 59, 60, 61, 62, 63, 64):
        for j, n in enumerate(NS):
            row = {}
            for g, rows in groups.items():
                sh = [shifted(r['first'][j], lo_up) for r in rows]
                row[g] = round(float(np.mean([x != 0 for x in sh])), 4)
            out[f'lo_up={lo_up} n={n}'] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if os.path.exists(path):
        sys.exit('refusing to overwrite ' + path)
    res = {'note': 'TRAIN = what systems may fit on; val / low cases for reference only (tuned on val, no test split)'}
    groups = {}
    tr = bench3.train_sets()
    allp_tr = []
    res['train'] = {}
    for ds, songs in tr.items():
        rows = song_rows(songs)
        allp = [m for s in songs for _, _, m in s['sop']]
        allp_tr += allp
        res['train'][ds] = summarize(rows, allp)
        groups['train_' + ds] = rows
    res['train_pooled_note_pitch'] = q(allp_tr)
    va = bench3.eval_sets('val')
    res['val'] = {}
    for ds, songs in va.items():
        rows = song_rows(songs)
        res['val'][ds] = summarize(rows, [m for s in songs for _, _, m in s['sop']])
        groups['val_' + ds] = rows
    low = [dict(s, sop=[[on, off, m - 12] for on, off, m in s['sop']]) for s in va['bach']]
    rows = song_rows(low)
    res['val_bach_minus12'] = summarize(rows, [m for s in low for _, _, m in s['sop']])
    groups['val_bach_minus12'] = rows
    tw = twinkle()
    for name, sh in (('twinkle', 0), ('twinkle_minus12', -12)):
        s = dict(tw, sop=[[on, off, m + sh] for on, off, m in tw['sop']])
        rows = song_rows([s])
        res[name] = rows[0]
        groups[name] = rows
    with open(V11) as fh:
        v = json.load(fh)
    hp = [int(e[1]) for e in v['phone']['answered']]
    res['v11_heard'] = {'pitches': hp, 'median': float(np.median(hp)),
                        'first_n_median': {str(n): float(np.median(hp[:n])) for n in NS}, 'min': min(hp), 'max': max(hp)}
    groups['v11'] = [{'first': [float(np.median(hp[:n])) for n in NS]}]
    res['rule_share_shifted'] = rule_table(groups)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('train pooled note pitch', res['train_pooled_note_pitch'])
    for ds in res['train']:
        t = res['train'][ds]
        print(f"train {ds:7s} songs {t['songs']:5d} note {t['note_pitch']} ")
        print(f"      song_median {t['song_median']}")
        print(f"      first1 {t['first_n_median']['1']}  first3 {t['first_n_median']['3']}")
    for ds in res['val']:
        t = res['val'][ds]
        print(f"val   {ds:7s} songs {t['songs']:5d} song_median {t['song_median']}")
        print(f"      first1 {t['first_n_median']['1']}  first3 {t['first_n_median']['3']}  onset note3 "
              f"{t['onset_of_note_n_beats']['3']}  note4 {t['onset_of_note_n_beats']['4']}")
    t = res['val_bach_minus12']
    print('val bach - 12 song_median', t['song_median'], 'first1', t['first_n_median']['1'], 'first3',
          t['first_n_median']['3'])
    print('twinkle', res['twinkle']['med'], res['twinkle']['first'], 'twinkle-12', res['twinkle_minus12']['first'])
    print('v11', res['v11_heard'])
    for k, row in res['rule_share_shifted'].items():
        print(k, row)
    print('wrote', path)


if __name__ == '__main__':
    main()
