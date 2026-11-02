"""v5rt_1101 phase 1, read-only data check for bench_rt (val split only): do her true notes overlap, share onsets,
or last less than L; how often does onset + L land exactly on a plan point; are onset beats labelled. Prints only."""
import os
import sys
from collections import Counter

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402

for ds, songs in bench3.eval_sets('val').items():
    n = ov = dup = nonpos = neg_on = offgrid = near_half = lab = unlab = kmis = beyond = 0
    ovt = maxov = 0.0
    short, ties = Counter(), Counter()
    Ks, ns = [], []
    for s in songs:
        notes = sorted(s['sop'], key=lambda x: x[0])
        z = [b['z'] for b in s['beats']]
        kmis += s['K'] != len(z)
        Ks.append(s['K'])
        ns.append(len(notes))
        for i, (on, off, m) in enumerate(notes):
            n += 1
            neg_on += on < 0
            nonpos += off - on <= 1e-9
            for L in (0.05, 0.1, 0.15):
                short[L] += off - on < L - 1e-12
                ties[L] += (on + L) / 0.5 == int((on + L) / 0.5)
            offgrid += abs(on * 48 - round(on * 48)) > 1e-6
            x = on / 0.5
            near_half += abs(x - round(x)) < 1e-6 and x != round(x)
            if i + 1 < len(notes):
                nx = notes[i + 1][0]
                dup += nx - on <= 1e-12
                if off > nx + 1e-9:
                    ov += 1
                    ovt += off - nx
                    maxov = max(maxov, off - nx)
            k = int(on)
            if on >= bench3.J and k < len(z):
                if z[k] >= 0:
                    lab += 1
                else:
                    unlab += 1
            beyond += k >= len(z)
    print(f'{ds}: songs {len(songs)} notes {n} (per song mean {n / len(songs):.0f}, max {max(ns)}), K mean {sum(Ks) / len(Ks):.0f} max {max(Ks)}, '
          f'K!=len(beats) {kmis}; overlaps {ov} (total {ovt:.3f} beats, max {maxov:.3f}); same-onset pairs {dup}; '
          f'dur<=0 {nonpos}; onset<0 {neg_on}; onsets off the 1/48 grid {offgrid}; onsets within 1e-6 of a half beat but not equal {near_half}; '
          f'notes shorter than L {dict(short)}; onset+L exactly on a half beat {dict(ties)}; '
          f'onset beat (onset>=J) labelled {lab} unlabelled {unlab}; onset beyond beats {beyond}', flush=True)
