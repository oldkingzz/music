"""explore_seq_E_v1 (run v5rt_1101, round 3, idea E): read-only look at the TRAIN answered-note sequences
(models/E_tracker_seq_v1.json, from train_tracker_seq_E_v1.py) to choose the features of a tracker-aware hear.
Prints tables only; writes nothing. TRAIN renders only.

  .venv/bin/python runs_mini/v5rt_1101/r3/E/explore_seq_E_v1.py
"""
import json
from collections import Counter, defaultdict

SEQ = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/models/E_tracker_seq_v1.json'


def gclass(on):
    if abs(on * 4 - round(on * 4)) > 1e-6:
        return 'raw'
    f = round((on % 1) * 4) % 4
    return ('beat', 'q', 'half', 'q')[f]


def dtb(dt):
    return 0 if dt < 0.35 else 1 if dt < 0.75 else 2 if dt < 1.25 else 3


def main():
    d = json.load(open(SEQ))
    F = d['fields']
    ix = {k: j for j, k in enumerate(F)}
    seqs = defaultdict(list)
    for r in d['rows']:
        seqs[(r[0], r[1])].append(r)
    recs = []
    for key, rows in seqs.items():
        rows.sort(key=lambda r: r[ix['i']])
        for j, r in enumerate(rows):
            if r[ix['k']] < 0:
                continue
            h = r[ix['heard']]
            e = r[ix['true_midi']] - h
            first = j == 0
            p1 = rows[j - 1][ix['heard']] if j >= 1 else None
            p2 = rows[j - 2][ix['heard']] if j >= 2 else None
            c1 = None if p1 is None else h - p1
            c2 = None if p2 is None else p1 - p2
            dt = None if first else r[ix['onset']] - rows[j - 1][ix['onset']]
            recs.append({'voice': key[1], 'e': e, 'first': first, 'c1': c1, 'c2': c2, 'g': gclass(r[ix['onset']]),
                         'dt': dt, 'frag': r[ix['frag']], 'pd': r[ix['prev_dur']],
                         'prev_e': (rows[j - 1][ix['true_midi']] - p1) if (j >= 1 and rows[j - 1][ix['k']] >= 0) else None,
                         'k': r[ix['k']], 'pk': rows[j - 1][ix['k']] if j >= 1 else None})
    print('records', len(recs), 'songs', d['songs'])
    for v in ('sung', 'sung_low', None):
        rr = [x for x in recs if (v is None or x['voice'] == v)]
        n = len(rr)
        print(f"\n=== voice {v or 'pooled'}: {n} answered; heard right {sum(x['e'] == 0 for x in rr) / n:.3f}; "
              f"fragments {sum(bool(x['frag']) for x in rr) / n:.3f}")
        # by c1
        byc = defaultdict(list)
        for x in rr:
            if x['c1'] is not None:
                byc[max(-8, min(8, x['c1']))].append(x)
        print(' heard interval c1: n, P(right), P(frag), top e')
        for c in sorted(byc):
            g = byc[c]
            ec = Counter(x['e'] for x in g)
            print(f"  c1 {c:+3d}: n {len(g):5d} right {ec[0] / len(g):.3f} frag {sum(bool(x['frag']) for x in g) / len(g):.3f} "
                  + ' '.join(f"e{k:+d}:{v / len(g):.3f}" for k, v in sorted(ec.items(), key=lambda t: -t[1])[:5]))
        # fragments by onset class and dt
        print(' onset class g x dt bucket: n, P(frag), P(right)')
        bg = defaultdict(list)
        for x in rr:
            if x['dt'] is not None:
                bg[(x['g'], dtb(x['dt']))].append(x)
        for k in sorted(bg):
            g = bg[k]
            print(f"  {k}: n {len(g):5d} frag {sum(bool(x['frag']) for x in g) / len(g):.3f} "
                  f"right {sum(x['e'] == 0 for x in g) / len(g):.3f}")
        print(' |c1| == 1 by onset class: n, P(frag), P(right), P(e == -c1)')
        b1 = defaultdict(list)
        for x in rr:
            if x['c1'] is not None and abs(x['c1']) == 1:
                b1[(x['g'], dtb(x['dt']))].append(x)
        for k in sorted(b1):
            g = b1[k]
            print(f"  {k}: n {len(g):5d} frag {sum(bool(x['frag']) for x in g) / len(g):.3f} "
                  f"right {sum(x['e'] == 0 for x in g) / len(g):.3f} back {sum(x['e'] == -x['c1'] for x in g) / len(g):.3f}")
        print(' fragments: e distribution, by c1')
        fr = [x for x in rr if x['frag']]
        ec = Counter(x['e'] for x in fr)
        print('  all fragments', len(fr), {k: round(v / len(fr), 3) for k, v in sorted(ec.items())})
        print('  fragment heard right (a correction) share', round(ec[0] / max(len(fr), 1), 3))
        print(' first answered notes: e', Counter(x['e'] for x in rr if x['first']))
        print(' previous answered note misheard (prev_e != 0, same true note not counted): next note e distribution')
        pm = [x for x in rr if x['prev_e'] not in (None, 0) and not x['frag']]
        print('  n', len(pm), {k: round(v / max(len(pm), 1), 3) for k, v in sorted(Counter(x['e'] for x in pm).items())})
        print(' by c2 leap (|p1 - p2| >= 3) and c1: P(right)')
        bl = defaultdict(list)
        for x in rr:
            if x['c2'] is not None and x['c1'] is not None:
                bl[(abs(x['c2']) >= 3, max(-4, min(4, x['c1'])))].append(x)
        for k in sorted(bl):
            g = bl[k]
            print(f"  leap_before {k[0]!s:5s} c1 {k[1]:+d}: n {len(g):5d} right {sum(x['e'] == 0 for x in g) / len(g):.3f}")


if __name__ == '__main__':
    main()
