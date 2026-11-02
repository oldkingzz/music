"""Fresh clashes: how often Duet STARTS a note that is harsh against what she is singing (orchestrator's add-on metric).

Vincent heard the hand-built look-ahead (r0) as clearly worse than reactive although bench_rt's harsh TIME was a little
lower for r0: a newly attacked wrong note is far more audible than an old note ringing on for 60 ms into her new one.
This splits bench_rt's step function (bench_rt.simulate, unchanged) by where each harsh moment comes from:
  fresh_clash  = Duet attacks (starts) a new note while she sings, and at that moment it is harsh (2nd, tritone, 7th)
                 against her TRUE note; counted per 100 of her notes (onset >= J, labelled beat, as bench_rt's notes).
  attack_miss  = Duet attacks a new note while she sings and it is not a hit (bench_rt.hit) at that moment; per 100 notes.
  harsh_fresh  = harsh time inside the first 0.25 beat after each Duet attack during her note (share of counted time).
  harsh_bleed  = the rest of bench_rt's harsh time (a note that was already sounding).
Uses bench_rt.load / simulate / hit / HARSH / J and the same heard melody; val only, tuned on val.

  .venv/bin/python runs_mini/v5rt_1101/tools/attack_metric.py <spec> [--L 0.1] [--seed2] --out <new json>
"""
import argparse
import bisect
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402  (puts the v3 and duet_beat paths in place)

FRESH = 0.25                                             # beats after an attack that count as 'fresh'


def song_counts(song, T, H, L, acc):
    true = sorted(song['sop'], key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    t = song['tonic']
    starts = [x for j, x in enumerate(T) if H[j] >= 0 and (j == 0 or H[j] != H[j - 1])]   # attack times
    for on, off, m in true:
        k0 = int(on)
        if on >= B.J and k0 < len(z) and z[k0] >= 0:
            acc['notes'] += 1
        a0 = bisect.bisect_left(starts, on)
        for x in starts[a0:bisect.bisect_left(starts, off)]:
            k = int(x)
            if x < B.J or k >= len(z) or z[k] < 0:
                continue
            h = B.at(T, H, x)
            acc['attacks'] += 1
            acc['fresh_clash'] += abs(m - h) % 12 in B.HARSH
            acc['attack_miss'] += not B.hit(h, m, t, z[k])
        for k in range(max(B.J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            acc['d'] += ov
            for x0, x1, h in B.pieces(T, H, max(on, k), min(off, k + 1)):
                if not (h >= 0 and abs(m - h) % 12 in B.HARSH):
                    continue
                j = bisect.bisect_right(T, x0) - 1                 # the decision this piece belongs to
                a = T[j]
                while j > 0 and H[j - 1] == H[j]:
                    j -= 1
                    a = T[j]
                fresh_end = max(a, on) + FRESH if a >= on else on    # attacked during her note: fresh for FRESH
                f = max(0.0, min(x1, fresh_end) - x0) if a >= on else 0.0
                acc['harsh_fresh'] += f
                acc['harsh_bleed'] += (x1 - x0) - f


def evaluate(spec, L=0.1, seed2=False):
    heard = B.heard_seed2 if seed2 else B.HEARD1
    B.bench3.B2.heard = heard
    obj = B.load(spec, B.bench3.train_sets())
    rows = {}
    for ds, songs in B.bench3.eval_sets('val').items():
        acc = dict.fromkeys(('notes', 'attacks', 'fresh_clash', 'attack_miss', 'd', 'harsh_fresh', 'harsh_bleed'), 0.0)
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            if isinstance(obj, B.Oracle):
                obj.tell(notes)
            T, H, _, _ = B.simulate(obj, meta, notes, L)
            song_counts(s, T, H, L, acc)
        n = acc['notes'] or 1
        rows[ds] = {'fresh_clash_per100': 100 * acc['fresh_clash'] / n, 'attack_miss_per100': 100 * acc['attack_miss'] / n,
                    'attacks_per100': 100 * acc['attacks'] / n, 'harsh_fresh': acc['harsh_fresh'] / acc['d'],
                    'harsh_bleed': acc['harsh_bleed'] / acc['d'], 'notes': int(acc['notes'])}
    head = {k: sum(r[k] for r in rows.values()) / len(rows) for k in next(iter(rows.values())) if k != 'notes'}
    return {'spec': spec, 'L': L, 'seed': 'seed2' if seed2 else 'seed1', 'split': 'val', 'headline': head, 'rows': rows,
            'fresh_beats': FRESH, 'bench': B.__file__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--L', type=float, default=0.1)
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = evaluate(a.spec, a.L, a.seed2)
    with open(a.out, 'x') as fh:
        json.dump(out, fh, indent=1)
    h = out['headline']
    print(f"{a.spec} L={a.L:g} {out['seed']}: fresh clashes {h['fresh_clash_per100']:.1f} per 100 notes, "
          f"wrong attacks {h['attack_miss_per100']:.1f} per 100, attacks {h['attacks_per100']:.1f} per 100, "
          f"harsh time fresh {h['harsh_fresh']:.4f} bleed {h['harsh_bleed']:.4f}", flush=True)


if __name__ == '__main__':
    main()
