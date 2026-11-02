"""v5rt_1101 phase 1 DIAGNOSTIC (not part of the frozen bench, not a keep-rule metric). bench_rt's harsh counts a
dissonance the same whether Duet HOLDS its previous note into her new note (reactive: one late switch per note) or
ATTACKS a new note together with her onset and then switches again by the hear point (a 'stumble': the
wrong-then-corrected start Vincent heard from live_predict). This splits note starts and first-L harsh time by that
difference, using the frozen bench_rt's own simulate() and pieces() on val (seed1). Val only.

  python diag_attacks_v1.py <spec> <L> <new .json under runs_mini/v5rt_1101/diag/>
"""
import bisect
import importlib.util
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
_s = importlib.util.spec_from_file_location('bench_rt', os.path.join(ABS, 'futures', 'v5_rt', 'bench_rt.py'))
B = importlib.util.module_from_spec(_s)
_s.loader.exec_module(B)
import bench3  # noqa: E402  (the bench's own module)

W = 0.05     # beats (30 ms): a Duet attack this close before her onset, or at it, counts as attacked WITH her onset


def changes(T, H):
    """(time, new value) where Duet's sounding value really changes (silence = -1)."""
    out, cur = [], -1
    for t, h in zip(T, H):
        if h != cur:
            out.append((t, h))
            cur = h
    return out


def song_diag(song, T, H, L, acc):
    true = sorted(song['sop'], key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    tonic = song['tonic']
    ch = changes(T, H)
    ct = [c[0] for c in ch]
    for on, off, m in true:
        k0 = int(on)
        if on >= B.J and k0 < len(z) and z[k0] >= 0:                      # bench_rt's per-note set
            lo, hi = bisect.bisect_left(ct, on - W), bisect.bisect_right(ct, on)
            att_on = any(ch[j][1] >= 0 for j in range(lo, hi))               # a note attacked with her onset
            j2 = bisect.bisect_right(ct, on + L)
            chg_hear = j2 > hi                                               # a change in (onset, onset + L]
            n_att = sum(ch[j][1] >= 0 for j in range(lo, j2))
            v0 = B.at(T, H, on)
            harsh0 = v0 >= 0 and abs(m - v0) % 12 in B.HARSH
            acc['notes'] += 1
            acc['attacks'] += n_att
            acc['stumble'] += att_on and chg_hear
            acc['attack_harsh'] += att_on and harsh0
            acc['held_harsh'] += (not att_on) and harsh0
        e = on + L
        for k in range(max(B.J, int(on)), int(off) + 1):                    # bench_rt's counted first-L time
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            for x0, x1, h in B.pieces(T, H, max(on, k), min(off, k + 1)):
                f = min(x1, e) - x0
                if f <= 0:
                    continue
                j = bisect.bisect_right(ct, x0) - 1
                fresh = j >= 0 and ct[j] >= on - W
                bad = h >= 0 and abs(m - h) % 12 in B.HARSH
                acc['first'] += f
                acc['harsh_first_fresh'] += f * (bad and fresh)
                acc['harsh_first_held'] += f * (bad and not fresh)


def main():
    spec, L, path = sys.argv[1], float(sys.argv[2]), os.path.abspath(sys.argv[3])
    if not path.startswith(os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'diag') + os.sep) or os.path.exists(path):
        sys.exit('out must be a new file under runs_mini/v5rt_1101/diag/')
    obj = B.load(spec, bench3.train_sets())
    rows = {}
    for ds, songs in bench3.eval_sets('val').items():
        acc = dict.fromkeys(('notes', 'attacks', 'stumble', 'attack_harsh', 'held_harsh', 'first',
                             'harsh_first_fresh', 'harsh_first_held'), 0.0)
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, _, _ = B.simulate(obj, meta, notes, L)
            song_diag(s, T, H, L, acc)
        n, f = acc['notes'], acc['first']
        rows[ds] = {'attacks_per_note': acc['attacks'] / n, 'stumble': acc['stumble'] / n,
                    'attack_harsh': acc['attack_harsh'] / n, 'held_harsh': acc['held_harsh'] / n,
                    'harsh_first_fresh': acc['harsh_first_fresh'] / f, 'harsh_first_held': acc['harsh_first_held'] / f,
                    'notes': int(n)}
    head = {k: sum(r[k] for r in rows.values()) / len(rows) for k in next(iter(rows.values())) if k != 'notes'}
    out = {'spec': spec, 'L': L, 'seed': 'seed1', 'split': 'val', 'W_beats': W, 'bench_sha256': None,
           'headline': head, 'rows': rows,
           'definitions': {
               'attacks_per_note': 'Duet note attacks (changes to a note >= 0) in [onset - W, onset + L], per her note',
               'stumble': 'share of her notes where Duet attacks a note in [onset - W, onset] AND changes again in '
                          '(onset, onset + L]: a wrong-then-corrected start',
               'attack_harsh': 'share of her notes where the note sounding just after her onset was attacked in '
                               '[onset - W, onset] and is a 2nd/tritone/7th against her true note',
               'held_harsh': 'share of her notes where the note sounding just after her onset was held over from '
                             'before onset - W and is a 2nd/tritone/7th against her true note',
               'harsh_first_fresh / harsh_first_held': "bench_rt's harsh_first split by whether the sounding note "
                                                       'was attacked with her onset (fresh) or held over'}}
    with open(B.__file__, 'rb') as fh:
        import hashlib
        out['bench_sha256'] = hashlib.sha256(fh.read()).hexdigest()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1)
    print(f'{spec} L={L:g}: ' + ' '.join(f'{k} {v:.4f}' for k, v in head.items()) + ' | per dataset stumble '
          + ' '.join(f"{ds} {r['stumble']:.3f}" for ds, r in rows.items()), flush=True)


if __name__ == '__main__':
    main()
