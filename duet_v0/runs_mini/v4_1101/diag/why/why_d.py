"""why_d: is there anything to lock on to? val only. (1) chord-tone coverage of each TRUE chord span by her TRUE melody:
is the labelled chord the (unique) triad that covers most of her sung time? (2) 'perfect lock-on' ceiling: m1's evidence
accumulated over the TRUE span (oracle boundaries) with the chord-frequency prior, by note index; (3) how often m1's
argmax changes between consecutive notes of one true span (flip rate); (4) the best causal knobs combined.
Writes why_d_v1.json (mode 'x')."""
import sys
import time

import numpy as np

from why_b import partial_v
from why_c import AccFilter
from why_lib import CT, DS, J, S, FilterV, ModelV, bench3, line, m1, prep, run, save, sticky

NAME = 'why_d_v1'
ORDER = (0, 4, 3, 5, 1, 2, 6)


def main():
    t0 = time.time()
    train = bench3.train_sets()
    base = ModelV(train)
    dsm = {ds: ModelV(train, only={ds}) for ds in DS}
    tot = sum(dsm[ds].Acount.sum((0, 1)) for ds in DS)
    lprior = np.log((tot + 1) / (tot + 1).sum())
    data = prep()
    out = {'what': __doc__.strip().split('\n')[0], 'notes': 'val only (no test split)', 'datasets': {}}
    for ds, L in data.items():
        a = dict(spans=[0] * 4, cons=[0] * 4, uniq=[0] * 4, tied=[0.0] * 4, tmpl_ok=[0] * 4,
                 lock=[[0, 0] for _ in range(4)], flips=[[0, 0], [0, 0]])
        f_base, f_st = FilterV(base), FilterV(base)
        st = sticky(base.A, 0.85)
        for d in L:
            z, t, span = d['z'], d['t'], d['span']
            # spans and the notes that start in them (TRUE melody)
            notes_in = {}
            for i, (on, off, m) in enumerate(d['true']):
                k0 = int(on)
                if k0 < len(z) and z[k0] >= 0:
                    notes_in.setdefault(span[k0], []).append(i)
            start = {}
            for k in range(len(z)):
                if span[k] is not None and span[k] not in start:
                    start[span[k]] = k
            for sid, idx in notes_in.items():
                b0 = start[sid]
                if b0 < J:
                    continue
                b1 = b0
                while b1 + 1 < len(z) and span[b1 + 1] == sid:
                    b1 += 1
                cov = np.zeros(S)
                tt = 0.0
                for (on, off, m) in d['trel']:
                    ov = min(off, b1 + 1) - max(on, b0)
                    if ov > 1e-9:
                        tt += ov
                        cov += ov * np.array([float(m in CT[c]) for c in range(S)])
                if tt <= 0:
                    continue
                q = min(len(idx), 4) - 1
                top = cov.max()
                tied = [c for c in range(S) if cov[c] >= top - 1e-9]
                zt = z[b0]
                a['spans'][q] += 1
                a['cons'][q] += int(zt in tied)
                a['uniq'][q] += int(tied == [zt])
                a['tied'][q] += len(tied)
                a['tmpl_ok'][q] += int([c for c in ORDER if c in tied][0] == zt)
            # perfect lock-on ceiling: heard evidence over the TRUE span so far + chord prior, m1's lam
            hn = [[on, off, (m - t) % 12] for on, off, m in d['heard']]
            oh = m1.beat_obs(d['hrel'], len(d['pos']))
            seen = {}
            for i, (on, off, m) in enumerate(d['true']):
                k0 = int(on)
                z0 = z[k0] if k0 < len(z) else -1
                if z0 < 0:
                    continue
                seen[span[k0]] = seen.get(span[k0], 0) + 1
                if on < J:
                    continue
                v = partial_v(hn, i, k0) + oh[start[span[k0]]:k0].sum(0)
                q = a['lock'][min(seen[span[k0]], 4) - 1]
                q[0] += 1
                q[1] += int(np.argmax(lprior + 1.5 * (base.logB @ v)) == z0)
            # flip rate of m1's argmax (and sticky85's) between consecutive notes of one true span
            for fi, (f, A) in enumerate(((f_base, None), (f_st, st))):
                f.set_song(A, None, d['durs'])
                f.start(d['meta'])
                arg, prev = [], None
                for i, (on, off, m) in enumerate(d['heard']):
                    f.step(i, float(on), int(m), None if prev is None else float(prev[1] - prev[0]))
                    arg.append(int(np.argmax(f.last)))
                    prev = (on, off)
                for i in range(1, len(arg)):
                    k0, kp = int(d['true'][i][0]), int(d['true'][i - 1][0])
                    if d['true'][i][0] < J or k0 >= len(z) or z[k0] < 0 or kp >= len(z) or span[kp] != span[k0]:
                        continue
                    a['flips'][fi][0] += 1
                    a['flips'][fi][1] += int(arg[i] != arg[i - 1])
        out['datasets'][ds] = {
            'spans_by_notes_1_2_3_4plus': a['spans'],
            'true_chord_among_best_covering': [x / max(n, 1) for x, n in zip(a['cons'], a['spans'])],
            'true_chord_unique_best': [x / max(n, 1) for x, n in zip(a['uniq'], a['spans'])],
            'mean_triads_tied_best': [x / max(n, 1) for x, n in zip(a['tied'], a['spans'])],
            'best_cover_tiebreak_IVIVvi_acc': [x / max(n, 1) for x, n in zip(a['tmpl_ok'], a['spans'])],
            'all_spans': {'among_best': sum(a['cons']) / sum(a['spans']), 'unique_best': sum(a['uniq']) / sum(a['spans']),
                          'mean_tied': sum(a['tied']) / sum(a['spans']), 'tiebreak_acc': sum(a['tmpl_ok']) / sum(a['spans'])},
            'perfect_lockon_ceiling_acc_by_note_index': [q[1] / max(q[0], 1) for q in a['lock']],
            'argmax_flip_rate_within_span': {'m1': a['flips'][0][1] / a['flips'][0][0],
                                             'sticky85': a['flips'][1][1] / a['flips'][1][0]}}
        print(ds, out['datasets'][ds], flush=True)
    print(f'{time.time() - t0:.0f}s', flush=True)

    def comb(mo_):
        f = AccFilter(mo_, 1.0, 0.5)
        f.cur_mode = 'med'
        return f
    out['combined_causal_lam1_acc0.5_curmed'] = r = run(data, base, comb)
    print(line('comb lam1 acc.5 med', r), flush=True)
    r2 = run(data, base, lambda mo_: AccFilter(mo_, 1.0, 0.5))
    out['combined_causal_lam1_acc0.5'] = r2
    print(line('comb lam1 acc.5', r2), flush=True)
    out['secs'] = round(time.time() - t0, 1)
    print('wrote', save(NAME, out))


if __name__ == '__main__':
    sys.exit(main())
