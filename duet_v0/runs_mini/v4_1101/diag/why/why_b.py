"""why_b: (iii) how much does her melody say about the chord? val only, m1's pooled model (bal=1, lam=1.5).
  emission-only per beat (no transitions), TRUE and HEARD melody; share of her sung time on a tone of the true chord
  (all / at the beat onset / notes starting off the beat / short notes); span-accumulated evidence with the TRUE span
  boundaries (oracle segmentation, causal evidence) by note index; memoryless current-beat guess; confusions of m1;
  plus two causal mixtures of HMMs (song-adaptive tables) as E3 probes. Writes why_b_v1.json (mode 'x')."""
import sys
import time

import numpy as np

from why_lib import (DS, CT, J, S, FilterV, MixFilter, ModelV, bench3, line, m1, prep, run, save, smooth)

NAME = 'why_b_v1'
SHARE2 = {(a, b) for a in range(S) for b in range(S) if a != b and len(CT[a] & CT[b]) == 2}


def partial_v(notes, i, k0, cur_end=None):
    """pc-time vector of beat k0 as m1 sees it at note i's onset: earlier notes with their lengths, note i to the
    end of the beat (cur_end None) or to cur_end."""
    v = np.zeros(12)
    on_i = notes[i][0]
    for j in range(i, -1, -1):
        on, off, pc = notes[j]
        e = (k0 + 1 if cur_end is None else cur_end) if j == i else off
        if e <= k0:
            break
        ov = min(e, k0 + 1) - max(on, k0)
        if ov > 0:
            v[pc] += ov
    assert on_i >= k0
    return v


def evidence(data, mo):
    res = {}
    prior = mo.Bc.sum(1) * 0 + 1.0 / S
    for ds, L in data.items():
        a = dict(beats=0, em_true_clean=0, em_true=0, em_heard=0, em_heard_template=0, rest_beats=0,
                 ct=[0.0, 0.0], ct_heard=[0.0, 0.0], ct_onset=[0.0, 0.0], ct_offstart=[0.0, 0.0],
                 ct_onstart=[0.0, 0.0], ct_short=[0.0, 0.0], ct_long=[0.0, 0.0],
                 span_c=[[0, 0] for _ in range(4)], span_full=[0, 0], span_full_true_clean=[0, 0],
                 mem0=[0, 0], beats_in_span=[[0, 0] for _ in range(4)])
        for d in L:
            z, t = d['z'], d['t']
            K = len(d['pos'])
            ot = m1.beat_obs(d['trel'], K)
            oh = m1.beat_obs(d['hrel'], K)
            for k in range(J, K):
                if z[k] < 0:
                    continue
                if ot[k].sum() <= 0:
                    a['rest_beats'] += 1
                    continue
                a['beats'] += 1
                a['em_true_clean'] += int(np.argmax(mo.logBc @ ot[k]) == z[k])
                a['em_true'] += int(np.argmax(mo.logB @ ot[k]) == z[k])
                if oh[k].sum() > 0:
                    a['em_heard'] += int(np.argmax(mo.logB @ oh[k]) == z[k])
                    sc = [oh[k][sorted(CT[c])].sum() for c in range(S)]
                    best = max(sc)
                    a['em_heard_template'] += int([c for c in (0, 4, 3, 5, 1, 2, 6) if sc[c] == best][0] == z[k])
            # chord-tone share of sung time, from beat J, labelled beats
            for (on, off, pc), (_, _, hpc) in zip(sorted(d['trel']), sorted(d['hrel'])):
                for k in range(max(J, int(on)), int(np.ceil(off))):
                    if k >= K or z[k] < 0:
                        continue
                    ov = min(off, k + 1) - max(on, k)
                    if ov <= 1e-9:
                        continue
                    c = float(pc in CT[z[k]])
                    a['ct'][0] += ov
                    a['ct'][1] += ov * c
                    a['ct_heard'][0] += ov
                    a['ct_heard'][1] += ov * float(hpc in CT[z[k]])
                    if on <= k < off:
                        a['ct_onset'][0] += 1
                        a['ct_onset'][1] += c
                    q = a['ct_onstart'] if abs(on - round(on)) < 1e-6 else a['ct_offstart']
                    q[0] += ov
                    q[1] += ov * c
                    q = a['ct_short'] if off - on < 0.75 else a['ct_long']
                    q[0] += ov
                    q[1] += ov * c
            # span-accumulated evidence with the TRUE boundaries (heard notes, m1's current-note assumption)
            hn = [[on, off, (m - t) % 12] for on, off, m in d['heard']]
            span = d['span']
            start = {}
            for k in range(len(z)):
                if span[k] is not None and span[k] not in start:
                    start[span[k]] = k
            seen = {}
            for i, (on, off, m) in enumerate(d['true']):
                k0 = int(on)
                z0 = z[k0] if k0 < len(z) else -1
                if z0 < 0:
                    continue
                seen[span[k0]] = seen.get(span[k0], 0) + 1
                if on < J:
                    continue
                b0 = start[span[k0]]
                v = partial_v(hn, i, k0)
                cur = mo.logB @ v if v.sum() > 0 else np.zeros(S)
                a['mem0'][0] += 1
                a['mem0'][1] += int(np.argmax(cur) == z0)          # memoryless: this beat only, flat prior
                acc_v = v + oh[b0:k0].sum(0)
                q = a['span_c'][min(seen[span[k0]], 4) - 1]
                q[0] += 1
                q[1] += int(np.argmax(mo.logB @ acc_v) == z0)
                q = a['beats_in_span'][min(k0 - b0, 3)]
                q[0] += 1
                q[1] += int(np.argmax(mo.logB @ acc_v) == z0)
            for sid, b0 in start.items():
                b1 = b0
                while b1 + 1 < len(z) and span[b1 + 1] == sid:
                    b1 += 1
                if b0 < J or oh[b0:b1 + 1].sum() <= 0:
                    continue
                a['span_full'][0] += 1
                a['span_full'][1] += int(np.argmax(mo.logB @ oh[b0:b1 + 1].sum(0)) == z[b0])
                a['span_full_true_clean'][0] += 1
                a['span_full_true_clean'][1] += int(np.argmax(mo.logBc @ ot[b0:b1 + 1].sum(0)) == z[b0])
        n = a['beats']
        res[ds] = {'emission_only_beat_acc': {'true_clean': a['em_true_clean'] / n, 'true': a['em_true'] / n,
                                              'heard': a['em_heard'] / n, 'heard_template': a['em_heard_template'] / n},
                   'rest_beat_share': a['rest_beats'] / (n + a['rest_beats']),
                   'chord_tone_share': {'true_all': a['ct'][1] / a['ct'][0], 'heard_all': a['ct_heard'][1] / a['ct_heard'][0],
                                        'true_at_beat_onset': a['ct_onset'][1] / a['ct_onset'][0],
                                        'true_note_starts_on_beat': a['ct_onstart'][1] / a['ct_onstart'][0],
                                        'true_note_starts_off_beat': a['ct_offstart'][1] / max(a['ct_offstart'][0], 1e-9),
                                        'true_short_lt_0.75': a['ct_short'][1] / max(a['ct_short'][0], 1e-9),
                                        'true_long_ge_0.75': a['ct_long'][1] / max(a['ct_long'][0], 1e-9)},
                   'memoryless_current_beat_acc': a['mem0'][1] / a['mem0'][0],
                   'span_accumulated_acc_by_note_index': [q[1] / max(q[0], 1) for q in a['span_c']],
                   'span_accumulated_acc_by_beats_into_span': [q[1] / max(q[0], 1) for q in a['beats_in_span']],
                   'span_accumulated_n_by_beats_into_span': [q[0] for q in a['beats_in_span']],
                   'span_full_heard_acc': a['span_full'][1] / a['span_full'][0],
                   'span_full_true_clean_acc': a['span_full_true_clean'][1] / a['span_full_true_clean'][0]}
    return res


def confusions(data, mo):
    res = {}
    for ds, L in data.items():
        C = np.zeros((S, S))
        f = FilterV(mo)
        for d in L:
            f.set_song(None, None, d['durs'])
            f.start(d['meta'])
            bels, prev = [], None
            for i, (on, off, m) in enumerate(d['heard']):
                f.step(i, float(on), int(m), None if prev is None else float(prev[1] - prev[0]))
                bels.append(f.last)
                prev = (on, off)
            for i, (on, off, m) in enumerate(d['true']):
                k0 = int(on)
                z0 = d['z'][k0] if k0 < len(d['z']) else -1
                if z0 >= 0 and on >= J:
                    C[z0, int(np.argmax(bels[i]))] += 1
        wrong = C.sum() - np.trace(C)
        pairs = sorted(((C[a, b], a, b) for a in range(S) for b in range(S) if a != b), reverse=True)[:6]
        names = ('I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii')
        res[ds] = {'true_chord_share': [round(x, 3) for x in (C.sum(1) / C.sum())],
                   'guess_share': [round(x, 3) for x in (C.sum(0) / C.sum())],
                   'recall_per_chord': [round(C[a, a] / max(C[a].sum(), 1), 3) for a in range(S)],
                   'wrong_share_relative_2_common_tones': sum(C[a, b] for a, b in SHARE2) / wrong,
                   'top_confusions_true_to_guess': [f'{names[a]}->{names[b]} {c / C.sum():.3f}' for c, a, b in pairs]}
    return res


def main():
    t0 = time.time()
    train = bench3.train_sets()
    base = ModelV(train)
    dsm = {ds: ModelV(train, only={ds}) for ds in DS}
    data = prep()
    out = {'what': __doc__.strip().split('\n')[0], 'chord_order': 'I ii iii IV V vi vii',
           'notes': 'val only (no test split); m1 pooled emission, flat prior unless said'}
    out['evidence'] = evidence(data, base)
    print(f'evidence done {time.time() - t0:.0f}s', flush=True)
    for ds, v in out['evidence'].items():
        print(ds, {k: (round(x, 3) if isinstance(x, float) else ({kk: round(xx, 3) for kk, xx in x.items()}
                                                              if isinstance(x, dict) else [round(y, 3) for y in x]))
                   for k, x in v.items()}, flush=True)
    out['confusions_m1_rt'] = confusions(data, base)
    for ds, v in out['confusions_m1_rt'].items():
        print(ds, v, flush=True)
    print(f'confusions done {time.time() - t0:.0f}s', flush=True)
    # E3 probes: causal mixtures of HMMs (no dataset name used)
    tabs = [dsm[ds].A for ds in DS]
    mix = {}
    for nm, models, eta, lam in (('MIX4_tables_lam1.5', [base] * 4, 1.0, 1.5),
                                 ('MIX4_tables_lam1.0', [base] * 4, 1.0, 1.0),
                                 ('MIX4_models_lam1.5', [dsm[ds] for ds in DS], 1.0, 1.5),
                                 ('MIX4_models_lam1.0', [dsm[ds] for ds in DS], 1.0, 1.0)):
        fn = (lambda mo_, models=models, eta=eta, lam=lam: MixFilter(models, tabs, lam=lam, eta=eta))
        r = run(data, base, fn)
        mix[nm] = r
        print(line(nm + ' rt', r), f'{time.time() - t0:.0f}s', flush=True)
    out['mixtures_rt'] = mix
    out['secs'] = round(time.time() - t0, 1)
    print('wrote', save(NAME, out))


if __name__ == '__main__':
    sys.exit(main())
