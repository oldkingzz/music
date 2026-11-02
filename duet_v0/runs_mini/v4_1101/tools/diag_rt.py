"""Where does Duet's REAL-TIME chord guess fail? (v4_1101 orchestrator diagnosis; reads the bundle, edits nothing)

Replays futures/v3_nolatency/bench3.run_song exactly (heard notes with 10 % mishearing, seed ^ 5160) on every
val song of bach, folk, pop, pop909, with m1_filter_v5lo's HMM chord filter and its chooser, and compares four
ways of knowing the chord of the beat her note starts on, ALL followed by the same m1 chooser:
  rt      real time: the filter itself (= m1_filter_v5lo; must reproduce music_nd 0.6206)
  past    causal, but told the TRUE chord of the previous beat (a perfect memory of past chords)
  smooth  NOT causal: HMM forward-backward over the whole heard song (an offline listener that heard it all)
  told    NOT causal: told the TRUE chord of the onset beat (the ceiling of this chooser)
plus the GRU ensemble's real-time guess (m3_seq_v5's three GRUs; chord accuracy only, no chooser).
Scores exactly like rescore_nd.py (music_nd, double). Every missed scored segment goes in ONE bucket, in this
order: onset_beat_unlabelled, held_into_new_chord (the chord changed while her note was still sounding),
no_good_note_below (no alto note under her TRUE note is a consonant chord tone), misheard, wrong_chord_guess
(argmax belief != onset chord), right_chord_still_missed.
Writes runs_mini/v4_1101/diag/<out>.json (mode 'x').

  cd ABS/runs_mini/v4_1101/diag && ABS/.venv/bin/python ABS/futures/v3_nolatency/tl.py 600 \
      ABS/.venv/bin/python ABS/runs_mini/v4_1101/tools/diag_rt.py diag_rt_v1
"""
import json
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402  (also puts duet_beat on sys.path)
import torch  # noqa: E402
from systems import m1_filter_v5lo as m1  # noqa: E402
from systems.m3_seq_v4 import V3, Ens, _load  # noqa: E402
from theory import ALTO, CONSONANT, CT  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v4_1101', 'diag')
GOOD = CONSONANT - {0}
J = bench3.J
MODES = ('rt', 'past', 'smooth', 'told')
GUESSERS = ('rt', 'past', 'smooth', 'gru')
BUCKETS = ('onset_beat_unlabelled', 'held_into_new_chord', 'no_good_note_below', 'misheard', 'wrong_chord_guess',
           'right_chord_still_missed')
EXPECT_RT = 0.620637625406759


class Probe(m1.Filter):
    """m1's filter and chooser; belief() replaced per mode and recorded."""

    def __init__(self, model, mode):
        super().__init__(model)
        self.mode = mode

    def set_truth(self, z, gamma):
        self.ztrue, self.gamma = z, gamma

    def belief(self, k):
        b = super().belief(k)
        z = self.ztrue
        if self.mode == 'told' and k < len(z) and z[k] >= 0:
            b = np.eye(m1.S)[z[k]]
        elif self.mode == 'past' and 1 <= k < len(z) and z[k - 1] >= 0:
            ll = self._beat_ll(k) * m1.P['cur_w']
            w = self.m.A[m1.tkey(self._pos(k), self.M)][z[k - 1]] * np.exp(ll - ll.max())
            b = w / w.sum()
        elif self.mode == 'smooth' and k < len(self.gamma):
            b = self.gamma[k]
        self.last = b
        return b


def smooth(model, h):
    """HMM forward-backward over the whole HEARD song: P(chord of beat k | every note of the song)."""
    t, M = h['tonic'], h['M']
    pos = [b['pos'] for b in h['beats']]
    K = len(pos)
    obs = m1.beat_obs([[on, off, (m - t) % 12] for on, off, m in h['sop']], K)
    ll = np.array([m1.P['lam'] * (model.logB @ o) if o.sum() > 0 else np.zeros(m1.S) for o in obs])
    e = np.exp(ll - ll.max(1, keepdims=True))
    al = np.zeros((K, m1.S))
    for k in range(K):
        pr = model.pi if k == 0 else al[k - 1] @ model.A[m1.tkey(pos[k], M)]
        w = pr * e[k]
        al[k] = w / w.sum()
    be = np.ones((K, m1.S))
    for k in range(K - 2, -1, -1):
        w = model.A[m1.tkey(pos[k + 1], M)] @ (e[k + 1] * be[k + 1])
        be[k] = w / w.sum()
    g = al * be
    return g / g.sum(1, keepdims=True)


def feasible_below(m, z, t):
    return any((m - h) % 12 in GOOD and (h - t) % 12 in CT[z] for h in range(ALTO[0], min(ALTO[1], m - 1) + 1))


def feasible_any(m, z, t):
    return any(abs(m - h) % 12 in GOOD and (h - t) % 12 in CT[z] for h in range(ALTO[0], ALTO[1] + 1))


def new_acc():
    return {'nd': {md: [0.0, 0.0, 0.0] for md in MODES},              # num, den, double
            'miss': {md: {b: 0.0 for b in BUCKETS} for md in MODES},
            'no_good_below_but_above_ok': 0.0,
            'acc': {g: {} for g in GUESSERS},                          # key -> [notes, right, sum P(true)]
            'notes': 0}


def one_song(song, probes, ens, model, r):
    h = bench3.B2.heard(song)
    heard = sorted(h['sop'], key=lambda n: n[0])
    true = sorted(song['sop'], key=lambda n: n[0])
    t = song['tonic']
    z = [b['z'] for b in song['beats']]
    pos = [b['pos'] for b in song['beats']]
    meta = {'tonic': t, 'M': song['M'], 'K': song['K'], 'pos': pos}
    gamma = smooth(model, h)
    outs, bels = {}, {}
    for md, p in probes.items():
        p.set_truth(z, gamma)
        p.start(meta)
        outs[md], bels[md], prev = [], [], None
        for i, (on, off, m) in enumerate(heard):
            outs[md].append(int(p.step(i, float(on), int(m), None if prev is None else float(prev[1] - prev[0]))))
            bels[md].append(p.last)
            prev = (on, off)
    ens.start(meta)
    bels['gru'], prev = [], None
    for on, off, m in heard:
        bels['gru'].append(ens.chord_probs(float(on), int(m), None if prev is None else float(prev[1] - prev[0])))
        prev = (on, off)

    span, sid = [], -1                    # chord span of each beat: a run of beats with the same label
    for k in range(len(z)):
        if z[k] < 0:
            span.append(None)
            continue
        if k == 0 or z[k - 1] != z[k]:
            sid += 1
        span.append(sid)
    seen, kprev = {}, None
    for i, (on, off, m) in enumerate(true):
        mh = heard[i][2]
        k0 = int(on)
        z0 = z[k0] if k0 < len(z) else -1
        if z0 >= 0:
            seen[span[k0]] = seen.get(span[k0], 0) + 1
            if on >= J:
                r['notes'] += 1
                zp = z[kprev] if kprev is not None and kprev < len(z) else -1
                ch = 'first_or_after_unlabelled' if zp < 0 else ('chord_changed' if zp != z0 else 'chord_same')
                keys = ('all', ch, 'downbeat' if pos[k0] == 0 else 'other_beat',
                        f'note_{min(seen[span[k0]], 4)}_in_chord', 'misheard' if mh != m else 'heard_right')
                for g in GUESSERS:
                    b = bels[g][i]
                    ok = int(int(np.argmax(b)) == z0)
                    for key in keys:
                        a = r['acc'][g].setdefault(key, [0, 0, 0.0])
                        a[0] += 1
                        a[1] += ok
                        a[2] += float(b[z0])
        kprev = k0
        for k in range(max(J, k0), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            for md in MODES:
                hh = outs[md][i]
                a = r['nd'][md]
                a[1] += ov
                good = False
                if hh >= 0:
                    iv = abs(m - hh) % 12
                    good = iv in GOOD and (hh - t) % 12 in CT[z[k]]
                    a[0] += ov * good
                    a[2] += ov * (iv == 0)
                if good:
                    continue
                if z0 < 0:
                    bk = 'onset_beat_unlabelled'
                elif z[k] != z0:
                    bk = 'held_into_new_chord'
                elif not feasible_below(m, z[k], t):
                    bk = 'no_good_note_below'
                elif mh != m:
                    bk = 'misheard'
                elif int(np.argmax(bels[md][i])) != z0:
                    bk = 'wrong_chord_guess'
                else:
                    bk = 'right_chord_still_missed'
                r['miss'][md][bk] += ov
                if md == 'rt' and bk == 'no_good_note_below' and feasible_any(m, z[k], t):
                    r['no_good_below_but_above_ok'] += ov


def finish(r):
    out = {'notes_scored': r['notes'], 'music_nd': {}, 'double': {}, 'miss_share': {}, 'chord_acc': {}}
    for md in MODES:
        num, den, dbl = r['nd'][md]
        out['music_nd'][md] = num / den
        out['double'][md] = dbl / den
        out['miss_share'][md] = {b: v / den for b, v in r['miss'][md].items()}
    out['no_good_below_but_above_ok_share'] = r['no_good_below_but_above_ok'] / r['nd']['rt'][1]
    for g in GUESSERS:
        out['chord_acc'][g] = {k: {'notes': a[0], 'acc': a[1] / a[0], 'p_true': a[2] / a[0]}
                               for k, a in sorted(r['acc'][g].items())}
    return out


def mean_over(res, f):
    return sum(f(v) for v in res.values()) / len(res)


def main():
    name = sys.argv[1]
    path = os.path.join(OUT, name + '.json')
    if os.path.exists(path):
        sys.exit(f'refusing: {path} exists')
    t0 = time.time()
    torch.set_num_threads(1)
    model = m1.build(bench3.train_sets()).m
    probes = {md: Probe(model, md) for md in MODES}
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0, dict(V3, step=0.01))
    res = {}
    for ds, songs in bench3.eval_sets('val').items():
        r = new_acc()
        for song in songs:
            one_song(song, probes, ens, model, r)
        res[ds] = finish(r)
    head = {'music_nd': {md: mean_over(res, lambda v: v['music_nd'][md]) for md in MODES},
            'double': {md: max(v['double'][md] for v in res.values()) for md in MODES},
            'miss_share': {md: {b: mean_over(res, lambda v: v['miss_share'][md][b]) for b in BUCKETS} for md in MODES},
            'no_good_below_but_above_ok_share': mean_over(res, lambda v: v['no_good_below_but_above_ok_share']),
            'chord_acc': {g: {k: {'acc': mean_over(res, lambda v: v['chord_acc'][g][k]['acc']),
                                  'p_true': mean_over(res, lambda v: v['chord_acc'][g][k]['p_true']),
                                  'notes': sum(v['chord_acc'][g][k]['notes'] for v in res.values())}
                              for k in res['bach']['chord_acc'][g]} for g in GUESSERS}}
    ok = abs(head['music_nd']['rt'] - EXPECT_RT) < 1e-9
    out = {'what': __doc__.strip().split('\n')[0], 'reproduces_m1': ok, 'secs': round(time.time() - t0, 1),
           'headline_mean_of_4': head, 'datasets': res,
           'notes': 'val only (the bundle has no test split); headline = mean over bach, folk, pop, pop909'}
    os.makedirs(OUT, exist_ok=True)
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1)
    print('reproduces m1_filter_v5lo:', ok, round(head['music_nd']['rt'], 6), f"{out['secs']}s")
    for md in MODES:
        print(f"{md:6s} music_nd {head['music_nd'][md]:.4f} ", ' '.join(
            f"{ds} {v['music_nd'][md]:.3f}" for ds, v in res.items()),
            ' miss:', ' '.join(f"{b} {x:.3f}" for b, x in head['miss_share'][md].items() if x > 0))
    for g in GUESSERS:
        print(f"acc {g:6s}", ' '.join(f"{k} {v['acc']:.3f}" for k, v in head['chord_acc'][g].items()))


if __name__ == '__main__':
    main()
