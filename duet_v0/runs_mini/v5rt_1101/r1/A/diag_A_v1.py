"""Idea A diagnostic (run v5rt_1101, round 1). VAL ONLY. Where do a setting's gains and losses come from?
Runs r1_A_core_v1.PlanA (one JSON setting) through bench_rt.simulate and logs every decision that changed Duet's note
or kept it against the reactive choice. For each logged decision it integrates, over the time that decision governs
(until Duet's next change), the counted hit and harsh time (her TRUE notes and chords, labelled beats from J = 4) of
the note chosen and of the counterfactual (plan: the note that was sounding; hear-keep: the reactive choice h_r).
Plan attacks are split by what she really did at q: a new note starts within 0.01 beat ('onset'), her heard note
holds on ('hold'), or she rests ('rest'); fresh = the attack is harsh against her true note at q.

  python diag_A_v1.py '<json overrides>' [--sub N]
"""
import argparse
import bisect
import json
import os
import sys
from collections import defaultdict

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'A'))
import bench_rt as B  # noqa: E402
from rt_systems import r1_A_core_v1 as core  # noqa: E402
from systems import r2_mix_v2  # noqa: E402
from tune_A_v1 import BASE  # noqa: E402


class Logged(core.PlanA):
    def start(self, meta):
        super().start(meta)
        self.log = []

    def plan(self, q, heard):
        old = self.h
        out = super().plan(q, heard)
        if out is not None:
            self.log.append(('plan', q, out, old, None))
        return out

    def hear(self, i, onset, midi, prev_dur):
        old = self.h
        f = self.f
        prev, prev_m = f.prev, f.prev_m
        out = super().hear(i, onset, midi, prev_dur)
        if out is None and old >= 0 and self.her and len(self.her) > 1:
            # kept: what would the reactive choice have been? (recompute m1's argmax over the same scores)
            pfit = f.belief(int(onset)) @ core.M1.CTM
            best, bs = -1, -1e9
            for h in range(core.ALTO[0], core.ALTO[1] + 1):
                s = self._m1score(h, midi, pfit, prev, prev_m)
                if s is not None and s > bs:
                    best, bs = h, s
            if best != old:
                self.log.append(('keep', i, old, best, onset))
        return out


def integ(song, a, b, h, L):
    """Counted hit and harsh time of Duet note h over [a, b) against her true notes and chords."""
    z = [x['z'] for x in song['beats']]
    t = song['tonic']
    hit = harsh = 0.0
    for on, off, m in song['sop']:
        lo, hi = max(a, on), min(b, off)
        if hi <= lo:
            continue
        for k in range(max(B.J, int(lo)), int(hi) + 1):
            ov = min(hi, k + 1) - max(lo, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            hit += ov * B.hit(h, m, t, z[k])
            harsh += ov * (h >= 0 and abs(m - h) % 12 in B.HARSH)
    return hit, harsh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg')
    ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--L', type=float, default=0.1)
    a = ap.parse_args()
    B.bench3.B2.heard = B.HEARD1
    train = B.bench3.train_sets()
    obj = Logged(r2_mix_v2.build(train), core.Prior(), dict(BASE, **json.loads(a.cfg)))
    L = a.L
    for ds, songs in B.bench3.eval_sets('val').items():
        if a.sub:
            songs = songs[:a.sub]
        agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0])  # n, hit_new, hit_old, harsh_new, harsh_old, fresh
        tot_d = 0.0
        n_notes = 0
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, _, _ = B.simulate(obj, meta, notes, L)
            acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                                 'on_time_right', 'changes', 'late_entry'), 0.0)
            B.score_song(s, T, H, L, acc)
            tot_d += acc['d']
            n_notes += acc['notes']
            true = sorted(s['sop'], key=lambda n: n[0])
            tons = [n[0] for n in true]
            for kind, x, hnew, hold, onset in obj.log:
                t0 = x if kind == 'plan' else onset + L
                j = bisect.bisect_right(T, t0)
                t1 = T[j] if j < len(T) else s['K']
                hn, hh = integ(s, t0, t1, hnew, L)
                ho, ho_h = integ(s, t0, t1, hold, L)
                if kind == 'plan':
                    u = bisect.bisect_left(tons, x - 1e-9)
                    v = bisect.bisect_right(tons, x + 1e-9) - 1
                    cur = [n for n in true if n[0] <= x + 0.01 and n[1] > x]
                    if u < len(true) and abs(true[u][0] - x) < 0.01 or any(abs(n[0] - x) < 0.01 for n in cur):
                        cat = 'onset'
                    elif cur:
                        cat = 'hold'
                    else:
                        cat = 'rest'
                    sound = [n for n in true if n[0] <= x < n[1]]
                    fresh = bool(sound) and hnew >= 0 and abs(sound[-1][2] - hnew) % 12 in B.HARSH and x >= B.J
                    cat = f"plan_{cat}{'_sil' if hnew < 0 else ''}"
                else:
                    cat, fresh = 'hear_keep', False
                r = agg[cat]
                r[0] += 1
                r[1] += hn
                r[2] += ho
                r[3] += hh
                r[4] += ho_h
                r[5] += fresh
        print(f'{ds}: counted {tot_d:.0f} beats, notes {int(n_notes)}', flush=True)
        for cat, r in sorted(agg.items()):
            print(f'   {cat:16s} n {r[0]:6d} ({100 * r[0] / max(n_notes, 1):5.1f} per 100 notes)  '
                  f'hit gain {(r[1] - r[2]) / tot_d:+.4f} of counted time  harsh change {(r[3] - r[4]) / tot_d:+.4f}  '
                  f'fresh clashes {100 * r[5] / max(n_notes, 1):.2f} per 100 notes', flush=True)


if __name__ == '__main__':
    main()
