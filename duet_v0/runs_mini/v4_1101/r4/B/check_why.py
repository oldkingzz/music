"""One-off diagnostic, v4 round 4, implementer B (E5); not a system, not a scored variant; ALL val songs (val, tuned on val).
Where does mishearing cost the champion r3_style_mix_v3? Three runs per song, rescore_nd.py's music_nd rule:
  heard   the real setting (Duet hears her notes with 10 % mishearing)
  clean   Duet hears her TRUE notes (no mishearing)
  hybrid  ORACLE, diagnosis only (not causal-fair): the chord belief (style HMM + 3 GRUs) is fed her TRUE notes, the chooser
          the heard ones (consonant with and below the heard note), so hybrid - heard = the cost that flows through the BELIEF,
          clean - hybrid = the cost that flows through the CHOICE.
Each difference is split into her notes that were misheard / heard right (heard pc in / out of the major scale).
Writes runs_mini/v4_1101/r4/B/check_why_<time>.json (new file, mode 'x').
"""
import json
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V3DIR = os.path.join(ABS, 'futures', 'v3_nolatency')
OUTDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'r4', 'B')
sys.path.insert(0, V3DIR)
import bench3  # noqa: E402
from theory import ALTO, ALTO_MID, CONSONANT, CT, MAJOR  # noqa: E402
from systems import r4_true_cons_v1 as R  # noqa: E402


class OracleBelief(R.TrueConsBelief):
    """Diagnosis only: belief from her TRUE note (self.true_m[i]), the champion's hard-rule chooser on the HEARD note."""
    true_m = None

    def step(self, i, onset, midi, prev_dur):
        tm = self.true_m[i]
        self._p = self.ens.chord_probs(onset, tm, prev_dur)
        if self.notes and prev_dur is not None:
            self.notes[-1][1] = self.notes[-1][0] + prev_dur
        k = int(onset)
        self._finish_beats(k)
        self.notes.append([onset, None, (tm - self.t) % 12])
        self._sus.append(False)
        b = self.belief(k)
        pfit = b @ R.CTM
        ref = self.prev if self.prev is not None else ALTO_MID
        best, bs = -1, -1e9
        for h in range(ALTO[0], min(ALTO[1], midi - 1) + 1):
            iv = abs(midi - h) % 12
            if iv not in CONSONANT:
                continue
            s = pfit[(h - self.t) % 12]
            if iv == 0:
                s -= R.W_OCT
            d = abs(h - ref) if self.prev is not None else 0
            s -= R.W_STEP * d + R.W_BIG * (d > 7) + R.W_MID * abs(h - ALTO_MID)
            if self.prev is not None and self.prev_m is not None:
                dm, dh = midi - self.prev_m, h - self.prev
                i0 = abs(self.prev_m - self.prev) % 12
                if dm != 0 and dh != 0 and (dm > 0) == (dh > 0) and iv in (0, 7) and i0 == iv:
                    s -= R.W_PAR
            if s > bs:
                best, bs = h, s
        if best >= 0:
            self.prev = best
        self.prev_m = midi
        return best


def run_clean(system, song):
    """bench3.run_song, but feeding her TRUE notes."""
    notes = sorted(song['sop'], key=lambda n: n[0])
    system.start({'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'pos': [b['pos'] for b in song['beats']]})
    out, prev = [], None
    for i, (on, off, m) in enumerate(notes):
        out.append(int(system.step(i, float(on), int(m), None if prev is None else float(prev[1] - prev[0]))))
        prev = (on, off)
    return out


def credit(song, hs):
    true = sorted(song['sop'], key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    t = song['tonic']
    num, den = np.zeros(len(true)), np.zeros(len(true))
    for i, (on, off, m) in enumerate(true):
        h = hs[i]
        for k in range(max(bench3.J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            den[i] += ov
            if h >= 0:
                iv = abs(m - h) % 12
                num[i] += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])
    return num, den


def main():
    t0 = time.time()
    train = bench3.train_sets()
    sets = bench3.eval_sets('val')
    champ = R.build(train)
    champ.belief_side, champ.choice_side = 'off', 'off'          # = r3_style_mix_v3 (check_copy.py)
    orc = OracleBelief(champ.models, champ.lams, champ.rh, champ.mprior, champ.ens)
    orc.belief_side, orc.choice_side = 'off', 'off'
    res = {'what': __doc__.splitlines()[0], 'rows': {}}
    for ds, songs in sets.items():
        tot = {r: 0.0 for r in ('heard', 'clean', 'hybrid')}
        part = {}
        den_all = 0.0
        for s in songs:
            heard = sorted(bench3.B2.heard(s)['sop'], key=lambda n: n[0])
            true = sorted(s['sop'], key=lambda n: n[0])
            orc.true_m = [int(n[2]) for n in true]
            outs = {'heard': bench3.run_song(champ, s), 'clean': run_clean(champ, s), 'hybrid': bench3.run_song(orc, s)}
            cr = {r: credit(s, o)[0] for r, o in outs.items()}
            den = credit(s, outs['heard'])[1]
            den_all += den.sum()
            for r in tot:
                tot[r] += cr[r].sum()
            for i in range(len(true)):
                mis = heard[i][2] != true[i][2]
                oos = (heard[i][2] - s['tonic']) % 12 not in MAJOR
                key = ('misheard' if mis else 'heard_right') + '|' + ('out_of_scale' if oos else 'in_scale')
                e = part.setdefault(key, {'notes': 0, 'belief_cost': 0.0, 'choice_cost': 0.0, 'total_cost': 0.0})
                e['notes'] += 1
                e['belief_cost'] += cr['hybrid'][i] - cr['heard'][i]
                e['choice_cost'] += cr['clean'][i] - cr['hybrid'][i]
                e['total_cost'] += cr['clean'][i] - cr['heard'][i]
        row = {r: round(v / den_all, 5) for r, v in tot.items()}
        row['cost_total'] = round(row['clean'] - row['heard'], 5)
        row['cost_belief'] = round(row['hybrid'] - row['heard'], 5)
        row['cost_choice'] = round(row['clean'] - row['hybrid'], 5)
        row['by_note'] = {k: {'notes': v['notes'], **{c: round(v[c] / den_all, 5) for c in ('belief_cost', 'choice_cost', 'total_cost')}}
                          for k, v in sorted(part.items())}
        res['rows'][ds] = row
        print(ds, {k: v for k, v in row.items() if k != 'by_note'}, flush=True)
        for k, v in row['by_note'].items():
            print('   ', k, v, flush=True)
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"check_why_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('secs', res['secs'], 'wrote', path)


if __name__ == '__main__':
    main()
