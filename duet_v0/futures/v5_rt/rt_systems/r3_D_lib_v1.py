"""r3_D_lib_v1 (run v5rt_1101, round 3, idea D 'register'): LIBRARY (no build(): not a system itself). Every scored
variant r3_D_<name>_v<k>.py passes its own constants (rule 7). Val only, tuned on val (this bundle has no test split).

REGISTER ADAPTER around a bench_rt system (the 'base', e.g. r2_copyhm_v1). The base systems choose Duet's note in
theory.ALTO 53..77 and (m1's chooser) below her, and were trained on soprano-range melodies (TRAIN pooled note pitch:
5 % 59, median 68, 95 % 78; runs_mini/v5rt_1101/r3/D/reg_stats_v1.json). Under a low voice (the live bench's
'sung_low', the real voice v11 at about Eb3-E4) m1's chooser has almost no room below her and falls silent, and the
plan / posterior hear's 'Duet at or above her' penalty applies to every alto note, so they sang notes above her that
clashed. The adapter decides CAUSALLY from her heard notes an octave shift s (a multiple of 12):

  med = median of her first min(#heard, n_dec) heard pitches;  s = the smallest k * 12 >= 0 with med + s >= lo_up
  (and s = -12 k when med > hi_down: a very high voice).

s is first decided at her n_min-th heard note and re-decided at every later heard note until n_dec notes are heard;
then it is locked for the song. The base gets her heard pitches + s (hear() and every plan()'s heard list), so it sees a
melody in the register it was trained on. When s changes, the base is RE-SYNCED: base.start(meta) and her heard notes so
far are replayed through base.hear() with the new shift (their answers are not sounded; the base's state - chord filter,
GRUs, melody prior, copy model, harmonic memory - is then exactly that of a base fed the shifted melody from the start).
Before n_min notes (pre = 'base'): the base as is with s = 0; (pre = 'silent'): Duet stays silent and the base is not
called; at note n_min it is fed all heard notes (replay) and answers the current one.

Duet's note: the wrapper's sounding note is always out(base.h) (base.h = the note the base believes it sounds, -1 =
silent; the A-family systems keep it exactly); a call returns None when that does not change Duet's note. out():
  'asis'  the base's note as it is (Duet in the alto range, above a low voice);
  'fold'  the base's note - s (the interval the base chose against the shifted melody, now against her real voice),
          moved by octaves into [out_lo, out_hi] when it falls outside.
Scoring note: bench_rt / live_bench_rt judge |her - Duet| mod 12, so moving Duet by an octave ACROSS her pitch maps an
interval class c to 12 - c: 3 <-> 9, 4 <-> 8 and 1/2/6/10/11 stay harsh, but a perfect 5th below (7, consonant) becomes a
4th above (5, not consonant). 'fold' keeps the base's own classes wherever the folded note stays below her.
With s = 0 for the whole song the wrapper is a pure pass-through: Duet's step function is the base's exactly.
"""
import numpy as np


class RegAdapter:
    def __init__(self, base, cfg):
        self.base, self.c = base, dict(cfg)
        self.songs = []                                   # per song: final s, resyncs, notes heard when locked

    def start(self, meta):
        self.meta = meta
        self.base.start(meta)
        self.s = 0
        self.hist = []                                    # (i, onset, heard midi, prev_dur) as the engine gave them
        self.cur = -1                                     # Duet's note per this wrapper's answers (-1 = silent)
        self.locked = False
        self.fed = self.c['pre'] == 'base'                # the base has been fed her notes
        self.info = {'s': [], 'resync': 0}
        self.songs.append(self.info)

    # ------------------------------------------------------------------ the shift
    def _decide(self):
        c = self.c
        med = float(np.median([m for _, _, m, _ in self.hist[:c['n_dec']]]))
        s = 0
        while med + s < c['lo_up']:
            s += 12
        while med + s > c['hi_down']:
            s -= 12
        return s

    def _resync(self):
        """base.start + replay of her heard notes before the current one with the current shift."""
        self.base.start(self.meta)
        for j, on, m, pd in self.hist[:-1]:
            self.base.hear(j, on, m + self.s, pd)
        self.info['resync'] += 1

    # ------------------------------------------------------------------ Duet's note
    def _map(self, h):
        if h < 0 or self.s == 0 or self.c['out'] == 'asis':
            return h
        o = h - self.s
        while o < self.c['out_lo']:
            o += 12
        while o > self.c['out_hi']:
            o -= 12
        return o

    def _ret(self):
        o = self._map(int(self.base.h))
        if o == self.cur:
            return None
        self.cur = o
        return o

    # ------------------------------------------------------------------ bench_rt interface
    def plan(self, q, heard):
        if not self.fed:
            return None
        if self.s:
            heard = [(on, off, m + self.s) for on, off, m in heard]
        self.base.plan(q, heard)
        return self._ret()

    def hear(self, i, onset, midi, prev_dur):
        midi = int(midi)
        self.hist.append((i, onset, midi, prev_dur))
        n = len(self.hist)
        if not self.locked and n >= self.c['n_min']:
            s = self._decide()
            if n >= self.c['n_dec']:
                self.locked = True
            if s != self.s or not self.fed:
                self.s = s
                self.info['s'].append([n, s])
                if n > 1 or not self.fed:
                    self._resync()
                self.fed = True
        if not self.fed:
            return None
        self.base.hear(i, onset, midi + self.s, prev_dur)
        return self._ret()
