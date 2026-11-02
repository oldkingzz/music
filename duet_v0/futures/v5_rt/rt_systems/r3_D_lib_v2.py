"""r3_D_lib_v2 (run v5rt_1101, round 3, idea D 'register'): LIBRARY (no build()). r3_D_lib_v1's register adapter (read
its docstring: causal octave shift s from the median of her first heard notes, base re-synced by replay, Duet's note
'asis' or 'fold') with one addition, a LISTENING PHASE:

  The first decision on s is made only once at least n_min notes are heard AND the decision time (a plan point q, or a
  heard note's onset) is at or after beat `listen`. Before that (pre = 'silent') Duet is silent and the base is not
  called; at the decision the base is fed all heard notes (replay) and Duet enters at once with the note the base
  believes it sounds (at a plan point: on the beat). With pre = 'base' the base runs as is (s = 0) before the decision.
  After the first decision, s is re-decided at every heard note until n_dec notes are heard, then locked.

Why: the register is unknown until she has sung a few notes; before the decision the base can only sing in the wrong
octave relation (for a low voice: above her, the plan / posterior hear's 'Duet at or above her' term flat over the whole
alto range). Both benches (bench_rt, live_bench_rt) count her singing from beat J = 4 only, so with listen = 4 the
listening phase costs only what she sings from beat 4 until Duet's entry at the first plan point. With listen = 0,
n_min = 1 and pre = 'base' this is r3_D_lib_v1's adapter exactly.
Val only, tuned on val (no test split).
"""
import numpy as np


class RegAdapter:
    def __init__(self, base, cfg):
        self.base, self.c = base, dict(cfg)
        self.songs = []                                   # per song: shift decisions [notes heard, s], resyncs

    def start(self, meta):
        self.meta = meta
        self.base.start(meta)
        self.s = 0
        self.hist = []                                    # (i, onset, heard midi, prev_dur) as the engine gave them
        self.cur = -1                                     # Duet's note per this wrapper's answers (-1 = silent)
        self.decided = False
        self.locked = False
        self.fed = self.c['pre'] == 'base'                # the base has been fed her notes (with the current s)
        self.info = {'s': [], 'resync': 0, 'first': None}
        self.songs.append(self.info)

    # ------------------------------------------------------------------ the shift
    def _shift(self):
        c = self.c
        med = float(np.median([m for _, _, m, _ in self.hist[:c['n_dec']]]))
        s = 0
        while med + s < c['lo_up']:
            s += 12
        while med + s > c['hi_down']:
            s -= 12
        return s

    def _ready(self, t):
        return len(self.hist) >= self.c['n_min'] and t >= self.c.get('listen', 0.0)

    def _feed(self, upto):
        """base.start + replay of her heard notes hist[:upto] with the current shift."""
        self.base.start(self.meta)
        for j, on, m, pd in self.hist[:upto]:
            self.base.hear(j, on, m + self.s, pd)
        self.info['resync'] += 1
        self.fed = True

    def _decide(self, t, upto):
        """(Re-)decide s at decision time t; the base is re-fed hist[:upto] when s changes or it was not fed yet."""
        s = self._shift()
        n = len(self.hist)
        first = not self.decided
        self.decided = True
        if n >= self.c['n_dec']:
            self.locked = True
        if first:
            self.info['first'] = [n, float(t)]
        if s != self.s or not self.fed:
            changed = s != self.s
            self.s = s
            self.info['s'].append([n, s])
            if changed or not self.fed:
                if upto > 0 or not self.fed:
                    self._feed(upto)

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
        if not self.decided and self._ready(q):
            self._decide(q, len(self.hist))
        if not self.fed:
            return None
        if self.s:
            heard = [(on, off, m + self.s) for on, off, m in heard]
        self.base.plan(q, heard)
        return self._ret()

    def hear(self, i, onset, midi, prev_dur):
        midi = int(midi)
        self.hist.append((i, onset, midi, prev_dur))
        if not self.locked and self._ready(onset):
            self._decide(onset, len(self.hist) - 1)
        if not self.fed:
            return None
        self.base.hear(i, onset, midi + self.s, prev_dur)
        return self._ret()
