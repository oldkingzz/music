"""r3_D_lib_v3 (run v5rt_1101, round 3, idea D 'register'): LIBRARY (no build()). r3_D_lib_v2's register adapter (read
r3_D_lib_v1 and v2: causal octave shift s from the median of her first heard notes, base re-synced by replay, Duet's note
'asis' or 'fold', optional listening phase) with two options:

1. listen_at = 'hear': the first decision after the listening phase is made only at a heard note (not at a plan point),
   so Duet's first note is the base's answer to a note she has just been heard to start. (v2 decided at the first plan
   point q >= listen and entered there with the base's answer to her PREVIOUS note: on the live bench that entry added
   fresh clashes, because she often starts a new note exactly on that beat.)
2. guard (default None = off): a heard pitch outside the song's major scale is suspicious live (the tracker's errors are
   +-1..2 semitone near-misses; a real chromatic note is rarer). At such a heard note the base still hears it (its state
   advances as usual), but Duet does not attack a new note for it:
     'hold'      Duet keeps its sounding note if that is not harsh (2nd, tritone, 7th) against the heard pitch, else the
                 base's answer sounds;
     'hold_sil'  Duet keeps its sounding note if that is not harsh against the heard pitch, else Duet goes silent.
   The base's own belief (its sounding note h and m1's voice-leading memory f.prev) is set to what Duet really does, so
   the base stays consistent. This option is a hearing guard, not a register rule: it is here to measure what the
   real-voice gate G3 needs on top of the register adapter.
With listen_at = 'plan' and guard = None this is r3_D_lib_v2's adapter exactly. Val only, tuned on val (no test split).
"""
import numpy as np

from theory import MAJOR

HARSH = frozenset({1, 2, 6, 10, 11})


class RegAdapter:
    def __init__(self, base, cfg):
        self.base, self.c = base, dict(cfg)
        self.songs = []

    def start(self, meta):
        self.meta = meta
        self.tonic = meta['tonic']
        self.base.start(meta)
        self.s = 0
        self.hist = []
        self.cur = -1                                     # Duet's note per this wrapper's answers (-1 = silent)
        self.bcur = -1                                    # the base's note that sounds as cur
        self.decided = False
        self.locked = False
        self.fed = self.c['pre'] == 'base'
        self.info = {'s': [], 'resync': 0, 'first': None, 'guard_hold': 0, 'guard_sil': 0}
        self.songs.append(self.info)

    # ------------------------------------------------------------------ the shift (as r3_D_lib_v2)
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
        self.base.start(self.meta)
        for j, on, m, pd in self.hist[:upto]:
            self.base.hear(j, on, m + self.s, pd)
        self.info['resync'] += 1
        self.fed = True

    def _decide(self, t, upto):
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
        b = int(self.base.h)
        o = self._map(b)
        self.bcur = b
        if o == self.cur:
            return None
        self.cur = o
        return o

    def _override(self, b):
        """Duet really sounds base note b (-1 = silent): the base's belief follows."""
        base = self.base
        base.h = b
        base.planned = False
        if b >= 0:
            base.f.prev = b

    # ------------------------------------------------------------------ bench_rt interface
    def plan(self, q, heard):
        if not self.decided and self.c.get('listen_at', 'plan') == 'plan' and self._ready(q):
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
        g = self.c.get('guard')
        if g and (midi - self.tonic) % 12 not in MAJOR and len(self.hist) > 1:
            if self.cur >= 0 and abs(midi - self.cur) % 12 not in HARSH:
                self._override(self.bcur)
                self.info['guard_hold'] += 1
                return None
            if g == 'hold_sil':
                self._override(-1)
                self.info['guard_sil'] += 1
                if self.cur >= 0:
                    self.cur = -1
                    self.bcur = -1
                    return -1
                return None
        return self._ret()
