"""Count-in detector: finds four evenly spaced clicks in the mic signal, block by block.

Clicks are short tone bursts (about 1-1.5 kHz). The mic block is band-passed (700-2500 Hz, filter state
kept across blocks) and its envelope taken every 2 ms. An event starts when the envelope rises 15 dB above
the median of the last second and ends when it falls back; its strength is its peak; events longer than
250 ms (a sung note) are not clicks (real phone clicks in a room last 60-150 ms). When a click ends,
we look for four events c1 < c2 < c3 < c4 = the newest, evenly spaced (within 8 % of the spacing d,
d between 0.3 and 1.5 s), such that c2..c4 are among the strongest events heard (within 12 dB of the
strongest), nothing between c1 and c4 is nearly as strong (6 dB), and c1 is not much weaker (25 dB; the
first click can be quieter). Then the tempo is (c4 - c1) / 3 and beat 1 is one beat after c4.
Tested on synthetic clips and on a real phone-to-MacBook recording with echoes and knocks.
"""
import numpy as np
import scipy.signal as ss


class CountIn:
    def __init__(self, sr, dmin=0.3, dmax=1.5, tol=0.08, rise_db=15.0, refractory=0.06):
        self.sr, self.dmin, self.dmax, self.tol, self.rise, self.refr = sr, dmin, dmax, tol, rise_db, refractory
        self.b, self.a = ss.butter(4, [700 / (sr / 2), 2500 / (sr / 2)], 'band')
        self.zi = ss.lfilter_zi(self.b, self.a) * 0.0
        self.H = max(1, int(0.002 * sr))
        self.env = []                    # envelope values (dB), 2 ms each
        self.cand, self.cs = [], []      # click-like event start times (s) and peak strengths (dB)
        self.above = False
        self.ev = None                   # the event in progress: [start, peak]
        self.max_click = 0.25            # a click (with room echo) lasts 60-150 ms; a sung note 0.5 s and more
        self.result = None               # (beat_s, beat1_time) once found

    def feed(self, x, t_start):
        """x: one mic block; t_start: acoustic time of its first sample. Returns result once found."""
        if self.result is not None:
            return self.result
        y, self.zi = ss.lfilter(self.b, self.a, np.asarray(x, float), zi=self.zi)
        for i in range(0, len(y) - self.H + 1, self.H):
            e = 20 * np.log10(np.abs(y[i:i + self.H]).max() + 1e-12)
            self.env.append(e)
            floor = float(np.median(self.env[-500:]))       # the last second
            t = t_start + i / self.sr
            if e > floor + self.rise:
                if not self.above:
                    self.ev = [t, e]
                    self.above = True
                elif self.ev[1] is not None:
                    self.ev[1] = max(self.ev[1], e)
                    if t - self.ev[0] > self.max_click:   # too long for a click: a note, not a count-in
                        self.ev[1] = None
            elif self.above and e < floor + self.rise - 6:
                self.above = False
                st, pk = self.ev
                if pk is None:
                    continue
                if self.cand and st - self.cand[-1] <= self.refr:
                    if pk <= self.cs[-1]:
                        continue
                    self.cand[-1], self.cs[-1] = st, pk          # a faint precursor must not hide the click
                else:
                    self.cand.append(st)
                    self.cs.append(pk)
                if True:
                    r = self._check()
                    if r is not None:
                        self.result = r
                        return r
        return None

    def _check(self):
        c, s = self.cand, self.cs
        n = len(c)
        if n < 4:
            return None
        top = max(s)
        i4 = n - 1
        c4 = c[i4]
        for i3 in range(i4 - 1, -1, -1):
            d = c4 - c[i3]
            if d < self.dmin:
                continue
            if d > self.dmax:
                break
            tol = self.tol * d
            i2 = min(range(i3), key=lambda j: abs(c[j] - (c4 - 2 * d)), default=None)
            if i2 is None or abs(c[i2] - (c4 - 2 * d)) > tol:
                continue
            i1 = min(range(i2), key=lambda j: abs(c[j] - (c4 - 3 * d)), default=None)
            if i1 is None or abs(c[i1] - (c4 - 3 * d)) > tol:
                continue
            weak = min(s[i2], s[i3], s[i4])
            if weak < top - 12 or s[i1] < weak - 25:
                continue
            others = [s[j] for j in range(i1 + 1, i4) if j not in (i2, i3)]
            if any(v >= weak - 6 for v in others):
                continue
            T = (c4 - c[i1]) / 3
            return T, c4 + T
        return None
