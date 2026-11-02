"""r3_true_cons_v3 (v4 round 3, idea E5): r3_true_cons_v1 on every note, except that a guessed true note other than the one Duet
heard only counts as covered when Duet's note is consonant with it and not its unison/octave (doubling her is not harmony)."""
import numpy as np

from systems.r3_true_cons_v1 import CTM, W_BIG, W_MID, W_OCT, W_PAR, W_STEP, W_UNIS, TrueCons, make
from theory import ALTO, ALTO_MID, CONSONANT

MODE = 'all_nd'            # E5 on every note, misheard hypotheses scored as consonant AND not unison/octave
TRUE_LO = -6               # her true note at heard + ((t - heard + 6) % 12) - 6, as in m1's robust branch


class TrueConsNd(TrueCons):
    """TrueCons whose E5 score gives a misheard hypothesis t != heard no credit when Duet would only double it."""
    mode = MODE
    true_lo = TRUE_LO

    def use_e5(self, hp):
        return True

    def step(self, i, onset, midi, prev_dur):
        self._p = self.ens.chord_probs(onset, midi, prev_dur)   # exactly once per note: advances the GRU state
        if self.notes and prev_dur is not None:
            self.notes[-1][1] = self.notes[-1][0] + prev_dur
        k = int(onset)
        self._finish_beats(k)
        self.notes.append([onset, None, (midi - self.t) % 12])
        b = self.belief(k)                                     # r2_mix_v2: b_HMM^0.3 * b_GRU^0.7, normalized
        pfit = b @ CTM
        hp = (midi - self.t) % 12
        robust = self.mode != 'off' and self.use_e5(hp)
        if robust:
            pt = self.m.Bc * self.m.N[:, hp][None, :]          # [z, t]: P(t | z) * P(heard | t)
            pt /= pt.sum(1, keepdims=True)
            jz = b[:, None] * pt                               # [z, t]
            lo = self.true_lo
            tm = np.array([midi + ((t - hp - lo) % 12) + lo for t in range(12)])
        ref = self.prev if self.prev is not None else ALTO_MID
        hi = min(ALTO[1], midi - 1)                            # below her heard note
        best, bs = -1, -1e9
        for h in range(ALTO[0], hi + 1):
            iv = abs(midi - h) % 12
            if robust:
                cv = np.fromiter((((abs(x - h) % 12) in CONSONANT) and (t == hp or abs(x - h) % 12 != 0)
                                  for t, x in enumerate(tm)), float, 12)
                s = float(CTM[:, (h - self.t) % 12] @ (jz @ cv))
                if s <= 0:
                    continue
            else:
                if iv not in CONSONANT:
                    continue
                s = pfit[(h - self.t) % 12]
            if iv == 0:
                s -= W_OCT if h != midi else W_UNIS
            d = abs(h - ref) if self.prev is not None else 0
            s -= W_STEP * d + W_BIG * (d > 7) + W_MID * abs(h - ALTO_MID)
            if self.prev is not None and self.prev_m is not None:
                dm, dh = midi - self.prev_m, h - self.prev
                i0 = abs(self.prev_m - self.prev) % 12
                if dm != 0 and dh != 0 and (dm > 0) == (dh > 0) and iv in (0, 7) and i0 == iv:
                    s -= W_PAR
            if s > bs:
                best, bs = h, s
        if best >= 0:
            self.prev = best
        self.prev_m = midi
        return best


def build(train):
    return make(TrueConsNd, train)
