"""r3_cover_fit_v1 (v4 round 3, idea E4): r2_mix_v2's chord belief + m1_filter_v5lo's chooser (copied in), but a candidate
note is rated by the expected share of her note's time on which it is a chord tone (later beats forecast with m1's tables)."""
import numpy as np
import torch

from systems import m1_filter_v5lo
from systems.m1_filter_v5lo import CTM, tkey
from systems.m3_seq_v4 import V3, Ens, _load
from systems.r2_mix_v2 import FLOOR, Mix
from theory import ALTO, ALTO_MID, CONSONANT

# m1_filter_v5lo's six chooser weights (its P['w_*'] values), all unchanged
W_STEP = 0.01     # cost per semitone that Duet's line moves
W_BIG = 0.4       # extra cost of a leap of more than 7 semitones
W_PAR = 0.6       # cost of a parallel 5th/8ve (both voices move the same way into the same interval)
W_MID = 0.004     # cost per semitone away from the middle of the alto range
W_OCT = 0.3       # cost of an octave with her note
W_UNIS = 1.0      # cost of a unison with her note (inert: Duet stays strictly below her)

# E4: how the time her note covers is estimated (causal: only her finished notes and the beat grid)
AHEAD = 8         # beats after her onset beat that are forecast (0 = onset beat only = r2_mix_v2 exactly)
NDUR = 8          # her last NDUR finished note lengths form the length estimate
LENGTH = 'emp'    # 'emp': each of those lengths counts equally; 'med': only their median (as m1's look option)
SIDE = 'mix'      # 'mix': forecast the champion's mixed belief; 'hmm': forecast only the HMM part, then mix with the GRU
EVID = 0          # 1: on a forecast beat her held note also counts as evidence (m1's emission model, as its filter does)
FIRST_LEN = 1.0   # length (beats) assumed before she has finished any note


class Cover(Mix):
    """r2_mix_v2's belief and bookkeeping; m1_filter_v5lo's chooser path (robust 0, look 0, below 1, order2 0) copied in,
    with P(chord tone at her onset beat) replaced by sum_j share_j * P(chord tone at beat k+j)."""
    w_step, w_big, w_par, w_mid, w_oct, w_unis = W_STEP, W_BIG, W_PAR, W_MID, W_OCT, W_UNIS
    ahead, ndur, length, side, evid = AHEAD, NDUR, LENGTH, SIDE, EVID

    def _lengths(self):
        durs = [n[1] - n[0] for n in self.notes[-(self.ndur + 1):-1] if n[1] is not None]
        if not durs:
            return [FIRST_LEN]
        if self.length == 'med':
            return [float(np.median(durs))]
        return durs

    def _cover(self, b, k, onset, hp):
        """Expected chord distribution over her note's time: share_0 * b + sum_{j>=1} share_j * forecast(beat k+j)."""
        nb = max(1, min(self.ahead + 1, self.K - k))
        lens = self._lengths()
        ov = np.zeros((len(lens), nb))
        for r, d in enumerate(lens):
            end = onset + d
            for j in range(nb):
                o = min(end, k + j + 1) - max(onset, k + j)
                if o <= 0:
                    break
                ov[r, j] = o
        e = ov.sum(0)
        tot = e.sum()
        if tot <= 0:
            return b
        q = (e[0] / tot) * b                          # with nb = 1 this is exactly b
        if not self.evid:
            bj = b
            for j in range(1, nb):
                if e[j] <= 0:
                    break
                bj = bj @ self.m.A[tkey(self._pos(k + j), self.M)]
                q = q + (e[j] / tot) * bj
            return q
        lb = m1_filter_v5lo.P['lam'] * self.m.logB[:, hp]
        for r in range(len(lens)):
            bj = b
            for j in range(1, nb):
                if ov[r, j] <= 0:
                    break
                ll = ov[r, j] * lb
                bj = (bj @ self.m.A[tkey(self._pos(k + j), self.M)]) * np.exp(ll - ll.max())
                bj = bj / bj.sum()
                q = q + (ov[r, j] / tot) * bj
        return q

    def step(self, i, onset, midi, prev_dur):
        self._p = self.ens.chord_probs(onset, midi, prev_dur)   # exactly once per note: advances the GRU state
        if self.notes and prev_dur is not None:
            self.notes[-1][1] = self.notes[-1][0] + prev_dur
        k = int(onset)
        self._finish_beats(k)
        self.notes.append([onset, None, (midi - self.t) % 12])
        hp = (midi - self.t) % 12
        if self.side == 'hmm':
            bh = np.maximum(np.asarray(m1_filter_v5lo.Filter.belief(self, k), dtype=np.float64), FLOOR)
            qh = np.maximum(self._cover(bh, k, onset, hp), FLOOR)
            bg = np.maximum(np.asarray(self._p, dtype=np.float64), FLOOR)
            lg = (1.0 - self.a) * np.log(qh) + self.a * np.log(bg)
            w = np.exp(lg - lg.max())
            pfit = (w / w.sum()) @ CTM
        else:
            pfit = self._cover(self.belief(k), k, onset, hp) @ CTM
        ref = self.prev if self.prev is not None else ALTO_MID
        hi = min(ALTO[1], midi - 1)                            # below = 1: strictly under her heard note
        best, bs = -1, -1e9
        for h in range(ALTO[0], hi + 1):
            iv = abs(midi - h) % 12
            if iv not in CONSONANT:
                continue
            s = pfit[(h - self.t) % 12]
            if iv == 0:
                s -= self.w_oct if h != midi else self.w_unis
            d = abs(h - ref) if self.prev is not None else 0
            s -= self.w_step * d + self.w_big * (d > 7) + self.w_mid * abs(h - ALTO_MID)
            if self.prev is not None and self.prev_m is not None:
                dm, dh = midi - self.prev_m, h - self.prev
                i0 = abs(self.prev_m - self.prev) % 12
                if dm != 0 and dh != 0 and (dm > 0) == (dh > 0) and iv in (0, 7) and i0 == iv:
                    s -= self.w_par
            if s > bs:
                best, bs = h, s
        if best >= 0:
            self.prev = best
        self.prev_m = midi
        return best


def build(train):
    torch.set_num_threads(1)
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0, dict(V3, step=0.01))
    return Cover(m1_filter_v5lo.build(train).m, ens)
