"""r4_cover_fit_v1 (v4 round 4, idea E4, last attempt): r3_style_mix_v3's belief + m1_filter_v5lo's chooser (copied in), but a
note is rated by the expected share of her note's time on which it is a chord tone (each style's HMM forecast by its own table)."""
import numpy as np
import torch

from systems import r2_style_hmm_v1 as sty
from systems.m1_filter_v5lo import CTM, tkey
from systems.m3_seq_v4 import V3, Ens, _load
from systems.r3_style_mix_v3 import FLOOR, StyleMix
from theory import ALTO, ALTO_MID, CONSONANT

# m1_filter_v5lo's six chooser weights (its P['w_*'] values), all unchanged
W_STEP = 0.01     # cost per semitone that Duet's line moves
W_BIG = 0.4       # extra cost of a leap of more than 7 semitones
W_PAR = 0.6       # cost of a parallel 5th/8ve (both voices move the same way into the same interval)
W_MID = 0.004     # cost per semitone away from the middle of the alto range
W_OCT = 0.3       # cost of an octave with her note
W_UNIS = 1.0      # cost of a unison with her note (inert: Duet stays strictly below her)

# E4: which beats her note will cover (causal: only her finished notes and the beat grid)
AHEAD = 8         # beats after her onset beat that are forecast (0 = onset beat only = r3_style_mix_v3 exactly)
NDUR = 32         # at most NDUR lengths of her finished notes that started like this one (on the beat / off it) ...
WINDOW = 128      # ... from her last WINDOW finished notes; fewer than 2 such notes -> her last NDUR notes of any kind
FIRST_LEN = 1.0   # length (beats) assumed before she has finished any note
MIX = 'beat'      # 'beat': each covered beat's HMM forecast is mixed with the GRU, then averaged by time share;
                  # 'cover': the time-share average of the HMM forecasts is mixed with the GRU once


class CoverStyle(StyleMix):
    """r3_style_mix_v3's HMM, GRU and mix (a = 0.5); m1's chooser path (robust 0, look 0, below 1, order2 0) copied in, with
    P(chord tone at her onset beat) replaced by sum_j share_j * P(chord tone at beat k+j). The HMM part of beat k+j is
    sum_s P(style s) * (style s's belief at beat k) @ A_s ... A_s (style s's own chord-change tables, per bar position)."""
    w_step, w_big, w_par, w_mid, w_oct, w_unis = W_STEP, W_BIG, W_PAR, W_MID, W_OCT, W_UNIS
    ahead, ndur, window, mix = AHEAD, NDUR, WINDOW, MIX

    def _lengths(self):
        """Her recent finished note lengths that started like the current note (on the beat, or off it)."""
        cur = self.notes[-1][0]
        off_beat = (cur - int(cur)) > 1e-6
        prev = [n for n in self.notes[-(self.window + 1):-1] if n[1] is not None]
        same = [n[1] - n[0] for n in prev if ((n[0] - int(n[0])) > 1e-6) == off_beat][-self.ndur:]
        if len(same) >= 2:
            return same
        durs = [n[1] - n[0] for n in prev][-self.ndur:]
        return durs or [FIRST_LEN]

    def _shares(self, k, onset):
        """Expected share of her note's time in beats k, k+1, ... (each length of the estimate counted equally)."""
        nb = max(1, min(self.ahead + 1, self.K - k))
        if nb == 1:
            return np.ones(1)
        e = np.zeros(nb)
        for d in self._lengths():
            end = onset + d
            for j in range(nb):
                o = min(end, k + j + 1) - max(onset, k + j)
                if o <= 0:
                    break
                e[j] += o
        tot = e.sum()
        if tot <= 0 or e[0] >= tot:
            return np.ones(1)
        last = int(np.nonzero(e)[0].max())
        return e[:last + 1] / tot

    def _mixed(self, bh, bg):
        """r3_style_mix_v3.belief's log-linear mix of an HMM belief with the GRU belief, same operations."""
        bh = np.maximum(np.asarray(bh, dtype=np.float64), FLOOR)
        lg = (1.0 - self.a) * np.log(bh) + self.a * np.log(bg)
        w = np.exp(lg - lg.max())
        return w / w.sum()

    def _pfit(self, k, onset):
        v = self._beat_v(k)
        bs = np.array([self._step(j, k, v) for j in range(len(self.models))])   # each style's belief at beat k
        wt = np.exp(self.logw - self.logw.max())
        wt = wt / wt.sum()                                                    # running style probability
        bg = np.maximum(np.asarray(self._p, dtype=np.float64), FLOOR)         # 3-GRU average at her current note
        b0 = self._mixed(wt @ bs, bg)                                         # = r3_style_mix_v3.belief(k)
        sh = self._shares(k, onset)
        if len(sh) == 1:
            return b0 @ CTM
        fc, hmm = bs, [wt @ bs]
        for j in range(1, len(sh)):
            key = tkey(self._pos(k + j), self.M)
            fc = np.array([fc[s] @ self.models[s].A[key] for s in range(len(self.models))])
            hmm.append(wt @ fc)
        if self.mix == 'cover':
            q = sh[0] * hmm[0]
            for j in range(1, len(sh)):
                q = q + sh[j] * hmm[j]
            return self._mixed(q, bg) @ CTM
        q = sh[0] * b0
        for j in range(1, len(sh)):
            q = q + sh[j] * self._mixed(hmm[j], bg)
        return q @ CTM

    def step(self, i, onset, midi, prev_dur):
        self._p = self.ens.chord_probs(onset, midi, prev_dur)   # exactly once per note: advances the GRU state
        if self.notes and prev_dur is not None:
            self.notes[-1][1] = self.notes[-1][0] + prev_dur
        k = int(onset)
        self._finish_beats(k)
        self.notes.append([onset, None, (midi - self.t) % 12])
        pfit = self._pfit(k, onset)
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
    base = sty.build(train)
    ens = Ens([_load('full', 96), _load('big', 128), _load('s1', 96)], -1.0, dict(V3, step=0.01))
    return CoverStyle(base.models, base.lams, base.rh, base.mprior, ens)
