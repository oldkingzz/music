"""r4_true_cons_v1 (v4 round 4, idea E5, belief side only): r3_style_mix_v3, except that a SUSPECT heard note (out of the major scale and
not a pitch class she keeps singing) does not move its own chord belief: HMM without its evidence, GRU part = last trusted note's guess."""
import numpy as np
import torch

from systems import m1_filter_v5lo
from systems import r3_style_mix_v3 as champ
from theory import ALTO, ALTO_MID, CONSONANT, MAJOR

CTM = m1_filter_v5lo.CTM   # [chord, relative pc] = 1 for a chord tone (read only)
W_STEP = 0.01              # m1_filter_v5lo's 6 chooser weights, unchanged
W_BIG = 0.4
W_PAR = 0.6
W_MID = 0.004
W_OCT = 0.3
W_UNIS = 1.0               # inert: Duet stays strictly below her heard note
MIN_COUNT = 2              # r3_true_cons_v4's trust rule: an out-of-scale pc heard on >= 2 of her earlier notes ...
MIN_SHARE = 0.03           # ... and on >= 3 % of them is her real chromatic note (a mishearing hits one pc on ~0.9 %)
BELIEF = 'cur'             # 'cur': a suspect note leaves its own chord belief alone; 'all': suspect notes never enter the HMM; 'off'
GRU_SUSPECT = 'prev'       # GRU part of the mix at a suspect note: 'prev' = the GRU guess at her last trusted note; 'heard'
CHOICE = 'off'             # 'v4': r3_true_cons_v4's scoring (expected consonance with her TRUE note); 'off': the champion's hard rule
TRUE_LO = -6               # her true note is placed at heard + ((t - heard + 6) % 12) - 6, i.e. -6..+5 (as r3_true_cons_v1)


class TrueConsBelief(champ.StyleMix):
    """r3_style_mix_v3 (style-aware HMM x 3-GRU average, m1's chooser) with E5 on the belief side and optionally the choice side.
    belief_side = choice_side = 'off' sings exactly the champion's notes."""
    belief_side = BELIEF
    gru_suspect = GRU_SUSPECT
    choice_side = CHOICE
    true_lo = TRUE_LO

    def start(self, meta):
        super().start(meta)
        self._sus = []          # per heard note: suspect at its onset?
        self._trusted = None    # 3-GRU chord probabilities at her last trusted (not suspect) note
        self._cut = False       # True while the belief of her current note must leave that note out

    def suspect(self, hp):
        """Her current heard pc (already appended to self.notes) is out of the scale and not one she keeps singing."""
        if hp in MAJOR:
            return False
        earlier = [n[2] for n in self.notes[:-1]]          # relative pcs of her earlier heard notes in this song
        c = earlier.count(hp)
        return not (c >= MIN_COUNT and c >= MIN_SHARE * len(earlier))

    def _beat_v(self, k):
        """r2_style_hmm_v1's beat vector; without the notes this class does not trust ('cur': the current one while
        _cut, 'all': every suspect note, finished beats included)."""
        if not self._cut and self.belief_side != 'all':
            return super()._beat_v(k)
        v = np.zeros(12)
        last = len(self.notes) - 1
        for j in range(last, -1, -1):
            on, off, pc = self.notes[j]
            e = off if off is not None else k + 1
            if e <= k:
                break
            if (j == last and self._cut) or (self.belief_side == 'all' and self._sus[j]):
                continue
            ov = min(e, k + 1) - max(on, k)
            if ov > 0:
                v[pc] += ov
        return v

    def step(self, i, onset, midi, prev_dur):
        p = self.ens.chord_probs(onset, midi, prev_dur)        # exactly once per note, heard note: advances the GRU state
        if self.notes and prev_dur is not None:
            self.notes[-1][1] = self.notes[-1][0] + prev_dur
        k = int(onset)
        self._finish_beats(k)
        hp = (midi - self.t) % 12
        self.notes.append([onset, None, hp])
        sus = self.suspect(hp)
        self._sus.append(sus)
        cut = self.belief_side != 'off' and sus
        self._p = self._trusted if (cut and self.gru_suspect == 'prev' and self._trusted is not None) else p
        if not cut:
            self._trusted = p
        self._cut = cut
        b = self.belief(k)                                     # normalized b_styleHMM^0.5 * b_GRU^0.5
        self._cut = False
        pfit = b @ CTM                                         # P(pc is a chord tone), relative pcs
        robust = self.choice_side == 'v4' and (hp in MAJOR or sus)   # r3_true_cons_v4: all but a recurring chromatic pc
        if robust:
            # joint over (chord z, her TRUE pc t) given what was heard: P(t | z, heard) ~ P(t | z) * P(heard | t)
            pt = self.m.Bc * self.m.N[:, hp][None, :]          # [z, t], m1's pooled model
            pt /= pt.sum(1, keepdims=True)
            jz = b[:, None] * pt                               # [z, t]
            lo = self.true_lo
            tm = np.array([midi + ((t - hp - lo) % 12) + lo for t in range(12)])   # true midi for each t, near the heard one
        ref = self.prev if self.prev is not None else ALTO_MID
        hi = min(ALTO[1], midi - 1)                            # below her heard note
        best, bs = -1, -1e9
        for h in range(ALTO[0], hi + 1):
            iv = abs(midi - h) % 12
            if robust:
                cv = np.fromiter(((abs(x - h) % 12) in CONSONANT for x in tm), float, 12)
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


def make(cls, train):
    """The champion's build (style-aware HMM + the 3 GRUs full 96, big 128, s1 96 via m3_seq_v4) with this file's class."""
    torch.set_num_threads(1)
    base = champ.build(train)
    return cls(base.models, base.lams, base.rh, base.mprior, base.ens)


def build(train):
    return make(TrueConsBelief, train)
