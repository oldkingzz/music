"""r3_E_lib_v2 (run v5rt_1101, round 3, idea E: tracker-aware hearing): LIBRARY for the r3_E_* systems (no build():
not a system itself). v2 = r3_E_lib_v1 (never run, kept unchanged) with three small fixes: a copy prediction whose
next onset has already passed after a skipped fragment is dropped, the redundant _ioi_pred override is gone, and the
plausible set always contains the posterior's mode. Every scored variant r3_E_<name>_v<k>.py passes its own constants
(rule 7). Val only, tuned on val (this bundle has no test split).

Base: r2_lib_v3.PlanR2C (round 2: A's melody prior + expected-value plan + copy model + hedged hear + harmonic memory;
r2_mix_v2's step() still runs exactly once per heard note). With trk = False every decision is PlanR2C's exactly.

Why: the round-2 posterior hear multiplies the melody prior by bench3's synthetic mishearing channel (10 % of notes, any
of the other 11 pitch classes). The live tracker errs differently (runs_mini/v5rt_1101/models/LB_tracker_profile_v1.json
and the answered-note sequences models/E_tracker_seq_v1.json, TRAIN renders only): about 21 % of its answered notes are
split fragments of a held note (vibrato and detune crossing a semitone: the fragment is heard +-1 from the note, or it
is the correction of a misheard onset), errors are +-1..4 semitones back toward her previous note after leaps, and
tritones / 7ths almost never occur. On the val live bench every fresh clash of reactive and of r2_copyhm_v1 that came
from a hear decision came from a misheard note or a split fragment (runs_mini/v5rt_1101/r3/E/D_*.json).

What changes (trk = True):
1. TRACKER POSTERIOR at hear(): P(true - heard = e | c1, onset class x dt, ret), e in -6..6, a train-only lookup table
   (models/E_trk_tab_v1.npz by runs_mini/v5rt_1101/r3/E/fit_trk_tab_E_v1.py): c1 = heard - previous answered pitch,
   the snapped onset on a beat / half beat / quarter / unsnapped, the time since the previous answered onset, and
   whether the heard pitch returns to the one two answered notes back. Mixed with eps_u of a flat channel, and times
   the melody prior (A's prior sharpened by the copy model, as round 2) to the power mel_w * (1 - P(fragment)).
2. GUARDED CHOICE when P(misheard) = 1 - P(e = 0) >= tau_t: Duet may only keep its note, go silent, or attack the
   reactive / hedged choice or (free) another alto note; an attack must have P(harsh against her true pitch) <=
   harsh_max under the posterior and (unless above_att) lie below every plausible true pitch (posterior >= p_min, the
   mode always included) and below the heard pitch. Among those, the best expected  pfit * P(consonant) - lam_t *
   P(harsh) - w_x * P(at or above her) - m1's voice-leading costs; an attack pays kappa_t * P(harsh) more; silence is
   worth sil_v. Below tau_t the hear is round 2's hedged hear unchanged (the round-2 posterior hear with bench3's
   channel is off).
3. frag_copy: an answered note with P(fragment) >= frag_copy is not fed to the copy model (it is not a new note of hers;
   it would break the repeat matching); the copy prediction's next onset stays relative to her last real note (dropped
   once it has passed).
4. hold_trk: the plan's posterior of her held note's true pitch is the tracker posterior of that note (round 2: the
   melody prior times bench3's channel).
"""
import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID

from rt_systems import r2_lib_v3 as R3
from rt_systems.r2_lib_v1 import GOODV, HARSHV, HV, Prior, load_copy_tab

ABS = '/Users/san/claude_sandbox/music/duet_v0'
TRK_TAB = ABS + '/runs_mini/v5rt_1101/models/E_trk_tab_v1.npz'
EV = np.arange(-6, 7)                                     # her true pitch - heard pitch
E0 = 6                                                    # index of e = 0
A0 = ALTO[0]
UNI = np.where(EV == 0, 0.0, 1.0 / (len(EV) - 1))


def load_trk_tab(path=TRK_TAB):
    z = np.load(path)
    assert np.array_equal(z['EV'], EV)
    return {'P3': z['P3'].astype(np.float64), 'PF3': z['PF3'].astype(np.float64), 'F': z['F'].astype(np.float64)}


def gidx(on):
    """Snapped onset class: 0 on a beat, 1 on a half beat, 2 on a quarter, 3 not snapped (fit_trk_tab_E_v1's)."""
    if abs(on * 4 - round(on * 4)) > 1e-6:
        return 3
    f = round((on % 1) * 4) % 4
    return (0, 2, 1, 2)[f]


def dtb3(dt):
    return 0 if dt < 0.35 else 1 if dt < 0.75 else 2


class PlanE(R3.PlanR2C):
    def __init__(self, mix, prior, cfg, copy_tab=None, hm_tab=None, trk=None):
        super().__init__(mix, prior, cfg, copy_tab, hm_tab)
        self.trk = trk

    def start(self, meta):
        super().start(meta)
        self.raw = []                                      # every answered note (onset, heard midi), hear() order
        self.last_post = None                              # tracker posterior (over EV) of her latest answered note
        self.copy_on = None                                # onset of the last note fed to the copy model

    # ------------------------------------------------------------------ tracker posterior
    def _trk_cell(self, onset, midi):
        raw = self.raw
        if not raw:
            return None
        p_on, p1 = raw[-1]
        c1b = int(np.clip(midi - p1, -5, 5)) + 5
        gc = gidx(onset) * 3 + dtb3(onset - p_on)
        ret = int(len(raw) >= 2 and midi == raw[-2][1])
        return c1b, gc, ret

    def _post(self, onset, midi, cell, pf):
        c = self.c
        tp = self.trk['P3'][cell]
        u = c.get('eps_u', 0.0)
        if u > 0:
            tp = (1.0 - u) * tp + u * UNI
        w = c.get('mel_w', 0.0) * (1.0 - pf)
        if w > 0 and self.her:
            her = self.her
            m_b = her[-1][1]
            iv = 11 if len(her) < 2 else int(np.clip(m_b - her[-2][1], -5, 5)) + 5
            pr = self.pr.mel[(m_b - self.t) % 12, iv, self._cls(onset)]
            pr = self._mix_copy(pr, m_b, self.pred_cur)
            d = np.clip(midi + EV - m_b, -12, 12) + 12
            pm = np.maximum(pr[d], 1e-12)
            tp = tp * (pm / pm.sum()) ** w
        return tp / tp.sum()

    def _hold_post(self, heard):
        if (self.c.get('trk') and self.c.get('hold_trk') and self.last_post is not None and self.raw and heard
                and heard[-1][0] == self.raw[-1][0]):
            out = np.zeros(25)
            out[12 + EV] = self.last_post
            return out
        return super()._hold_post(heard)

    def _shifted(self, pred, onset):
        """The copy prediction made after her last real note, its interval re-based on this (skipped) note's onset so
        that onset + ioi is still her predicted next real onset; None once that onset has passed."""
        if pred is None or self.copy_on is None:
            return pred
        ioi = pred['ioi'] - (onset - self.copy_on)
        if ioi <= 0.05:
            return None
        return dict(pred, ioi=ioi)

    # ------------------------------------------------------------------ hear
    def hear(self, i, onset, midi, prev_dur):
        c = self.c
        if not c.get('trk'):
            return super().hear(i, onset, midi, prev_dur)
        cell = self._trk_cell(onset, midi)
        pf = 0.0 if cell is None else float(self.trk['PF3'][cell])
        self.raw.append((onset, midi))
        if self.copy is not None:
            if pf < c.get('frag_copy', 2.0):
                self.pred_cur = self.copy.pred
                self.copy.add(onset, midi)
                self.pred_next = self.copy.pred
                self.copy_on = onset
                if self._hm_on():
                    self._align()
            else:
                self.stats['trk_fragskip'] += 1
                self.pred_next = self._shifted(self.copy.pred, onset)
        if cell is None or not self.her:
            self.last_post = None
            return self._hear_hedge(i, onset, midi, prev_dur)
        post = self._post(onset, midi, cell, pf)
        self.last_post = post
        if 1.0 - post[E0] >= c['tau_t']:
            return self._hear_trk(i, onset, midi, prev_dur, post)
        return self._hear_hedge(i, onset, midi, prev_dur)

    def _hear_trk(self, i, onset, midi, prev_dur, post):
        c, f = self.c, self.f
        prev, prev_m = f.prev, f.prev_m
        h_r = int(f.step(i, onset, midi, prev_dur))       # r2_mix_v2's step: GRU state and filter, exactly once
        self.her.append((onset, midi))
        self.planned = False
        hs = self.h
        k = int(onset)
        pfit = self._belief(k) @ M1.CTM
        mt = midi + EV
        ivm = (mt[:, None] - HV[None, :]) % 12
        pg = post @ GOODV[ivm]
        ph = post @ HARSHV[ivm]
        px = post @ (mt[:, None] <= HV[None, :]).astype(np.float64)
        P = M1.P
        d = np.abs(HV - prev) if prev is not None else np.zeros(len(HV))
        s = (pfit[(HV - self.t) % 12] * pg - c['lam_t'] * ph - c['w_x'] * px
             - P['w_step'] * d - P['w_big'] * (d > 7) - P['w_mid'] * np.abs(HV - ALTO_MID))
        lo = int(min(mt[(post >= c['p_min']) | (post >= post.max())].min(), midi))
        h_c = h_r
        w = self._w_next(onset)
        if (w > 0 or (self._hm_on() and self.shift is not None)) and h_r >= 0:
            v = self._m1vec(midi, pfit, prev, prev_m)
            if w > 0:
                v = v + w * self._u_next(onset, midi, k)
            j = int(np.argmax(v))
            if np.isfinite(v[j]):
                h_c = int(HV[j])
        cand = {}
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        if keep_ok:
            cand[hs] = float(s[hs - A0])

        def att_ok(x):
            return ph[x - A0] <= c['harsh_max'] and (c['above_att'] or x < lo)

        for x in (h_r, h_c):
            if x >= 0 and x != hs and att_ok(x):
                cand[x] = float(s[x - A0] - c['kappa_t'] * ph[x - A0])
        if c['free']:
            for j in range(len(HV)):
                x = int(HV[j])
                if x != hs and x not in cand and att_ok(x):
                    cand[x] = float(s[j] - c['kappa_t'] * ph[j])
        best = max(cand, key=cand.get) if cand else None
        if best is None or cand[best] < c['sil_v']:
            self.stats['trk_sil'] += 1
            self.h = -1
            return -1
        f.prev = best
        if best == hs:
            self.stats['trk_keep'] += 1
            return None
        self.stats['trk_react' if best in (h_r, h_c) else 'trk_attack'] += 1
        self.h = best
        return best


def make(train, cfg):
    tab = load_copy_tab() if cfg.get('copy') else None
    hm = R3.load_hm_tab() if cfg.get('hm_mode') == 'lr' else None
    trk = load_trk_tab() if cfg.get('trk') else None
    return PlanE(r2_mix_v2.build(train), Prior(), cfg, tab, hm, trk)
