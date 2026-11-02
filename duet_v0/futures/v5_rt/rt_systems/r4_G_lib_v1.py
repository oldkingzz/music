"""r4_G_lib_v1 (run v5rt_1101, round 4, idea G 'late-hear policy'): LIBRARY (no build(): not a system itself). Every
scored variant r4_G_<name>_v<k>.py passes its own constants (rule 7). Val only, tuned on val (this bundle has no test
split).

Base: W = rt_systems.r3_X_lib_v3.PlanEF3 (round 3's page default r3_X_regtrkst_v1: tracker-aware hear, stumble guards,
key-aware doubt rule; idea D's register adapter wraps it in make()). With g_on off (None) every decision is W's exactly.

Why: at a calibrated large latency a hear decision sounds lag = Li + detect + Lo after her onset (bench v2's bt_cal:
about 0.6 beat at 90 bpm; micbt_cal about 0.36; wired about 0.15). W's hedged hear weighs her NEXT note by
w = w_dyn * L0 / max(d - L0, 0.15) with L0 = 0.1 beat (the share of the note's sounding time that bleeds into her next
note when the answer sounds 0.1 beat after her onset; d = her predicted inter-onset interval). At lag 0.6 the answer
sounds on her current note for only d - lag (0.4 beat of a quarter note, nothing of an eighth note: it lands on her next
note) and, if the next plan point keeps it, on her next note until that note's own answer arrives, lag later. So:

1. LAG FROM THE CALL ORDER (the only source; nothing about latency, time or settings is read). At each hear call for
   her note with onset o, the most recent plan point q_last bounds the lag: q_last - o <= lag < q_last + 0.5 - o.
   lob = q_last - o is recorded per answered note (once per note: a register-adapter replay re-hears old notes and is
   not recorded; the samples survive the adapter's re-start inside the same song). The song's lag regime is LARGE when
   at least g_min notes are recorded and the median of the last g_n lob values is >= g_on; then lag_hat = that median
   + g_add. On the live bench v2 the median lob is 0 at wired, micbt and micbt_cal (the plan for o + 0.5 is decided
   after her note is answered) and 0.5 at bt_cal (decided about 120 ms before); in bench_rt it is 0 for L < 0.5 and
   0.5 for L = 0.5 (ties: plan before hear). In the small regime nothing below applies (W exactly).
2. LAG-AWARE HEDGE (large regime): w = w_dyn * lag_hat / max(d - lag_hat, g_floor), at most g_wmax, in place of W's L0
   form, wherever W reads it (the hedged hear's chooser and E's guarded choice's hedged candidate). The chooser still
   allows only notes consonant with (and below) the heard pitch, so the answer stays right for her current note if she
   holds it longer than predicted.
3. LATE OVERRIDE (large regime, g_late > 0): when her note is predicted to end before the answer sounds
   (d <= lag_hat + g_late_pad: the answer lands on her next note) and W's hear (with 2.) attacks a new note x while Duet
   sounds hs, Duet keeps hs unless U_next(x) - U_next(hs) >= g_late_mu (U_next = r2's next-note utility: P(hit) -
   lam_n P(harsh) against her likely next note); x is also refused (keep hs; silence if hs is harsh against her likely
   next note with probability >= g_late_sil) when P(x harsh against her likely next note) > g_late_hmax. A late attack
   that lands on her next note is the main source of fresh clashes and stumbles at large lag (round-4 phase-1
   diagnostic: 121 of W's 701 bt_cal hear attacks, 43 of its 71 fresh clashes, only +0.0115 rt_nd).
4. LATE FOLLOW GUARD (large regime, g_follow > 0): the first plan point after a hear attack made in the large regime
   needs g_follow more (W's plan threshold theta) to move away from it: the hear's note sounds at most a few hundred ms
   before that plan point, so a move there is a stumble when her new note started just before the answer.
Every override sets the base's own state to what Duet really does (sounding note h, m1's voice-leading memory f.prev,
F's attack marks), as W's own keep / silence paths do.

make(train, cfg, reg=None) as r3_X_lib_v3.make, with this class.
"""
import numpy as np

from systems import r2_mix_v2
from theory import ALTO

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_D_lib_v3 as DL
from rt_systems import r3_E_lib_v2 as EL
from rt_systems import r3_X_lib_v3 as XL
from rt_systems.r2_lib_v1 import DV, HARSHV, HV, Prior, load_copy_tab


class PlanG(XL.PlanEF3):
    # ------------------------------------------------------------------ song start (lag samples survive a re-start)
    def start(self, meta):
        super().start(meta)
        if meta is not getattr(self, 'g_meta', None):
            self.g_meta = meta
            self.g_lob = []                                # per answered note: q_last - onset (beats)
            self.g_last_on = -1e9                          # newest onset recorded
            self.g_follow_q = None
        self.q_last = None
        self.g_follow_q = None

    def _g_lag(self):
        """lag_hat (beats) in the large regime, else None (W's behaviour)."""
        c = self.c
        on = c.get('g_on')
        if on is None or len(self.g_lob) < c.get('g_min', 3):
            return None
        lo = float(np.median(self.g_lob[-int(c.get('g_n', 16)):]))
        if lo < on - 1e-9:
            return None
        return lo + c.get('g_add', 0.1)

    # ------------------------------------------------------------------ plan: remember q_last; follow guard
    def plan(self, q, heard):
        self.q_last = q
        fq = self.g_follow_q
        if fq is not None and self.c.get('g_follow', 0) > 0 and q > fq:
            self.g_follow_q = None
            th0 = self.c['theta']
            self.c['theta'] = th0 + self.c['g_follow']
            try:
                r = super().plan(q, heard)
            finally:
                self.c['theta'] = th0
            if r is not None:
                self.stats['g_follow_moved'] += 1
            return r
        return super().plan(q, heard)

    # ------------------------------------------------------------------ hedge weight
    def _w_next(self, onset):
        lam = self._g_lag()
        if lam is None:
            return super()._w_next(onset)
        c = self.c
        d = self._ioi_pred(onset) - onset
        return min(c['w_dyn'] * lam / max(d - lam, c.get('g_floor', 0.15)), c.get('g_wmax', 1e9))

    # ------------------------------------------------------------------ next-note harsh probability
    def _ph_next(self, onset, midi, k):
        """P(Duet's note h is harsh against her likely next note), over HV (r2_lib_v2._u_next's forecast)."""
        her = self.her
        iv = 11 if len(her) < 2 else int(np.clip(midi - her[-2][1], -5, 5)) + 5
        tn = self._ioi_pred(onset)
        pmel = self.pr.mel[(midi - self.t) % 12, iv, self._cls(tn)]
        pmel = self._mix_copy(pmel, midi, self.pred_next)
        b = self._belief(k)
        for j in range(k + 1, min(int(tn + 1e-9), k + 4) + 1):
            b = b @ self.ctrans[self._tkey(j)]
        pz = pmel[None, :] * self.g[:, (midi + DV - self.t) % 12]
        pz /= pz.sum(1, keepdims=True)
        pm = b @ pz
        ivm = (midi + DV[:, None] - HV[None, :]) % 12
        return pm @ HARSHV[ivm]

    # ------------------------------------------------------------------ hear
    def hear(self, i, onset, midi, prev_dur):
        c = self.c
        if self.q_last is not None and onset > self.g_last_on + 1e-9:
            self.g_lob.append(self.q_last - onset)
            self.g_last_on = onset
        lam = self._g_lag()
        hs0, att0 = self.h, (self.att_kind, self.att_t, self.att_on)
        r = super().hear(i, onset, midi, prev_dur)
        if lam is None:
            return r
        self.stats['g_large'] += 1
        if r is None or r < 0 or not (hs0 >= 0 and ALTO[0] <= hs0 <= ALTO[1]):
            if r is not None and r >= 0:
                self._g_attacked(onset)
            return r
        x = int(r)
        d = self._ioi_pred(onset) - onset
        if c.get('g_late', 0) > 0 and d <= lam + c.get('g_late_pad', 0.0):
            self.stats['g_late_pred'] += 1
            k = int(onset)
            u = self._u_next(onset, midi, k)
            ph = self._ph_next(onset, midi, k)
            gain = float(u[x - ALTO[0]] - u[hs0 - ALTO[0]])
            if ph[x - ALTO[0]] > c.get('g_late_hmax', 2.0):
                if ph[hs0 - ALTO[0]] >= c.get('g_late_sil', 2.0):
                    return self._g_set(-1, hs0, att0, 'g_late_sil')
                return self._g_set(hs0, hs0, att0, 'g_late_keep_harsh')
            if gain < c.get('g_late_mu', -1e9):
                return self._g_set(hs0, hs0, att0, 'g_late_keep')
            self.stats['g_late_attack'] += 1
        self._g_attacked(onset)
        return r

    def _g_attacked(self, onset):
        if self.c.get('g_follow', 0) > 0:
            self.g_follow_q = self.q_last if self.q_last is not None else onset

    def _g_set(self, h, hs0, att0, tag):
        """Duet really sounds h (hs0 kept, or -1 silence) instead of W's attack: the base's state follows."""
        self.stats[tag] += 1
        self.att_kind, self.att_t, self.att_on = att0
        if h >= 0:
            self.h = h
            self.f.prev = h
            return None
        self.h = -1
        self.f.prev = hs0
        self.att_kind = self.att_t = self.att_on = None
        return -1


def make_base(mix, prior, cfg, copy_tab, hm_tab, trk_tab):
    return PlanG(mix, prior, cfg, copy_tab if cfg.get('copy') else None,
                 hm_tab if cfg.get('hm_mode') == 'lr' else None, trk_tab if cfg.get('trk') else None)


def make(train, cfg, reg=None):
    base = make_base(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab())
    return DL.RegAdapter(base, reg) if reg else base
