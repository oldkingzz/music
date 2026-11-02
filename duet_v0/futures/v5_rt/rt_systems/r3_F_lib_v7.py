"""r3_F_lib_v7 (run v5rt_1101, round 3, idea F 'stumble guard'): LIBRARY (no build(): not a system itself). Val only,
tuned on val (no test split). r3_F_lib_v6's 'when in doubt, wait' (read it) with m1's voice-leading reference kept as
m1 keeps it on a silence (Duet's last sung note; r3_F_lib_v6 reset it to None and was never run). Built on
rt_systems.r3_F_lib_v5.PlanF5 (read r3_F_lib_v1 .. v5); each switch is off by default (then the decisions are
r3_F_lib_v5's exactly). They act only where the posterior hear doubts the heard pitch (P(misheard) >= hear_tau) and the
doubt is not backed by the copy model (post_copy):
  post_doubt_sil   the sounding note is harsh against the heard pitch (so post_veto cannot keep it): Duet falls silent
                   instead of attacking a guess.
  post_doubt_wait  Duet is silent: it stays silent for this note instead of entering on a guess.
The hear of her next note enters as usual.
"""
import numpy as np

from systems import r2_mix_v2
from theory import ALTO

from rt_systems import r2_lib_v3 as LIB
from rt_systems import r3_F_lib_v5 as F5
from rt_systems.r2_lib_v1 import E_ALL, HARSHV, Prior, load_copy_tab


class PlanF7(F5.PlanF5):
    def _hear_post(self, i, onset, midi, prev_dur, post):
        c = self.c
        thr = c.get('post_copy', 2.0)
        pred = self.pred_cur
        if thr <= 1.0 and pred is not None:
            j = int(np.argmax(post))
            if E_ALL[j] != 0 and midi + int(E_ALL[j]) == pred['top'] and post[j] >= thr:
                return super()._hear_post(i, onset, midi, prev_dur, post)      # copy-backed doubt: v5's rule
        hs = self.h
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        hs_harsh = keep_ok and HARSHV[(midi - hs) % 12] > 0
        wait = (c.get('post_doubt_sil', 0) and hs_harsh) or (c.get('post_doubt_wait', 0) and hs < 0)
        if not wait:
            return super()._hear_post(i, onset, midi, prev_dur, post)
        f = self.f
        prev0 = f.prev
        f.step(i, onset, midi, prev_dur)                  # r2_mix_v2's step: GRU state and filter, exactly once
        self.her.append((onset, midi))
        self.planned = False
        f.prev = hs if hs >= 0 else prev0                 # Duet's last sung note, as m1 keeps it on a silence
        self.stats['hear_post_wait'] += 1
        self.h = -1
        return -1 if hs >= 0 else None


def make(train, cfg):
    tab = load_copy_tab() if cfg.get('copy') else None
    hm = LIB.load_hm_tab() if cfg.get('hm_mode') == 'lr' else None
    return PlanF7(r2_mix_v2.build(train), Prior(), cfg, tab, hm)
