"""r3_F_lib_v5 (run v5rt_1101, round 3, idea F 'stumble guard'): LIBRARY (no build(): not a system itself). Val only,
tuned on val (no test split). rt_systems.r3_F_lib_v4.PlanF4 (read r3_F_lib_v1 .. v4) plus one switch, off by default
(then the decisions are r3_F_lib_v4's exactly):
  post_copy  the cautious posterior rules (post_veto, post_below, post_nh, post_nh_keep) are lifted when the doubt is
             backed by the song's own repeat: the copy model predicted a pitch for this note, the posterior's most
             likely true pitch is that pitch (not the heard one) with probability >= post_copy. The posterior then
             chooses as in round 2 (r2_lib_v2's _hear_post): e.g. on the Twinkle phone file the tracker heard F4 for
             a G4 after a leap; the repeat said G4, and round 2 kept its G3 where the cautious rule moved to A3 (a
             seventh against her G4).
"""
import numpy as np

from systems import r2_mix_v2

from rt_systems import r2_lib_v3 as LIB
from rt_systems import r3_F_lib_v4 as F4
from rt_systems.r2_lib_v1 import E_ALL, Prior, load_copy_tab

CAUTIOUS = ('post_veto', 'post_below', 'post_nh', 'post_nh_keep')


class PlanF5(F4.PlanF4):
    def _hear_post(self, i, onset, midi, prev_dur, post):
        c = self.c
        thr = c.get('post_copy', 2.0)
        pred = self.pred_cur
        if thr <= 1.0 and pred is not None:
            j = int(np.argmax(post))
            if E_ALL[j] != 0 and midi + int(E_ALL[j]) == pred['top'] and post[j] >= thr:
                saved = {k: c.get(k, 0) for k in CAUTIOUS}
                for k in CAUTIOUS:
                    c[k] = 0
                try:
                    self.stats['hear_post_copy'] += 1
                    return super()._hear_post(i, onset, midi, prev_dur, post)
                finally:
                    c.update(saved)
        return super()._hear_post(i, onset, midi, prev_dur, post)


def make(train, cfg):
    tab = load_copy_tab() if cfg.get('copy') else None
    hm = LIB.load_hm_tab() if cfg.get('hm_mode') == 'lr' else None
    return PlanF5(r2_mix_v2.build(train), Prior(), cfg, tab, hm)
