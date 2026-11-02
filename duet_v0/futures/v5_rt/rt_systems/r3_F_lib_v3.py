"""r3_F_lib_v3 (run v5rt_1101, round 3, idea F 'stumble guard'): LIBRARY (no build(): not a system itself). Val only,
tuned on val (no test split). rt_systems.r3_F_lib_v2.PlanF2 (read r3_F_lib_v1 and v2) with a wider hear keep guard,
each off by default (then the decisions are r3_F_lib_v2's exactly):
  g_semi  the keep guard (and the posterior's st_post) also applies when the heard pitch is within g_semi semitones of
          her previous heard pitch, however long ago Duet's note started: on the live tracker most wrong answers and
          most split answers of one long note are 1-2 semitones from the note before, and switching for them starts a
          note against her true pitch (a fresh clash), while the sounding note was chosen for that true pitch.
  g_all   the keep guard applies at every hear point.
The keep rule itself is r3_F_lib_v1's: the sounding note stays if it is consonant and not her unison/octave against
the heard pitch (above her only with g_above) and its chord-tone belief is at least the new choice's minus g_mu (or
>= g_sil when the new choice is silence).
"""
from systems import r2_mix_v2

from rt_systems import r2_lib_v3 as LIB
from rt_systems import r3_F_lib_v2 as F2
from rt_systems.r2_lib_v1 import Prior, load_copy_tab


class PlanF3(F2.PlanF2):
    def _recent(self, onset):
        if super()._recent(onset):
            return True
        c = self.c
        if c.get('g_all', 0):
            return True
        gs = c.get('g_semi', 0)
        her = self.her
        return gs > 0 and len(her) >= 2 and her[-1][0] == onset and abs(her[-1][1] - her[-2][1]) <= gs


def make(train, cfg):
    tab = load_copy_tab() if cfg.get('copy') else None
    hm = LIB.load_hm_tab() if cfg.get('hm_mode') == 'lr' else None
    return PlanF3(r2_mix_v2.build(train), Prior(), cfg, tab, hm)
