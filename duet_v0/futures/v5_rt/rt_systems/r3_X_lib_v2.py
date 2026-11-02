"""r3_X_lib_v2 (run v5rt_1101, round 3, combine step): LIBRARY (no build(): not a system itself). Val only, tuned on val
(this bundle has no test split). rt_systems.r3_X_lib_v1.PlanEF (read its docstring: idea E's tracker-aware hear + idea
F's stumble guards; make() adds idea D's register adapter) with one switch, off by default (then every decision is
r3_X_lib_v1's exactly):

  doubt_hold  when the tracker posterior says the heard note is very probably wrong (P(misheard) >= doubt_hold, which
              is far above E's tau_t), Duet attacks no new note at all: it keeps its sounding note if that note is not
              harsh (2nd, tritone, 7th) against the heard pitch, else it falls silent until her next note
              (doubt_mode 'hold_sil'); with doubt_mode 'keep' it keeps any sounding note and stays silent if silent.
              E's guarded choice may still attack a 'safe' note at such a point, safe under the TRAIN tracker table;
              on the one real voice (v11) the table's error pattern did not hold at these points (heard G3 for a sung
              Eb3, heard F3 at the start of a 60-130 ms note of a fast chromatic run), so the guess clashed. Keeping
              or waiting costs at most a late note, which Vincent hears as far less bad than a fresh wrong attack.

make(train, cfg, reg=None) as r3_X_lib_v1.make, with this class.
"""
from systems import r2_mix_v2
from theory import ALTO

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_D_lib_v3 as DL
from rt_systems import r3_E_lib_v2 as EL
from rt_systems import r3_X_lib_v1 as X1
from rt_systems.r2_lib_v1 import Prior, load_copy_tab

HARSH = frozenset({1, 2, 6, 10, 11})


class PlanEF2(X1.PlanEF):
    def _hear_trk(self, i, onset, midi, prev_dur, post):
        c = self.c
        if 1.0 - float(post[EL.E0]) < c.get('doubt_hold', 2.0):
            return super()._hear_trk(i, onset, midi, prev_dur, post)
        f = self.f
        prev0 = f.prev
        f.step(i, onset, midi, prev_dur)                  # r2_mix_v2's step: GRU state and filter, exactly once
        self.her.append((onset, midi))
        self.planned = False
        hs = self.h
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        if keep_ok and (c.get('doubt_mode', 'hold_sil') == 'keep' or abs(midi - hs) % 12 not in HARSH):
            f.prev = hs
            self.stats['doubt_keep'] += 1
            return None
        f.prev = hs if hs >= 0 else prev0                 # Duet's last sung note, as m1 keeps it on a silence
        self.stats['doubt_sil'] += 1
        self.h = -1
        return -1 if hs >= 0 else None


def make_base(mix, prior, cfg, copy_tab, hm_tab, trk_tab):
    return PlanEF2(mix, prior, cfg, copy_tab if cfg.get('copy') else None,
                   hm_tab if cfg.get('hm_mode') == 'lr' else None, trk_tab if cfg.get('trk') else None)


def make(train, cfg, reg=None):
    base = make_base(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab())
    return DL.RegAdapter(base, reg) if reg else base
