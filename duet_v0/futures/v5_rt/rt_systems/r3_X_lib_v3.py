"""r3_X_lib_v3 (run v5rt_1101, round 3, combine step): LIBRARY (no build(): not a system itself). Val only, tuned on val
(this bundle has no test split). rt_systems.r3_X_lib_v2.PlanEF2 (read r3_X_lib_v1 and v2) with the doubt rule made
more targeted; with doubt_key off and doubt_mode 'keep' or 'hold_sil' every decision is r3_X_lib_v2's exactly.

The doubt rule acts only inside idea E's guarded choice (P(misheard) >= tau_t). It fires when
  P(misheard) >= doubt_hold                                            (any heard pitch, as r3_X_lib_v2), or
  P(misheard) >= doubt_key and the heard pitch is outside the song's major scale (tonic + theory.MAJOR).
The second form joins idea D's observation (a heard pitch outside the key is suspicious live: the val melodies are
93-100 % in the major scale of the tonic, while the tracker's errors are +-1..2 semitone near-misses) with idea E's
posterior, so a CONFIDENTLY heard chromatic note is answered as usual (D's own guard blocked every out-of-key heard
note and cost about 0.01 rt_nd_live on the sung voice).
When it fires, Duet attacks no new note:
  doubt_mode 'keep'      keep the sounding note (silent stays silent);
  doubt_mode 'hold_sil'  keep it if it is not harsh against the heard pitch, else fall silent (r3_X_lib_v2's default);
  doubt_mode 'noatt'     E's guarded choice with its attack candidates removed: keep the sounding note if its expected
                         score under the tracker posterior is at least sil_v, else silence.

make(train, cfg, reg=None) as r3_X_lib_v1.make, with this class.
"""
from systems import r2_mix_v2
from theory import ALTO, MAJOR

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_D_lib_v3 as DL
from rt_systems import r3_E_lib_v2 as EL
from rt_systems import r3_X_lib_v2 as X2
from rt_systems.r2_lib_v1 import Prior, load_copy_tab

HARSH = X2.HARSH


class PlanEF3(X2.PlanEF2):
    def _hear_trk(self, i, onset, midi, prev_dur, post):
        c = self.c
        pm = 1.0 - float(post[EL.E0])
        fire = pm >= c.get('doubt_hold', 2.0) or ((midi - self.t) % 12 not in MAJOR and pm >= c.get('doubt_key', 2.0))
        if not fire:
            return EL.PlanE._hear_trk(self, i, onset, midi, prev_dur, post)
        mode = c.get('doubt_mode', 'hold_sil')
        if mode == 'noatt':
            hm0 = c['harsh_max']
            c['harsh_max'] = -1.0                          # no attack candidate passes E's harsh bound
            try:
                r = EL.PlanE._hear_trk(self, i, onset, midi, prev_dur, post)
            finally:
                c['harsh_max'] = hm0
            self.stats['doubt_noatt'] += 1
            return r
        f = self.f
        prev0 = f.prev
        f.step(i, onset, midi, prev_dur)                  # r2_mix_v2's step: GRU state and filter, exactly once
        self.her.append((onset, midi))
        self.planned = False
        hs = self.h
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        if keep_ok and (mode == 'keep' or abs(midi - hs) % 12 not in HARSH):
            f.prev = hs
            self.stats['doubt_keep'] += 1
            return None
        f.prev = hs if hs >= 0 else prev0                 # Duet's last sung note, as m1 keeps it on a silence
        self.stats['doubt_sil'] += 1
        self.h = -1
        return -1 if hs >= 0 else None


def make_base(mix, prior, cfg, copy_tab, hm_tab, trk_tab):
    return PlanEF3(mix, prior, cfg, copy_tab if cfg.get('copy') else None,
                   hm_tab if cfg.get('hm_mode') == 'lr' else None, trk_tab if cfg.get('trk') else None)


def make(train, cfg, reg=None):
    base = make_base(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab())
    return DL.RegAdapter(base, reg) if reg else base
