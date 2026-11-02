"""r3_D_regguard_v1 (run v5rt_1101, round 3, idea D 'register' + a hearing guard): the register adapter
rt_systems.r3_D_lib_v3.RegAdapter (read r3_D_lib_v1 / v2 / v3) around r2_copy_v1 with guard = 'hold_sil'.

Register: octave shift s from the median of her first 4 heard notes into [62, 74) (the TRAIN soprano register, pooled
train median 68 +- 6), base re-synced by replay, Duet's note = the base's note - s folded into 48..79.
Guard (not a register rule; measured because the real-voice gate G3 needs it): at a heard pitch outside the song's major
scale (live, mostly the tracker's +-1..2 semitone near-misses or an unstable real voice) Duet attacks no new note: it
keeps its sounding note when that is not harsh against the heard pitch, else it goes silent; the base still hears the
note and its belief follows what Duet really does. Tuned on val (no test split).
"""
from rt_systems import r2_copy_v1 as BASE
from rt_systems import r3_D_lib_v3 as DL

CFG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard='hold_sil')


def build(train):
    return DL.RegAdapter(BASE.build(train), CFG)
