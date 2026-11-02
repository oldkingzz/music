"""r3_D_regguardsafe_v1 (run v5rt_1101, round 3, idea D 'register' + a hearing guard): the register adapter
rt_systems.r3_D_lib_v3.RegAdapter (read r3_D_lib_v1 / v2 / v3) around r2_copysafe_v1 with guard = 'hold_sil'.
Same constants as r3_D_regguard_v1 (octave shift into the TRAIN soprano register [62, 74) from her first 4 heard notes,
replay re-sync, Duet = base note - s folded into 48..79; at a heard pitch outside the major scale no new attack: keep the
sounding note if it is not harsh against the heard pitch, else silence); only the base differs (the round-2 copy-model
variant with the fewest fresh clashes). Tuned on val (no test split).
"""
from rt_systems import r2_copysafe_v1 as BASE
from rt_systems import r3_D_lib_v3 as DL

CFG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard='hold_sil')


def build(train):
    return DL.RegAdapter(BASE.build(train), CFG)
