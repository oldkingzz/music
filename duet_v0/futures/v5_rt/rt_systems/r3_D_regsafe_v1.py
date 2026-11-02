"""r3_D_regsafe_v1 (run v5rt_1101, round 3, idea D 'register'): the register adapter rt_systems.r3_D_lib_v3.RegAdapter
(read r3_D_lib_v1 / v2 / v3) around r2_copysafe_v1 (the round-2 copy-model variant with the fewest fresh clashes), no
hearing guard. Same adapter constants as r3_D_reghm_v1: octave shift s from the median of her first 4 heard notes into
[62, 74) (the TRAIN soprano register, pooled train median 68 +- 6), base re-synced by replay, Duet's note = the base's
note - s folded into 48..79. With s = 0 (every val Bach chorale at her own pitch) this is r2_copysafe_v1 exactly.
Tuned on val (no test split).
"""
from rt_systems import r2_copysafe_v1 as BASE
from rt_systems import r3_D_lib_v3 as DL

CFG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard=None)


def build(train):
    return DL.RegAdapter(BASE.build(train), CFG)
