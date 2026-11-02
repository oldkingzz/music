"""r3_D_reghm_v1 (run v5rt_1101, round 3, idea D 'register'): the register adapter rt_systems.r3_D_lib_v3.RegAdapter
(read r3_D_lib_v1 / v2 / v3) around r2_copyhm_v1, no hearing guard.

From her first heard notes (median of the first 4, decided from the first note on, locked after 4) Duet picks the octave
shift s that puts her melody's median in [62, 74): the TRAIN soprano register the systems learned on (pooled train
note-pitch median 68 +- 6; runs_mini/v5rt_1101/r3/D/reg_stats_v1.json, train only). r2_copyhm_v1 then hears her melody
+ s (re-synced by replay when s changes) and Duet sings the base's note - s, moved by octaves into 48..79 ('fold': the
interval the base chose against the shifted melody, now against her real voice). s = 0 for every val Bach chorale at her
own pitch (then this is r2_copyhm_v1 exactly); s = 12 for the octave-lower 'sung_low' voice and the real low voice v11.
Tuned on val (no test split).
"""
from rt_systems import r2_copyhm_v1 as BASE
from rt_systems import r3_D_lib_v3 as DL

CFG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
           out='fold', out_lo=48, out_hi=79, guard=None)


def build(train):
    return DL.RegAdapter(BASE.build(train), CFG)
