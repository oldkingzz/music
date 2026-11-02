"""Round-4 select step (run v5rt_1101): print the method order of rt_systems.r4_X_lib_v1.PlanX and which class supplies
each method the merge relies on (plan, hear, _w_next, _u_next, _hear_trk, start, _plan_lag, _g_lag), to confirm the
library docstring. Imports only (no system is built, no file is written, nothing is scored).

  check_mro_X_v1.py
"""
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench3  # noqa: E402,F401  (puts duet_beat on sys.path, as bench_rt does)
from rt_systems import r4_X_lib_v1 as XL4  # noqa: E402

C = XL4.PlanX
print('MRO:', ' -> '.join(f'{k.__module__.split(".")[-1]}.{k.__name__}' for k in C.__mro__))
for m in ('plan', 'hear', '_w_next', '_u_next', '_hear_trk', 'start', '_plan_lag', '_g_lag', '_trk_g', '__init__'):
    owner = next(k for k in C.__mro__ if m in k.__dict__)
    print(f'{m:10s} first defined in {owner.__module__}.{owner.__name__}')
