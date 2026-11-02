"""check_cfg_G_v2 (run v5rt_1101, round 4, idea G): check_cfg_G_v1 for all r4_G systems, with the library defaults the dev
configs left implicit (g_n 16, g_min 3, g_late_pad 0.0) added to the dev side. Each system's constants must equal W's CFG
/ REG (rt_systems.r3_X_regtrkst_v1) plus its dev config, and its source must not read the environment. Read only.
"""
import importlib
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
sys.path.insert(0, os.path.join(ABS, 'duet_beat'))

LOG = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r4', 'G', 'dev_log_G_v1.jsonl')
DEFAULTS = {'g_n': 16, 'g_min': 3, 'g_late_pad': 0.0}
PAIRS = {'r4_G_hedge_v1': 'ln2_add0_mu0', 'r4_G_hedgeng_v1': 'c3_ln2_f25_g0', 'r4_G_hedgesafe_v1': 'c1_ln3_f25',
         'r4_G_ehedge_v1': 'e1_f25', 'r4_G_ehedgesafe_v1': 'e1_c3_ln3'}


def main():
    dev = {}
    with open(LOG) as fh:
        for line in fh:
            r = json.loads(line)
            dev[r['spec'].get('tag')] = (r['spec'].get('over', {}), r['lib'])
    W = importlib.import_module('rt_systems.r3_X_regtrkst_v1')
    for name, tag in PAIRS.items():
        m = importlib.import_module('rt_systems.' + name)
        over, lib = dev[tag]
        want = dict(W.CFG, **DEFAULTS, **over)
        src = open(os.path.join(ABS, 'futures', 'v5_rt', 'rt_systems', name + '.py')).read()
        diff = {k: (m.CFG.get(k), want.get(k)) for k in set(m.CFG) | set(want) if m.CFG.get(k) != want.get(k)}
        print(f"{name}: CFG == W + dev[{tag}] {dict(m.CFG) == want} {diff if diff else ''} | REG == W.REG {m.REG == W.REG} "
              f"| library {m.GL.__name__.split('.')[-1]} (dev {lib}) | reads environ {'environ' in src}")


if __name__ == '__main__':
    main()
