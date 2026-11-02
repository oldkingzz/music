"""check_cfg_G_v1 (run v5rt_1101, round 4, idea G): each scored r4_G system's constants = W's CFG / REG
(rt_systems.r3_X_regtrkst_v1) plus exactly the G keys of its dev config (dev_log_G_v1.jsonl tag), and it reads no
environment. Read only; prints one line per system. Val only (no test split).
"""
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
sys.path.insert(0, os.path.join(ABS, 'duet_beat'))
import importlib  # noqa: E402

LOG = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r4', 'G', 'dev_log_G_v1.jsonl')
PAIRS = {'r4_G_hedge_v1': 'ln2_add0_mu0', 'r4_G_hedgeng_v1': 'c3_ln2_f25_g0', 'r4_G_hedgesafe_v1': 'c1_ln3_f25'}


def main():
    dev = {}
    with open(LOG) as fh:
        for line in fh:
            r = json.loads(line)
            dev[r['spec'].get('tag')] = r['spec'].get('over', {})
    W = importlib.import_module('rt_systems.r3_X_regtrkst_v1')
    for name, tag in PAIRS.items():
        m = importlib.import_module('rt_systems.' + name)
        want = dict(W.CFG, **dev[tag])
        src = open(os.path.join(ABS, 'futures', 'v5_rt', 'rt_systems', name + '.py')).read()
        ok_cfg = dict(m.CFG) == want
        diff = {k: (m.CFG.get(k), want.get(k)) for k in set(m.CFG) | set(want) if m.CFG.get(k) != want.get(k)}
        print(f"{name}: CFG == W + dev[{tag}] {ok_cfg} {diff if diff else ''} | REG == W.REG {m.REG == W.REG} | "
              f"os.environ in source {'environ' in src} | g keys {sorted(k for k in m.CFG if k.startswith('g_') and k not in ('g_mu', 'g_sil', 'g_above'))}")


if __name__ == '__main__':
    main()
