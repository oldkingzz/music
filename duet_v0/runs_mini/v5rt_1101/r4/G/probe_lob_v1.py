"""probe_lob_v1 (run v5rt_1101, round 4, idea G): what the call-order lag bound looks like per latency setting. Runs W's
decisions (r4_G_lib_v2 with g_on None = W exactly) through the frozen live bench v2's own functions and records, for
every answered note, lob = q_last - onset (q_last = the most recent plan point before the hear call). Offline (tripwire),
val only (17 val chorales; no test split). Prints the distribution per cell and per song (median, mean, share >= 0.25).

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r4/G/probe_lob_v1.py sung@wired sung@micbt_cal ...
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('probe_lob_v1 is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import os  # noqa: E402
from collections import Counter  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import live_bench_rt_v2 as LB2  # noqa: E402

LB1 = LB2.LB1
np = LB1.np
from systems import r2_mix_v2  # noqa: E402
from rt_systems import r2_lib_v3 as R3  # noqa: E402
from rt_systems import r3_D_lib_v3 as DL  # noqa: E402
from rt_systems import r3_E_lib_v2 as EL  # noqa: E402
from rt_systems import r3_X_regtrkst_v1 as W  # noqa: E402
from rt_systems import r4_G_lib_v2 as G2  # noqa: E402
from rt_systems.r2_lib_v1 import Prior, load_copy_tab  # noqa: E402

SONGS = []


class Probe(G2.PlanG2):
    def start(self, meta):
        if meta is not getattr(self, 'g_meta', None) and getattr(self, 'g_lob', None) is not None:
            SONGS.append(list(self.g_lob))
        super().start(meta)


def main():
    train = LB1.bench3.train_sets()
    mix = r2_mix_v2.build(train)
    prior, tab, hm, trk = Prior(), load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab()
    for cell in sys.argv[1:]:
        voice, lat = cell.split('@')
        cfg = dict(W.CFG)
        base = Probe(mix, prior, cfg, tab, hm, trk)
        obj = DL.RegAdapter(base, dict(W.REG))
        SONGS.clear()
        for s in LB1.bench3.eval_sets('val')['bach']:
            LB1.rendered_song(obj, 'probe', s, voice, LB1.LAT[lat])
        SONGS.append(list(base.g_lob))
        allv = [round(x, 3) for sg in SONGS for x in sg]
        cnt = Counter(allv)
        print(f'{cell}: notes {len(allv)} ' + ' '.join(f'{k:+.2f}:{v}' for k, v in sorted(cnt.items())), flush=True)
        rows = []
        for sg in SONGS:
            if not sg:
                continue
            a = np.array(sg)
            rows.append(f'{np.median(a):+.2f}/{a.mean():+.3f}/{(a >= 0.25 - 1e-9).mean():.2f}/{(a <= -0.25 + 1e-9).mean():.2f}')
        print('   per song median/mean/share>=0.25/share<=-0.25: ' + ' '.join(rows), flush=True)


if __name__ == '__main__':
    main()
