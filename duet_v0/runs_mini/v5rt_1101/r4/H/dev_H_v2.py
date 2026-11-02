"""Idea H dev runner v2 (dev_H_v1 plus --lib, the rt_systems library to build from; run v5rt_1101, round 4): offline,
val only, tuned on val (no test split). Builds the idea-H system on W's constants ONCE, then for each config (overrides
of the core cfg and of the lag wrapper) runs live bench v2's evaluate (futures/v5_rt/live_bench_rt_v2.py, frozen; its
own sha check runs at import) on the given cells and writes one new JSON (mode 'x'). Nothing is played: the _NoSound
tripwire is installed before any import.

  dev_H_v2.py --lib r4_H_lib_v2 --cells sung:bt_cal,sung_low:bt_cal --cfgs '<json list of {"name", "core", "lag"}>'
              [--songs N] --out <new json under runs_mini/v5rt_1101/r4/H/>
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('dev_H is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V5 = os.path.join(ABS, 'futures', 'v5_rt')
HDIR = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r4', 'H')
if V5 not in sys.path:
    sys.path.insert(0, V5)
import live_bench_rt_v2 as LB2  # noqa: E402  (frozen; checks v1, voice_v1 and bench_rt shas)

LB1 = LB2.LB1

W_CFG = dict(
    lam=2.0, kappa=1.0, kappa_hold=0.25,
    mu=0.05, mu_old=-1.0,
    beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0, w_x=0.3, gamma=0.0, min_cov=0.0, silence=False,
    hold_win='beat', post_hold=True, pan_floor=0.0,
    hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False,
    w_next=0.0, w_dyn=1.0, lam_n=1.0, hx=0.0, hedge_post=0,
    copy=True, copy_rhy=1.0, copy_p=1.0,
    hm_mode='lr', hm_rho=1.0, hm_min=6, hm_bar=0, hm_lag=8, hm_plan=1,
    trk=True, eps_u=0.1, mel_w=1.0, tau_t=0.15, p_min=0.1, harsh_max=0.2, kappa_t=1.0, lam_t=1.0, sil_v=-0.2,
    above_att=False, free=True, frag_copy=2.0, hold_trk=True,
    kappa_st=0.3, hop_win=0.55, theta_hop=0.05,
    guard=1, rec_win=0.4, g_mu=0.1, g_sil=0.3, g_above=1,
    doubt_key=0.7,
    doubt_mode='noatt')
W_REG = dict(lo_up=62, hi_down=88, n_min=1, n_dec=4, listen=0.0, listen_at='plan', pre='base',
             out='fold', out_lo=48, out_hi=79, guard=None)
LAG0 = dict(n_win=5, n_min=3, n_need=3, d_on=0.9, grid_tol=0.02)
H = ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100', 'first_100ms_hit', 'on_time_right', 'attacks_per100')
ST = ('lag_plan', 'lag_attack', 'lag_sil', 'plan_attack', 'late_keep', 'late_switch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', default='r4_H_lib_v2')
    ap.add_argument('--cells', required=True)
    ap.add_argument('--cfgs', required=True)
    ap.add_argument('--songs', type=int, default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if not out.startswith(HDIR + os.sep) or os.path.exists(out):
        sys.exit('--out must be a new file under ' + HDIR)
    cells = [tuple(x.split(':')) for x in a.cells.split(',')]
    cfgs = json.loads(a.cfgs)
    t0 = time.time()
    HL = importlib.import_module('rt_systems.' + a.lib)
    train = LB1.bench3.train_sets()
    sysobj = HL.make(train, W_CFG, W_REG, LAG0)
    core = sysobj.core
    print(f'{a.lib} built in {time.time() - t0:.1f} s', flush=True)
    res = []
    for cf in cfgs:
        core.c = dict(W_CFG, **cf.get('core', {}))
        sysobj.c = dict(LAG0, **cf.get('lag', {}))
        for voice, lat in cells:
            st0 = dict(core.stats)
            t1 = time.time()
            r = LB2.evaluate('rt:r4_H_dev', voice, lat, a.songs, system=sysobj)
            if 'headline' in r:
                hd = r['headline']
                n_s = len(r['songs']) + 1
            else:
                hd = r['phone']
                n_s = 1
            st = {k: core.stats[k] - st0.get(k, 0) for k in ST}
            lag_songs = sum(1 for s in sysobj.songs[-n_s:] if s['lag_on_at'] is not None)
            row = {'name': cf['name'], 'voice': voice, 'lat': lat, 'stats': st, 'lag_songs': lag_songs,
                   'secs': round(time.time() - t1, 1)}
            if 'headline' in r:
                row.update({k: hd[k] for k in H})
                print(f"{cf['name']:16s} {voice:8s} {lat:9s} nd {hd['rt_nd_live']:.4f} harsh {hd['harsh']:.4f} "
                      f"fresh {hd['fresh_per100']:.2f} stum {hd['stumbles_per100']:.2f} "
                      f"f100 {hd['first_100ms_hit']:.3f} att {hd['attacks_per100']:.1f} lagsongs {lag_songs} {st} "
                      f"{row['secs']} s", flush=True)
            else:
                row['phone'] = {k: v for k, v in hd.items() if not isinstance(v, (list, dict))}
                print(f"{cf['name']:16s} {voice} {row['phone']} lagsongs {lag_songs} {st}", flush=True)
            res.append(row)
    with open(out, 'x') as fh:
        json.dump({'lib': a.lib, 'cells': cells, 'cfgs': cfgs, 'rows': res,
                   'note': 'val only, tuned on val (no test split)'}, fh, indent=1)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
