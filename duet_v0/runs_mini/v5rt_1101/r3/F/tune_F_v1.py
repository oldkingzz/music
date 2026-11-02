"""Round 3, idea F (stumble guard) tuning harness. VAL ONLY, tuned on val (this bundle has no test split). Not a scored
run: the scored variants are futures/v5_rt/rt_systems/r3_F_*_v<k>.py run through the frozen bench_rt.py,
attack_metric.py and live_bench_rt.py themselves.

For each setting (JSON overrides on r2_copyhm_v1's CFG) of rt_systems.r3_F_lib_v1.PlanF (r2_mix_v2, prior, copy and
harmonic-memory tables built once):
  bench  bench_rt's simulate / score_song and attack_metric's song_counts (unchanged) on val, L = 0.1, seed 1
  live   live_bench_rt's make_engine / run_engine / score / metrics (unchanged) on the 17 val chorales, wired latency,
         voices sung and sung_low (renders cached across settings: the render does not depend on the system)
  v11    live_bench_rt.phone_run(..., 'phone_v11') proxies (fresh_heard, harsh_heard)
Offline only: the _NoSound tripwire is installed before any other import; nothing is played. Appends one JSON line per
setting to runs_mini/v5rt_1101/r3/F/tune_log_F_v1.jsonl.

  .venv/bin/python runs_mini/v5rt_1101/r3/F/tune_F_v1.py [--no-bench] [--voices sung,sung_low] [--no-v11] [--tag T]
      '<json>' ['<json>' ...]
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('offline tuning: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'tools'))
import live_bench_rt as LBR  # noqa: E402
import attack_metric as AM  # noqa: E402

import numpy as np  # noqa: E402

from rt_systems import r3_F_lib_v1 as FL  # noqa: E402
from rt_systems import r2_lib_v3 as LIB3  # noqa: E402
from rt_systems.r2_lib_v1 import Prior, load_copy_tab  # noqa: E402
from systems import r2_mix_v2  # noqa: E402

B = LBR.BR
LOG = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r3', 'F', 'tune_log_F_v1.jsonl')
BASE = dict(                                    # r2_copyhm_v1's CFG, guards off
    lam=2.0, kappa=1.0, kappa_hold=0.5,
    mu=0.05, mu_old=-1.0,
    beta=1.0, theta=0.0, theta_sil=0.0, nu=1.0, w_x=0.3, gamma=0.0, min_cov=0.0, silence=False,
    hold_win='beat', post_hold=True, pan_floor=0.0,
    hear_tau=0.2, lam_h=1.0, kappa_h=1.0, hear_sil=False,
    w_next=0.0, w_dyn=1.0, lam_n=1.0, hx=0.0, hedge_post=0,
    copy=True, copy_rhy=1.0, copy_p=1.0,
    hm_mode='lr', hm_rho=1.0, hm_min=6, hm_bar=0, hm_lag=8, hm_plan=1)


def bench(obj, L=0.1):
    heard = B.HEARD1
    rows, att = {}, {}
    for ds, songs in B.bench3.eval_sets('val').items():
        acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                             'on_time_right', 'changes', 'late_entry'), 0.0)
        aacc = dict.fromkeys(('notes', 'attacks', 'fresh_clash', 'attack_miss', 'd', 'harsh_fresh', 'harsh_bleed'), 0.0)
        up, uh = [], []
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, a, b = B.simulate(obj, meta, notes, L)
            up += a
            uh += b
            B.score_song(s, T, H, L, acc)
            AM.song_counts(s, T, H, L, aacc)
        rows[ds] = B.row(acc, up, uh, len(songs))
        n = aacc['notes'] or 1
        att[ds] = {'fresh': 100 * aacc['fresh_clash'] / n, 'miss': 100 * aacc['attack_miss'] / n}
    head = {k: sum(r[k] for r in rows.values()) / 4 for k in ('rt_nd', 'harsh', 'nd_first', 'on_time_right', 'us_mean')}
    head['fresh'] = sum(a['fresh'] for a in att.values()) / 4
    return head, {ds: {'rt_nd': r['rt_nd'], 'harsh': r['harsh'], 'fresh': att[ds]['fresh']} for ds, r in rows.items()}


RENDER = {}


def live(obj, voice):
    lat = LBR.LAT['wired']
    T_b = 60.0 / LBR.BPM
    acc = LBR.new_acc()
    for song in LBR.bench3.eval_sets('val')['bach']:
        facts = {'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'bpm': LBR.BPM}
        eng = LBR.make_engine(obj, 'tune', facts, LBR.COUNT_IN)
        t0h = eng.t0 + (lat['Lo'] - lat['Lo_told'])
        key = (song['id'], voice, round(t0h, 6))
        if key not in RENDER:
            RENDER[key] = LBR.VV.render(song, voice, t0h, T_b)
        y, her, vf = RENDER[key]
        x = np.concatenate([np.zeros(int(round(lat['Li'] * LBR.SR))), y.astype(np.float64), np.zeros(LBR.SR)])
        track, notes, blips, secs = LBR.run_engine(eng, x, lat)
        LBR.score(her, sorted(notes + blips), [bt['z'] for bt in song['beats']], song['tonic'], t0h, T_b, acc)
    return LBR.metrics(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-bench', action='store_true')
    ap.add_argument('--voices', default='sung,sung_low')
    ap.add_argument('--no-v11', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('cfgs', nargs='+')
    a = ap.parse_args()
    train = B.bench3.train_sets()
    mix = r2_mix_v2.build(train)
    prior, tab, hm = Prior(), load_copy_tab(), LIB3.load_hm_tab()
    voices = [v for v in a.voices.split(',') if v]
    for js in a.cfgs:
        cfg = dict(BASE, **json.loads(js))
        t0 = time.time()
        rec = {'time': time.strftime('%H:%M:%S'), 'tag': a.tag, 'over': json.loads(js), 'cfg': cfg}
        parts = []
        if not a.no_bench:
            obj = FL.PlanF(mix, prior, cfg, tab, hm)
            head, rows = bench(obj)
            rec['bench'] = {'head': head, 'rows': rows, 'stats': dict(obj.stats)}
            parts.append(f"bench rt_nd {head['rt_nd']:.4f} harsh {head['harsh']:.4f} fresh {head['fresh']:.2f} "
                         f"ontime {head['on_time_right']:.3f} | " + ' '.join(
                             f"{d} {r['rt_nd']:.4f}/{r['harsh']:.4f}/{r['fresh']:.2f}" for d, r in rows.items()))
        rec['live'] = {}
        for v in voices:
            obj = FL.PlanF(mix, prior, cfg, tab, hm)
            m = live(obj, v)
            rec['live'][v] = m
            parts.append(f"{v} {m['rt_nd_live']:.4f} h {m['harsh']:.4f} fr {m['fresh_per100']:.2f} "
                         f"st {m['stumbles_per100']:.2f} wr {m['wrong_att_per100']:.1f} otr {m['on_time_right']:.3f}")
        if not a.no_v11:
            obj = FL.PlanF(mix, prior, cfg, tab, hm)
            out, st = LBR.phone_run(obj, 'tune', 'phone_v11')
            p = out['proxies']
            rec['v11'] = p
            parts.append(f"v11 fh {p['fresh_heard']} hh {p['harsh_heard']:.4f} att {p['attacks']}")
        rec['secs'] = round(time.time() - t0, 1)
        print(' | '.join(parts) + f" | {rec['secs']:.0f}s | {js}", flush=True)
        with open(LOG, 'a') as fh:
            fh.write(json.dumps(rec, default=LBR._js) + '\n')


if __name__ == '__main__':
    main()
