"""tune_E_v1 (run v5rt_1101, round 3, idea E): tuning harness for rt_systems.r3_E_lib_v2.PlanE. VAL ONLY, tuned on val
(this bundle has no test split). Offline: live_bench_rt is imported first (its _NoSound tripwire); nothing is played.

Each config (JSON, merged over BASE = r2_copyhm_v1's constants + idea E's switches) is built once and scored with the
frozen benches' own code paths, in process:
  live   live_bench_rt.evaluate(..., system=obj) on --voices (sung, sung_low wired; phone_v11), the G2 / G3 numbers
  bench  bench_rt simulate / score_song + attack_metric song_counts on all of val at L = 0.1 (seed 1, or --seed2), via
         runs_mini/v5rt_1101/r1/A/tune_A_v1.run (unchanged), the G1 numbers
Appends one JSON line per config to runs_mini/v5rt_1101/r3/E/tune_log_E_v1.jsonl and prints a summary line.

  .venv/bin/python runs_mini/v5rt_1101/r3/E/tune_E_v1.py [--bench] [--seed2] [--voices sung,sung_low,phone_v11]
      [--songs N] [--tag T] '<json>' ['<json>' ...]
"""
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402  (tripwire first)
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'A'))
from tune_A_v1 import B, run  # noqa: E402
from systems import r2_mix_v2  # noqa: E402
from rt_systems import r2_lib_v3 as R3  # noqa: E402
from rt_systems import r3_E_lib_v2 as E  # noqa: E402
from rt_systems.r2_copyhm_v1 import CFG as HM  # noqa: E402
from rt_systems.r2_lib_v1 import Prior, load_copy_tab  # noqa: E402

LOG = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r3', 'E', 'tune_log_E_v1.jsonl')
BASE = dict(HM, trk=True, eps_u=0.0, mel_w=0.0, tau_t=0.2, p_min=0.1, harsh_max=0.2, kappa_t=1.0, lam_t=1.0,
            sil_v=0.0, above_att=False, free=True, frag_copy=2.0, hold_trk=False)
REACT = {'sung': (0.5936451194275247, 0.0833604908808919, 5.069582504970179),
         'sung_low': (0.4012977048161021, 0.05156959122194172, 3.7773359840954273),
         'phone_v11': (4, 0.15520628683693516)}


def live(obj, voice, n_songs):
    st0 = dict(obj.stats)
    res = LBR.evaluate('rt:tuneE', voice, 'wired', n_songs, system=obj)
    st = {k: v - st0.get(k, 0) for k, v in obj.stats.items() if v - st0.get(k, 0)}
    if voice == 'phone_v11':
        p = res['phone']['proxies']
        out = {'fresh_heard': p['fresh_heard'], 'harsh_heard': p['harsh_heard'], 'attacks': p['attacks'],
               'sung': p['sung'], 'changes': p['changes'], 'on_time_cons': p['on_time_cons']}
        ok = p['fresh_heard'] <= REACT[voice][0] and p['harsh_heard'] <= REACT[voice][1] + 0.005
        txt = f"v11 fh {p['fresh_heard']} hh {p['harsh_heard']:.4f}{'' if ok else '!'}"
    else:
        m = res['headline']
        out = {k: m[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100', 'on_time_right',
                                 'wrong_att_per100', 'fresh', 'late_entry')}
        out['twinkle'] = res['twinkle']['rt_nd_live']
        r = REACT[voice]
        need = r[0] + (0.005 if voice == 'sung' else 0.0)
        ok = m['rt_nd_live'] >= need and m['harsh'] <= r[1] and m['fresh_per100'] <= r[2] + 1e-9
        txt = (f"{voice} {m['rt_nd_live']:.4f}/{m['harsh']:.4f}/{m['fresh_per100']:.2f} st {m['stumbles_per100']:.1f}"
               f"{'' if ok else '!'}")
    out['stats'] = st
    out['ok'] = bool(ok)
    return out, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--voices', default='sung,sung_low,phone_v11')
    ap.add_argument('--songs', type=int, default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('cfgs', nargs='+')
    a = ap.parse_args()
    heard = B.heard_seed2 if a.seed2 else B.HEARD1
    B.bench3.B2.heard = heard
    train = B.bench3.train_sets()
    mix = r2_mix_v2.build(train)
    prior = Prior()
    tab = load_copy_tab()
    hm = R3.load_hm_tab()
    trk = E.load_trk_tab()
    sets = B.bench3.eval_sets('val')
    voices = [v for v in a.voices.split(',') if v]
    for js in a.cfgs:
        cfg = dict(BASE, **json.loads(js))
        obj = E.PlanE(mix, prior, cfg, tab if cfg['copy'] else None, hm, trk if cfg['trk'] else None)
        t0 = time.time()
        rec = {'time': time.strftime('%H:%M:%S'), 'tag': a.tag, 'cfg_delta': json.loads(js), 'cfg': cfg, 'live': {}}
        parts = []
        for v in voices:
            out, txt = live(obj, v, a.songs)
            rec['live'][v] = out
            parts.append(txt)
        if a.bench:
            head, rows, ahead, att, n = run(obj, sets, 0.1, heard)
            rec['bench'] = {'seed2': a.seed2, 'head': head, 'attack_head': ahead,
                            'rows': {d: {k: r[k] for k in ('rt_nd', 'harsh')} for d, r in rows.items()}, 'attack': att}
            parts.append(f"bench{'2' if a.seed2 else ''} {head['rt_nd']:.4f}/{head['harsh']:.4f}/{ahead['fresh']:.2f} "
                         + ' '.join(f"{d[:2]} {r['rt_nd']:.3f}/{r['harsh']:.3f}/{att[d]['fresh']:.1f}"
                                    for d, r in rows.items()) + f" us {head['us_mean']:.0f}")
        rec['secs'] = round(time.time() - t0, 1)
        print(' | '.join(parts) + f" | {rec['secs']:.0f}s | {js}", flush=True)
        with open(LOG, 'a') as fh:
            fh.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')


if __name__ == '__main__':
    main()
