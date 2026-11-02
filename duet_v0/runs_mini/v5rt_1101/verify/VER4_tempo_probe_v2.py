"""VER4_tempo_probe_v2 (verifier, run v5rt_1101, round 4): REPORT ONLY. VER4_tempo_probe_v1.py with one fix: the lag-regime
facts are read only from a system that has the pick's LagWrap (attribute 'core'); v1 tested for 'info', which W's register
adapter also has, and stopped on W's first song before writing anything (its pick run wrote
VER4_tempo_probe_hgeh_v1.json correctly: the pick's LagWrap has both attributes).

The frozen live bench v2 renders every song at 90 bpm; both of the pick's lag detectors work in beats (H: q_last + 0.5 -
onset >= 0.9 beat for 3 of the last 5 grid notes; G: median q_last - onset >= 0.5 beat), so whether its large-lag code
runs depends on tempo x latency. This probe calls the frozen bench's own per-song runner (live_bench_rt.rendered_song,
imported through live_bench_rt_v2, which checks the frozen sha256 first; neither file is edited) with its bpm argument at
other tempos, for one system per process, on the 17 val Bach chorales, voices sung and sung_low, and pools them as
live_bench_rt.evaluate does (Twinkle left out). Val only, tuned on val, no test split. No sound device (v2's tripwire is
installed before any other import).

Settings x tempos: micbt_cal at 110 / 120 / 130 bpm, bt_cal at 75 and 120 bpm (as v1).

  .venv/bin/python runs_mini/v5rt_1101/verify/VER4_tempo_probe_v2.py <spec> --out runs_mini/v5rt_1101/verify/<new VER4_*.json>
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('VER4 tempo probe is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import live_bench_rt_v2 as LB2  # noqa: E402  (refuses to run unless v1, voice_v1 and bench_rt are the frozen files)

LB1 = LB2.LB1
COMBOS = (('micbt_cal', 110.0), ('micbt_cal', 120.0), ('micbt_cal', 130.0), ('bt_cal', 75.0), ('bt_cal', 120.0))
VOICES = ('sung', 'sung_low')
KEYS = ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out_path = os.path.abspath(a.out)
    vdir = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'verify')
    if not out_path.startswith(vdir + os.sep) or not os.path.basename(out_path).startswith('VER4_') \
            or os.path.exists(out_path):
        sys.exit('--out must be a new VER4_*.json under runs_mini/v5rt_1101/verify/')
    t0 = time.time()
    system = LB1.load_system(a.spec)
    lagwrap = hasattr(system, 'core') and hasattr(system, 'info')
    songs = LB1.bench3.eval_sets(LB1.SPLIT)['bach']
    assert len(songs) == 17
    res = {'spec': a.spec, 'note': 'report only; val only, tuned on val, no test split; Twinkle not included',
           'code': LB2.FROZEN, 'combos': {}}
    for lat_name, bpm in COMBOS:
        for voice in VOICES:
            lat = LB1.LAT[lat_name]
            acc = LB1.new_acc()
            per, lag_on, g_large = {}, 0, 0
            for s in songs:
                ac, r, tr, ex, st = LB1.rendered_song(system, a.spec, s, voice, lat, bpm=bpm)
                for k in LB1.ACC_KEYS:
                    acc[k] += ac[k]
                per[r['id']] = round(r['metrics']['rt_nd_live'], 4)
                if lagwrap:
                    lag_on += system.info.get('lag_on_at') is not None
                    lob = list(getattr(system.core, 'g_lob', []))
                    if len(lob) >= 3:
                        lob = sorted(lob[-16:])
                        med = lob[len(lob) // 2] if len(lob) % 2 else 0.5 * (lob[len(lob) // 2 - 1] + lob[len(lob) // 2])
                        g_large += med >= 0.5
            m = LB1.metrics(acc)
            key = f'{voice}@{lat_name}@{bpm:g}'
            res['combos'][key] = {**{k: m[k] for k in KEYS}, 'per_song': per,
                                  'h_lag_on_songs': lag_on if lagwrap else None,
                                  'g_large_songs_at_end': g_large if lagwrap else None}
            print(f"{a.spec} {key}: " + '/'.join(f'{m[k]:.4f}' if k in ('rt_nd_live', 'harsh') else f'{m[k]:.2f}'
                                                 for k in KEYS)
                  + (f'  H lag on {lag_on}/17, G large {g_large}/17' if lagwrap else '')
                  + f'  ({time.time() - t0:.0f} s)', flush=True)
    res['secs'] = round(time.time() - t0, 1)
    with open(out_path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()
