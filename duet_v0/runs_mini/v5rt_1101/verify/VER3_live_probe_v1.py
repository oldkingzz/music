"""VER3_live_probe_v1 (verifier, run v5rt_1101, round 3): extra offline live-bench conditions for ONE engine, built once
per process, using futures/v5_rt/live_bench_rt.py's own functions unchanged (rendered_song, new_acc, ACC_KEYS, metrics,
LAT). Val only (the 17 val Bach chorales); every system was tuned on val; this bundle has no test split.

Conditions, voices 'sung' and 'sung_low':
  base    wired, 90 bpm: the bench's own setting; must reproduce the VER3_* live_bench files exactly.
  bpm72 / bpm110   wired latency at 72 and 110 bpm (the systems assume L0 = 0.1 beat; the live lag is ~80 ms).
  phase   90 bpm, input latency 17 ms instead of 12 (told to the engine, so only the 10 ms block phase and the leading
          silence change).
Also recorded: the register adapter's octave shift per song (system.songs, when the system has one) and the base's
decision counters (system.base.stats) per condition.

Offline: live_bench_rt (imported first) installs the _NoSound tripwire before any other import; nothing is played;
only Bach chorales are rendered (voice_v1 refuses anything else). Writes ONE new JSON (mode 'x') under
runs_mini/v5rt_1101/verify/.

  .venv/bin/python runs_mini/v5rt_1101/verify/VER3_live_probe_v1.py <spec> --out runs_mini/v5rt_1101/verify/<new>.json
"""
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402  (tripwire first)
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
OUT_DIR = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'verify')
WIRED = LBR.LAT['wired']
PHASE = {'Li': 0.017, 'Lo': 0.010, 'Li_told': 0.017, 'Lo_told': 0.010}
CONDS = [('base', WIRED, 90.0), ('bpm72', WIRED, 72.0), ('bpm110', WIRED, 110.0), ('phase', PHASE, 90.0)]


def stats_of(system):
    base = getattr(system, 'base', None)
    st = getattr(base, 'stats', None)
    return Counter(st) if st is not None else Counter()


def song_set(system, spec, voice, lat, bpm):
    songs = LBR.bench3.eval_sets('val')['bach']
    acc = LBR.new_acc()
    per, regs = [], []
    s0 = stats_of(system)
    for s in songs:
        a, r, tr, ex, st = LBR.rendered_song(system, spec, s, voice, lat, bpm=bpm)
        for k in LBR.ACC_KEYS:
            acc[k] += a[k]
        m = r['metrics']
        per.append({'id': s['id'], 'rt_nd_live': m['rt_nd_live'], 'counted_beats': m['counted_beats'],
                    'harsh': m['harsh'], 'fresh': m['fresh'], 'stumbles_per100': m['stumbles_per100'],
                    'notes': m['notes'], 'pickup': r['pickup']})
        reg = getattr(system, 'songs', None)
        if reg:
            info = reg[-1]
            regs.append({'id': s['id'], 's': [list(x) for x in info['s']], 'resync': info['resync'],
                         'first': info['first'], 'guard_hold': info['guard_hold'], 'guard_sil': info['guard_sil']})
    s1 = stats_of(system)
    return LBR.metrics(acc), per, regs, dict(sorted((s1 - s0).items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(OUT_DIR + os.sep) or not os.path.basename(path).startswith('VER3_') \
            or not path.endswith('.json') or os.path.exists(path):
        sys.exit(f'--out must be a new VER3_*.json under {OUT_DIR}')
    c0 = time.perf_counter()
    system = LBR.load_system(a.spec)
    out = {'spec': a.spec, 'split': 'val', 'note': 'val only (17 val Bach chorales); systems tuned on val; no test split',
           'code': {'live_bench_rt.py': LBR.file_sha(os.path.join(LBR.HERE, 'live_bench_rt.py')),
                    'voice_v1.py': LBR.file_sha(os.path.join(LBR.HERE, 'voice_v1.py')),
                    'bench_rt.py': LBR.file_sha(os.path.join(LBR.HERE, 'bench_rt.py'))},
           'conditions': {}}
    for name, lat, bpm in CONDS:
        for voice in ('sung', 'sung_low'):
            m, per, regs, st = song_set(system, a.spec, voice, lat, bpm)
            key = f'{name}_{voice}'
            out['conditions'][key] = {'voice': voice, 'lat': lat, 'bpm': bpm, 'headline': m, 'songs': per,
                                      'reg': regs, 'base_stats': st}
            print(f"{a.spec} {key}: rt_nd_live {m['rt_nd_live']:.4f} harsh {m['harsh']:.4f} fresh/100 "
                  f"{m['fresh_per100']:.2f} stumbles/100 {m['stumbles_per100']:.2f} on_time_right "
                  f"{m['on_time_right']:.4f} | {time.perf_counter() - c0:.0f} s", flush=True)
    out['secs'] = round(time.perf_counter() - c0, 1)
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1, default=LBR._js)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
