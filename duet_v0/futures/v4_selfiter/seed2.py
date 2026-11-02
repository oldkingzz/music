"""Seed-2 scorer for v4_selfiter: music_nd exactly as rescore_nd.py, but her 10 % mishearing is drawn with the
seed ^ 5161 instead of bench.heard's ^ 5160 (a second, independent draw of the same noise). Val split only.
Writes $ABS/runs_mini/<RUN>/seed2/<name>.json and refuses if that file exists (never overwrites).

  cd $ABS/runs_mini/<RUN>/seed2 && $ABS/.venv/bin/python $ABS/futures/v3_nolatency/tl.py 600 \
      $ABS/.venv/bin/python $ABS/futures/v4_selfiter/seed2.py <name>
"""
import importlib
import json
import os
import random
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ABS = os.path.dirname(os.path.dirname(HERE))
RUN = 'v4_1101'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402  (also puts duet_beat on sys.path)
from theory import CONSONANT, CT  # noqa: E402


def heard(song, p=bench3.B2.NOISE):
    """bench.heard with the seed ^ 5161 instead of ^ 5160; everything else identical."""
    rng = random.Random(zlib.crc32(str(song.get('id', song.get('title', ''))).encode()) ^ 5161)
    sop = []
    for on, off, m in song['sop']:
        if rng.random() < p:
            pc = (m + rng.randint(1, 11)) % 12           # one of the other 11 pitch classes
            m = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        sop.append([on, off, m])
    beats = [{'pos': b['pos'], 'z': -1, 'onset_pcs': []} for b in song['beats']]
    return dict(song, sop=sop, beats=beats)


name = sys.argv[1]
path = os.path.join(ABS, 'runs_mini', RUN, 'seed2', f'{name}.json')
if os.path.exists(path):
    sys.exit(f'refusing to overwrite {path}')
bench3.B2.heard = heard                                  # before building the system
mod = importlib.import_module(f'systems.{name}')
sysm = mod.build(bench3.train_sets())
out = {}
for ds, songs in bench3.eval_sets('val').items():
    num = den = dbl = tot = 0.0
    for s in songs:
        hs = bench3.run_song(sysm, s)
        true = sorted(s['sop'], key=lambda n: n[0]); z = [b['z'] for b in s['beats']]; t = s['tonic']
        for i, (on, off, m) in enumerate(true):
            h = hs[i]
            for k in range(max(bench3.J, int(on)), int(off) + 1):
                ov = min(off, k + 1) - max(on, k)
                if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                    continue
                den += ov
                if h >= 0:
                    iv = abs(m - h) % 12
                    num += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])
                    dbl += ov * (iv == 0)
    out[ds] = {'music_nd': num / den, 'double': dbl / den}
out['headline_nd'] = sum(v['music_nd'] for v in out.values()) / 4
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'x') as fh:
    json.dump(out, fh, indent=1)
print(name, 'seed2', round(out['headline_nd'], 4), {k: round(v['music_nd'], 4) for k, v in out.items() if k != 'headline_nd'})
