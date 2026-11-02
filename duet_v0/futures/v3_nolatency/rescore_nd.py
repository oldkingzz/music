"""Re-score one system with 'music' that does not count copying her note: consonant AND chord tone AND
not unison/octave (her note itself). Writes results/nd_<name>.json. Val split only."""
import importlib, json, sys
import bench3
from theory import CONSONANT, CT

name = sys.argv[1]
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
json.dump(out, open(f'results/nd_{name}.json', 'w'), indent=1)
print(name, round(out['headline_nd'], 3), {k: round(v['music_nd'], 3) for k, v in out.items() if k != 'headline_nd'})
