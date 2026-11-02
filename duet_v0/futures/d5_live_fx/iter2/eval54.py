"""A5.4 on vocadito with the iter2 engine, scored by the builder's clash.audio (imported unmodified).
    .venv/bin/python futures/d5_live_fx/iter2/eval54.py --ids 1-20 --kw '{"unv_close": 16}' --tag x
"""
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import clash  # noqa  (builder's scorer, unmodified)
sys.path.insert(0, str(HERE))
import importlib.util
spec = importlib.util.spec_from_file_location('fx2', HERE / 'fx.py')
fx2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx2)
clash.fx = fx2

ap = argparse.ArgumentParser()
ap.add_argument('--ids', default='1-20')
ap.add_argument('--kw', default='{}')
ap.add_argument('--tag', default='run')
a = ap.parse_args()
lo, hi = map(int, a.ids.split('-'))
fx2.DEFAULTS.update(json.loads(a.kw))
t = time.time()
r = clash.audio(ids=range(lo, hi + 1))
out = {'ids': a.ids, 'kw': fx2.DEFAULTS, 'all': r['all'], 'ideal': r['ideal_follower_of_her_pitch'],
       'per': [{k: v for k, v in p.items() if k in ('id', 'good', 'harsh', 'fourth', 'silent', 'not_good')} for p in r['per_excerpt']]}
(HERE / 'runs').mkdir(exist_ok=True)
(HERE / 'runs' / f'{a.tag}_{a.ids}.json').write_text(json.dumps(out, indent=1, default=float))
print(a.tag, a.ids, {k: round(v, 4) for k, v in r['all'].items()}, 'ideal', round(r['ideal_follower_of_her_pitch']['not_good'], 4), f'{time.time()-t:.0f}s', flush=True)
