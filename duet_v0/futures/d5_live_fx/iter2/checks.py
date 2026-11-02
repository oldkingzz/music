"""A5.1 / A5.2 / A5.3 with the iter2 engine, using the builder's latency.py and sim5.py (imported unmodified,
their `fx` swapped for iter2/fx.py). Writes iter2/results/checks.json.
    .venv/bin/python futures/d5_live_fx/iter2/checks.py --kw '{...}'
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import importlib.util
spec = importlib.util.spec_from_file_location('fx2', HERE / 'fx.py')
fx2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx2)
import sim5, latency  # noqa  (builder's, unmodified)
sim5.fx = fx2
latency.fx = fx2
KW = {}


def init(kw):
    fx2.DEFAULTS.update(kw)


def job(a):
    if a == 'lat':
        return {'onset': latency.onset_latency(64), 'path_delay': latency.path_delay(64), 'pitch_step': latency.pitch_step(64)}
    r = sim5._job(a)
    r.pop('gate_metrics_open_loop', None)
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='new')
    a = ap.parse_args(); kw = json.loads(a.kw)
    jobs = ['lat', (180.0, 0.6, 0.0, True, True, None), (180.0, 0.3, 0.0, True, True, None),
            (60.0, 0.6, 24.0, True, True, None), (60.0, 0.3, 24.0, True, True, None)]
    with Pool(3, initializer=init, initargs=(kw,)) as p:
        res = p.map(job, jobs)
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / f'checks_{a.tag}.json').write_text(json.dumps({'kw': kw, 'latency': res[0], 'sim': res[1:]}, indent=1, default=float))
    L = res[0]
    print('onset_ms', [round(d['onset_ms'], 2) for d in L['onset']], 'path_ms', [round(d['delay_ms'], 2) for d in L['path_delay']],
          'settle_ms', [round(d['settle_ms'], 2) for d in L['pitch_step']])
    for r in res[1:]:
        g = r['gate_metrics']
        print(r['total_s'], r['rt60'], r['extra_db'], r['howl'], 'rel40', g['release_to_-40dBFS_s'], 'fa', g['false_trigger_events'],
              'gap>-40', round(g['gap_output_above_-40dBFS_s'], 2), {k: v for k, v in g.items() if 'silen' in k or 'between' in k})
