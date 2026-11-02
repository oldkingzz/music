"""Bleed sweep: every (mitigation x bleed level x scenario x echo path) closed-loop run.

    .venv/bin/python echo/sweep.py            # main sweep + stress sweep, ~10 min on 8 cores
    .venv/bin/python echo/sweep.py --quick    # 2 scenarios, 1 echo path (smoke test)

Writes echo/results/sweep.jsonl and echo/results/stress.jsonl (one JSON line per run, a few MB).
Then run echo/analyze.py for the tables and figures.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guard as G  # noqa: E402
import sim  # noqa: E402

LEVELS = [None, -36, -30, -24, -18, -15, -12, -9, -6, -3, 0, 3, 6]
METHODS = {
    'none':     dict(guard=None, soundcheck=None),
    'aec_cold': dict(guard=dict(aec=True, mask=False), soundcheck=None),
    'aec':      dict(guard=dict(aec=True, mask=False), soundcheck=2.0),
    'mask':     dict(guard=dict(aec=False, mask=True), soundcheck=2.0),   # sound check only measures the delay
    'aec+mask': dict(guard=dict(aec=True, mask=True), soundcheck=2.0),
}
SCENS = ['natural', 'expressive', 'pause', 'skip']
PATHS = [dict(acoustic_ms=1.0, seed=0), dict(acoustic_ms=2.0, seed=1), dict(acoustic_ms=3.0, seed=2)]


def make_ir(p):
    return sim.room_ir(acoustic_ms=p['acoustic_ms'], seed=p['seed'])


def perturbed(h, seed, keep=0.8, new=0.6, after_ms=23.5):
    """Same direct path, room tail partly changed (somebody walked in front of the speaker)."""
    h2, _ = sim.room_ir(seed=seed)
    k = int(after_ms / 1000 * sim.SR)
    out = h.copy()
    n = min(len(h), len(h2))
    out[k:n] = keep * h[k:n] + new * h2[k:n]
    return out


def job(task):
    method, level, scen, pi, stress = task
    p = PATHS[pi]
    h, info = make_ir(p)
    cfg = METHODS[method]
    guard = G.EchoGuard(**cfg['guard']) if cfg['guard'] else None
    kw = {}
    if stress == 'thd3':
        kw['thd'] = 0.03
    elif stress == 'pathchange':
        kw['ir_after'] = perturbed(h, seed=100 + pi)
    t0 = time.perf_counter()
    m, _ = sim.run(CTX, scen, level, guard=guard, ir=h, soundcheck=cfg['soundcheck'], **kw)
    m.update({'method': method, 'level': level, 'path': pi, 'path_info': info, 'stress': stress,
              'wall_s': time.perf_counter() - t0})
    return m


CTX = None


def main():
    global CTX
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--only', choices=['main', 'stress'])
    args = ap.parse_args()
    CTX = sim.synthetic_context()            # built once, inherited by the forked workers
    out_dir = HERE / 'results'
    out_dir.mkdir(exist_ok=True)
    scens = SCENS[:2] if args.quick else SCENS
    paths = [1] if args.quick else range(len(PATHS))
    sets = {
        'main': [(m, L, s, pi, None) for m in METHODS for L in LEVELS for s in scens for pi in paths],
        'stress': [(m, L, s, 1, st) for st in ('thd3', 'pathchange') for m in ('none', 'aec', 'mask', 'aec+mask')
                   for L in LEVELS for s in ('natural', 'pause')],
    }
    ctx = mp.get_context('fork')
    for name, tasks in sets.items():
        if args.only and name != args.only:
            continue
        if args.quick and name == 'stress':
            continue
        fn = out_dir / (f'{name}_quick.jsonl' if args.quick else f'{name}.jsonl')
        t0 = time.time()
        with ctx.Pool(args.procs) as pool, open(fn, 'w') as f:
            for i, m in enumerate(pool.imap_unordered(job, tasks, chunksize=1)):
                f.write(json.dumps(m) + '\n')
                f.flush()
                if i % 25 == 0:
                    print(f'{name}: {i + 1}/{len(tasks)}  {time.time() - t0:.0f} s', flush=True)
        print(f'{name}: {len(tasks)} runs in {time.time() - t0:.0f} s -> {fn}')


if __name__ == '__main__':
    main()
