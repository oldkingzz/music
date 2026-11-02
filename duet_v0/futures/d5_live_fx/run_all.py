"""Regenerate every D5 result: .venv/bin/python futures/d5_live_fx/run_all.py  (about 12 min on an M4, 3 processes)"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
ECORE = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')


def run(*cmd, env=None):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, env=env)


if __name__ == '__main__':
    run(PY, HERE / 'test_fx.py')
    run(PY, HERE / 'latency.py')
    run('taskpolicy', '-c', 'background', PY, HERE / 'latency.py', '--timing-only', '--tag', 'ecore', env=ECORE)
    run(PY, HERE / 'clash.py')
    run(PY, HERE / 'sim5.py', '--procs', '3')
    run(PY, HERE / 'demo.py')
    run(PY, HERE / 'make_results.py')
