"""Run a command with a wall-clock limit (macOS has no `timeout`):  python tl.py SECONDS cmd args..."""
import os, signal, subprocess, sys
sec, cmd = float(sys.argv[1]), sys.argv[2:]
p = subprocess.Popen(cmd, start_new_session=True)
try:
    sys.exit(p.wait(timeout=sec))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGKILL)
    print(f'TIMEOUT after {sec:.0f} s: {" ".join(cmd)}', file=sys.stderr)
    sys.exit(124)
