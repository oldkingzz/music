"""Heartbeat wrapper (v4_selfiter §4.1): append a START line to runs_mini/v4_1101/hb/<label>.log, run the command
under futures/v3_nolatency/tl.py <seconds>, append an END line with its exit code, exit with that code.

  cd $ABS/... && $ABS/.venv/bin/python $ABS/runs_mini/v4_1101/tools/hbrun.py <label> <seconds> <cmd> [args...]
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNDIR = os.path.dirname(HERE)
ABS = os.path.dirname(os.path.dirname(RUNDIR))
TL = os.path.join(ABS, 'futures', 'v3_nolatency', 'tl.py')


def hb(label, msg):
    os.makedirs(os.path.join(RUNDIR, 'hb'), exist_ok=True)
    with open(os.path.join(RUNDIR, 'hb', f'{label}.log'), 'a') as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


if __name__ == '__main__':
    label, sec, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', label) or not cmd:
        sys.exit('usage: hbrun.py <label [A-Za-z0-9_.-]> <seconds> <cmd> [args...]')
    float(sec)
    short = ' '.join(a.replace(ABS, '$ABS') for a in cmd)[:400]
    hb(label, f'START tl={sec} cwd={os.getcwd().replace(ABS, "$ABS")} {short}')
    t0 = time.time()
    rc = subprocess.call([sys.executable, TL, sec] + cmd)
    hb(label, f'END rc={rc} secs={time.time() - t0:.0f} {short}')
    sys.exit(rc)
