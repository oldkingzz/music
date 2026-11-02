"""iter2 demos: vocadito 37 and 2, her + iter2 engine harmony.  .venv/bin/python futures/d5_live_fx/iter2/demo_v2.py"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import importlib.util
spec = importlib.util.spec_from_file_location('fx2', HERE / 'fx.py')
fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)
import sim5  # noqa
from dsp_io import write_wav  # noqa
V2 = {"drop_db": 16, "onset_db": 26, "f0_tau": 0.015, "mode": "follow"}
fx.DEFAULTS.update(V2)
D = HERE / 'demo'; D.mkdir(exist_ok=True)
lines = []
names = 'C C# D Eb E F F# G Ab A Bb B'.split()
for i in (37, 2):
    x, notes, tonic, tun = sim5.load_excerpt(i)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
    y, _ = fx.run_open(x, tonic, aec=False, fmin=140.0 if i in sim5.FEMALE else 80.0)
    mix = x[:len(y)] + 0.8 * y
    mix *= 0.89 / np.max(np.abs(mix))
    w = D / f'vocadito_{i}_harmony_v2.wav'
    write_wav(w, mix)
    out = D / f'vocadito_{i}_harmony_v2.mp3'
    subprocess.run(['/opt/homebrew/bin/lame', '--silent', '-m', 'm', '-b', '64', str(w), str(out)], check=True)
    w.unlink()
    lines.append(f'- vocadito_{i}_harmony_v2.mp3 ({out.stat().st_size / 1e6:.2f} MB): vocadito track {i} (Bittner et al.), CC BY 4.0, '
                 f'plus Duet harmony from the iter2 engine (key {names[tonic]} major chosen before singing).')
(D / 'SOURCES.md').write_text('# Sources of the iter2 demo files (all page-safe)\n\n' + '\n'.join(lines) + '\n')
print('\n'.join(lines))
