"""Demos (page-safe: vocadito CC BY 4.0 + simulation). .venv/bin/python futures/d5_live_fx/demo.py

demo/vocadito_<id>_harmony.mp3     her (vocadito) + Duet's live third-below harmony, engine run block by block
demo/room_3min_closed_loop.mp3     3-minute closed-loop room simulation (guard on, RT60 0.6 s, normal volume):
                                   what the mic hears = her + Duet's speaker through the room + room noise
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fx  # noqa: E402
import sim5  # noqa: E402
from dsp_io import write_wav  # noqa: E402

SR = 24000
DEMO = HERE / 'demo'
LAME = '/opt/homebrew/bin/lame'


def mp3(wav, out, kbps=64):
    subprocess.run([LAME, '--silent', '-m', 'm', '-b', str(kbps), str(wav), str(out)], check=True)
    wav.unlink()
    return out.stat().st_size


def main(ids=(37, 2)):
    DEMO.mkdir(exist_ok=True)
    lines = []
    for i in ids:
        x, notes, tonic, tun = sim5.load_excerpt(i)
        x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
        y, _ = fx.run_open(x, tonic, aec=False)
        mix = x[:len(y)] + 0.8 * y
        mix *= 0.89 / np.max(np.abs(mix))
        w = DEMO / f'vocadito_{i}_harmony.wav'
        write_wav(w, mix)
        size = mp3(w, DEMO / f'vocadito_{i}_harmony.mp3')
        names = 'C C# D Eb E F F# G Ab A Bb B'.split()
        lines.append(f'- vocadito_{i}_harmony.mp3 ({size / 1e6:.2f} MB): vocadito track {i} (Bittner et al.), CC BY 4.0, '
                     f'plus Duet harmony computed from it (key {names[tonic]} major chosen before singing).')
    dry, segs = sim5.session(180.0)
    r = sim5.run(dry, segs, rt60=0.6, guard=True, return_signals=True)
    mic = r['mic'][:int(180 * SR)]
    mic *= 0.89 / np.max(np.abs(mic))
    w = DEMO / 'room_3min_closed_loop.wav'
    write_wav(w, mic)
    size = mp3(w, DEMO / 'room_3min_closed_loop.mp3')
    ids_used = ', '.join(str(s['id']) for s in segs)
    lines.append(f'- room_3min_closed_loop.mp3 ({size / 1e6:.2f} MB): simulated room (RT60 0.6 s, mic 0.2 m from her, '
                 f'speaker 1 m from mic, normal volume, guard on), what the mic hears. Her voice: vocadito tracks '
                 f'{ids_used} (Bittner et al.), CC BY 4.0; room and Duet are simulated.')
    (DEMO / 'SOURCES.md').write_text('# Sources of the demo files (all page-safe)\n\n' + '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
