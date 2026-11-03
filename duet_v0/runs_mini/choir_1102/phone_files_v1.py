"""Phone files for the choir rehearsal (run choir_1102), each starting after 3 s of silence so there is time to hold
the phone to the mic after pressing play (Vincent 2026-11-02: "不用启动的那么着急"):
- ode_C_96bpm_lead3s: Ode to Joy (choir_songs_v7.json 'ode', C major, 96 bpm, 64 beats) sung by the synthetic 'oo'
  voice of dsp.render_voice (the Twinkle phone file's kind of singer), about 40 s of singing;
- test_her_v11_lead3s: the real singer of v11 (futures/live_test/orig_v11.wav, 13 s), unchanged after the silence.
Writes futures/live_test/phone/<name>.wav (24 kHz) and <name>.mp3 (lame 192 kbps), new files only.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py phone_files_v1 300 .venv/bin/python runs_mini/choir_1102/phone_files_v1.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, ABS)
from dsp import SR, render_voice  # noqa: E402

PHONE = os.path.join(ABS, 'futures/live_test/phone')
LEAD, TAIL, PEAK = 3.0, 2.0, 0.6


def write(name, y):
    y = np.concatenate([np.zeros(int(LEAD * SR)), y, np.zeros(int(TAIL * SR))])
    y = (y * PEAK / np.abs(y).max()).astype(np.float32)
    wav, mp3 = os.path.join(PHONE, name + '.wav'), os.path.join(PHONE, name + '.mp3')
    assert not os.path.exists(wav) and not os.path.exists(mp3)
    sf.write(wav, y, SR, subtype='PCM_16')
    subprocess.run(['lame', '--quiet', '-b', '192', wav, mp3], check=True)
    print(name, '%.1f s' % (len(y) / SR), mp3, flush=True)


if __name__ == '__main__':
    with open(os.path.join(ABS, 'futures/v4_selfiter/choir_songs_v7.json')) as fh:
        ode = json.load(fh)['songs']['ode']
    sb = 60.0 / ode['bpm']
    notes = [(a * sb, (b - a) * sb - 0.03, m) for a, b, m in ode['melody']]   # a short breath between notes
    write('ode_C_96bpm_lead3s', render_voice(notes, int((ode['K'] * sb + 0.5) * SR)))
    x, sr = sf.read(os.path.join(ABS, 'futures/live_test/orig_v11.wav'), dtype='float32', always_2d=True)
    assert sr == SR, sr
    write('test_her_v11_lead3s', x.mean(1))
