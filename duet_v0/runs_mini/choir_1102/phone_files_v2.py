"""v2 of phone_files_v1.py (v1 wrote the Ode to Joy file, then found no orig_v11.wav): the v11 phone file
futures/live_test/phone/test_her_v11.mp3 (the real singer, 13 s, no count-in) after 3 s of silence ->
test_her_v11_lead3s.wav / .mp3, new files only.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py phone_files_v2 120 .venv/bin/python runs_mini/choir_1102/phone_files_v2.py
"""
import os
import sys

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phone_files_v1 as P1  # noqa: E402

if __name__ == '__main__':
    x, sr = sf.read(os.path.join(P1.PHONE, 'test_her_v11.mp3'), dtype='float32', always_2d=True)
    assert sr == P1.SR, sr
    P1.write('test_her_v11_lead3s', x.mean(1))
