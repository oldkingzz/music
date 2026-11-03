"""v2 of ode_phone_check_v1.py (v1's rough room did not reproduce run 205400's failure): the same files started the
way 205400 was, play pressed ~3 s before 开始 so the run opens right on her first note (the 3 s silence cut, plus
0 / 20 / 60 ms of the first note), clean and through the rough room. Offline (tripwire).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py ode_phone_check_v2 900 .venv/bin/python runs_mini/choir_1102/ode_phone_check_v2.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ode_phone_check_v1 as C  # noqa: E402

import soundfile as sf  # noqa: E402

E, S = C.E, C.S

if __name__ == '__main__':
    E.P3.MODELS.load()
    for name in ('ode_C_96bpm_lead3s', 'ode_C_96bpm_lead3s_v3'):
        x = sf.read(os.path.join(E.ABS, 'futures/live_test/phone', name + '.wav'), dtype='float32',
                    always_2d=True)[0][:, 0]
        for cut in (0.0, 0.02, 0.06):
            y = x[int((3.0 + cut) * E.SR):]
            for tag, z in (('clean', y), ('room', C.room(y))):
                r = S.run(z, 'ode', 0.174)
                print(name, f'cut {cut:.2f}', tag, json.dumps({k: r.get(k) for k in (
                    'song', 'chords', 'chord_med_abs_from_target', 'chord_p90_abs_from_target')}), flush=True)
