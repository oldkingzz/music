"""v3 of phone_files_v1.py: the Ode to Joy phone file with clearly parted notes. v1's 30 ms breath between notes
vanished through phone speaker -> AirPods mic (run 205400: the live tracker heard 1 of 14 repeated notes and 16 of
48 others); here each note ends 150 ms early (about 90 ms of silence before the next), like a singer's "la la".
Writes futures/live_test/phone/ode_C_96bpm_lead3s_v3.wav / .mp3, new files only.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py phone_files_v3 120 .venv/bin/python runs_mini/choir_1102/phone_files_v3.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phone_files_v1 as P1  # noqa: E402

if __name__ == '__main__':
    with open(os.path.join(P1.ABS, 'futures/v4_selfiter/choir_songs_v7.json')) as fh:
        ode = json.load(fh)['songs']['ode']
    sb = 60.0 / ode['bpm']
    notes = [(a * sb, (b - a) * sb - 0.15, m) for a, b, m in ode['melody']]
    P1.write('ode_C_96bpm_lead3s_v3', P1.render_voice(notes, int((ode['K'] * sb + 0.5) * P1.SR)))
