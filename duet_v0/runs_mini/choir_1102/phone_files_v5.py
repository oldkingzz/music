"""v5 of phone_files_v3.py: the phone files of the 10-song demo library (choir_songs_v9.json; Vincent 2026-11-02:
"把10首的录音全部发我 ... 这个就是明天我拿过去的demo"). Each song sung by the synthetic 'oo' voice of dsp.render_voice at
the library key and tempo, after 3 s of silence; every note ends GAP s early (at most GAP_FRAC of the note, so quick
eighths keep a sound), the parted "la la" that let the live tracker hear repeated notes (v3: 150 ms worked on the
Ode). v11 is the real singer (test_her_v11_lead3s.wav, already 3 s lead), copied unchanged. Writes
futures/live_test/phone/demo10_v1/<nn>_<id>.wav / .mp3, new files only.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py phone_files_v5 300 .venv/bin/python runs_mini/choir_1102/phone_files_v5.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phone_files_v1 as P1  # noqa: E402

LIB = os.path.join(P1.ABS, 'futures/v4_selfiter/choir_songs_v9.json')
OUT = os.path.join(P1.PHONE, 'demo10_v1')
ORDER = ['twinkle', 'ode', 'frere', 'mary', 'jingle', 'saints', 'macdonald', 'auld', 'goinghome', 'v11']
GAP, GAP_FRAC = 0.12, 0.35


def notes_for(s, semis=0, tempo=1.0):
    """(onset s, duration s, midi) of song s, transposed by semis and played tempo times as fast."""
    sb = 60.0 / (s['bpm'] * tempo)
    return [(a * sb, (b - a) * sb - min(GAP, GAP_FRAC * (b - a) * sb), m + semis) for a, b, m in s['melody']]


def render(s, semis=0, tempo=1.0):
    return P1.render_voice(notes_for(s, semis, tempo), int((s['K'] * 60.0 / (s['bpm'] * tempo) + 0.5) * P1.SR))


if __name__ == '__main__':
    with open(LIB) as fh:
        songs = json.load(fh)['songs']
    assert sorted(ORDER) == sorted(songs)
    os.makedirs(OUT)
    P1.PHONE = OUT
    for n, sid in enumerate(ORDER, 1):
        name = f'{n:02d}_{sid}'
        if sid == 'v11':
            for ext in ('wav', 'mp3'):
                shutil.copyfile(os.path.join(P1.ABS, 'futures/live_test/phone/test_her_v11_lead3s.' + ext),
                                os.path.join(OUT, f'{name}.{ext}'))
            print(name, 'copied (real singer)', flush=True)
        else:
            P1.write(name, render(songs[sid]))
