"""v2 of selftest_demo10_v1.py: page v17 now with choir_v8 (song re-identification running). Offline selftest of page v17, the 10-song demo library (run choir_1102). Each synthetic song as its phone file
(phone_files_v5.notes_for: library key and tempo, parted notes, 3 s silence) and two variants a singer might give
(3 semitones down and 5 % slower, 2 up and 6 % faster); v11 as its real phone recording; plus every earlier live
Twinkle / v11 / Ode run. Per run: song recognised (right / wrong / none), seconds from her first note to the
recognition, switches, chord timing (target -35 ms), worst block time. Tripwire (no sound), fixed 174 ms round trip
for the files.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py selftest_demo10_v2 3000 .venv/bin/python runs_mini/choir_1102/selftest_demo10_v2.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selftest_panel_v16 as S16  # noqa: E402  (run(), LIVE, tripwire via choir_timing_eval_v11)
sys.path.insert(0, os.path.join(S16.E.ABS, 'futures/v4_selfiter'))
import live_panel_v17 as P17  # noqa: E402
import phone_files_v5 as PF  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

E = S16.E
OUT = os.path.join(E.ABS, 'runs_mini/choir_1102/selftest_demo10_v2.json')
VARIANTS = [('phone', 0, 1.0), ('-3st x0.95', -3, 0.95), ('+2st x1.06', 2, 1.06)]

if __name__ == '__main__':
    E.P3.MODELS.load()
    lib = P17.P5.CHOIR
    assert len(lib) == 10
    S16.P16.P5.CHOIR = lib                       # S16.run() scores against this library
    res = {}
    for sid in PF.ORDER:
        s = lib[sid]
        if sid == 'v11':
            x = sf.read(os.path.join(E.ABS, 'futures/live_test/phone/test_her_v11_lead3s.wav'), dtype='float32',
                        always_2d=True)[0][:, 0]
            sung = E.true_notes(x)
            res['phone v11'] = r = S16.run(x, sid, 0.174, sung[0][0] + 0.174 - S16.LO if sung else 0.0)
            print('phone v11', json.dumps(r), flush=True)
            continue
        for name, semis, tempo in VARIANTS:
            y = PF.render(s, semis, tempo)
            y = np.concatenate([np.zeros(int(3.0 * E.SR)), y, np.zeros(int(2.0 * E.SR))])
            y = (y * 0.6 / np.abs(y).max()).astype(np.float32)
            first = 3.0 + s['melody'][0][0] * 60.0 / (s['bpm'] * tempo)
            res[f'{name} {sid}'] = r = S16.run(y, sid, 0.174, first)
            print(name, sid, json.dumps(r), flush=True)
    for name, (sid, rt) in S16.LIVE.items():
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', name, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        sung = E.true_notes(x)
        res[name] = r = S16.run(x, sid, rt, sung[0][0] + E.LI if sung else 0.0)
        print(name, json.dumps(r), flush=True)
    for grp in ('phone', '-3st', '+2st', '2'):
        rows = [r for k, r in res.items() if k.startswith(grp)] if grp != '2' else \
            [r for k, r in res.items() if k[0].isdigit()]
        ok = [r for r in rows if r['got'] == r['want']]
        ids = [r['id_after_s'] for r in ok if r['id_after_s'] is not None]
        ch = [r['chord_med_abs_from_target'] for r in ok if r.get('chords')]
        print(f'{grp}: right {len(ok)}/{len(rows)}, wrong {[(r["want"], r["got"]) for r in rows if r not in ok]}, '
              f'id after median {np.median(ids) if ids else None}, chord median {np.median(ch) if ch else None} ms, '
              f'block max {max(r.get("block_ms_max", 0) for r in rows):.2f} ms', flush=True)
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1)
