"""v3 of selftest_demo10_v1 (v2: choir_v8): page v18, whose choir waits until it is sure of the song (choir_v9).
Also per run: seconds from her first note until the choir sounds (wait_s) and choir notes sung for a wrong song.

Offline selftest of page v17, the 10-song demo library (run choir_1102). Each synthetic song as its phone file
(phone_files_v5.notes_for: library key and tempo, parted notes, 3 s silence) and two variants a singer might give
(3 semitones down and 5 % slower, 2 up and 6 % faster); v11 as its real phone recording; plus every earlier live
Twinkle / v11 / Ode run. Per run: song recognised (right / wrong / none), seconds from her first note to the
recognition, switches, chord timing (target -35 ms), worst block time. Tripwire (no sound), fixed 174 ms round trip
for the files.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py selftest_demo10_v3 3000 .venv/bin/python runs_mini/choir_1102/selftest_demo10_v3.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selftest_panel_v16 as S16  # noqa: E402  (run(), LIVE, tripwire via choir_timing_eval_v11)
sys.path.insert(0, os.path.join(S16.E.ABS, 'futures/v4_selfiter'))
import live_panel_v18 as P17  # noqa: E402
import phone_files_v5 as PF  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

E = S16.E
V = S16.V
_replay, CAP = E.replay, {}


def _cap(*a, **k):
    out = E_replay(*a, **k)
    CAP['eng'] = out[0]
    return out


E_replay = _replay
E.replay = _cap


def run(x, sid, rt, first):
    r = S16.run(x, sid, rt, first)
    eng = CAP['eng']
    sf_ = eng.diag.get('sing_from')
    r['wait_s'] = None if sf_ is None else round(sf_ - first, 2)
    r['wrong_song_notes'] = sum(1 for idx, song, nt in V.NOTES if nt is not None and song != sid)
    r['sung_notes'] = sum(1 for idx, song, nt in V.NOTES if nt is not None)
    return r

OUT = os.path.join(E.ABS, 'runs_mini/choir_1102/selftest_demo10_v3.json')
VARIANTS = [('phone', 0, 1.0), ('-3st x0.95', -3, 0.95), ('+2st x1.06', 2, 1.06)]

if __name__ == '__main__':
    E.P3.MODELS.load()
    lib = P17.P5.CHOIR
    assert len(lib) == 10
    S16.P16.P5.CHOIR = lib                       # run() scores against this library
    res = {}
    for sid in PF.ORDER:
        s = lib[sid]
        if sid == 'v11':
            x = sf.read(os.path.join(E.ABS, 'futures/live_test/phone/test_her_v11_lead3s.wav'), dtype='float32',
                        always_2d=True)[0][:, 0]
            sung = E.true_notes(x)
            res['phone v11'] = r = run(x, sid, 0.174, sung[0][0] + 0.174 - S16.LO if sung else 0.0)
            print('phone v11', json.dumps(r), flush=True)
            continue
        for name, semis, tempo in VARIANTS:
            y = PF.render(s, semis, tempo)
            y = np.concatenate([np.zeros(int(3.0 * E.SR)), y, np.zeros(int(2.0 * E.SR))])
            y = (y * 0.6 / np.abs(y).max()).astype(np.float32)
            first = 3.0 + s['melody'][0][0] * 60.0 / (s['bpm'] * tempo)
            res[f'{name} {sid}'] = r = run(y, sid, 0.174, first)
            print(name, sid, json.dumps(r), flush=True)
    for name, (sid, rt) in S16.LIVE.items():
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', name, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        sung = E.true_notes(x)
        res[name] = r = run(x, sid, rt, sung[0][0] + E.LI if sung else 0.0)
        print(name, json.dumps(r), flush=True)
    for grp in ('phone', '-3st', '+2st', '2'):
        rows = [r for k, r in res.items() if k.startswith(grp)] if grp != '2' else \
            [r for k, r in res.items() if k[0].isdigit()]
        ok = [r for r in rows if r['got'] == r['want']]
        ids = [r['id_after_s'] for r in ok if r['id_after_s'] is not None]
        ch = [r['chord_med_abs_from_target'] for r in ok if r.get('chords')]
        print(f'{grp}: right {len(ok)}/{len(rows)}, wrong {[(r["want"], r["got"]) for r in rows if r not in ok]}, '
              f'id after median {np.median(ids) if ids else None}, chord median {np.median(ch) if ch else None} ms, '
              f'block max {max(r.get("block_ms_max", 0) for r in rows):.2f} ms, '
              f'wait median {np.median([r["wait_s"] for r in ok if r["wait_s"] is not None]) if ok else None} s, '
              f'wrong-song notes {sum(r["wrong_song_notes"] for r in rows)}', flush=True)
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1)
