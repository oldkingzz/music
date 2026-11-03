"""Offline selftest of page v16 (run choir_1102), the 100-song library: 20 songs (Twinkle, v11, Ode and 17 POP909
picked evenly) sung by the synthetic voice from their MIDI melody (3 s silence, notes parted by 60 ms, 3 semitones
down, tempo x0.95), plus every earlier live Twinkle / v11 / Ode run. Per run: song recognised (right / wrong / none),
seconds from her first note to the recognition, chord timing (target -35 ms), and the engine's worst block time
(it must stay well under the 10 ms block). Tripwire (no sound), fixed 174 ms round trip for synthetic runs.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py selftest_v16 3000 .venv/bin/python runs_mini/choir_1102/selftest_panel_v16.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402  (tripwire, replay, NOTES)
import live_panel_v16 as P16  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from dsp import render_voice  # noqa: E402

E = V.E
LO = 0.0421
OUT = os.path.join(E.ABS, 'runs_mini/choir_1102/selftest_panel_v16.json')
LIVE = {'211440_r2_mix_v2': ('ode', 0.174), '211515_r2_mix_v2': ('v11', 0.174), '203926_r2_mix_v2': ('v11', 0.174),
        '195940_r2_mix_v2': ('v11', 0.174), '132458_r2_mix_v2': ('v11', 0.1305), '124132_r2_mix_v2': ('v11', 0.1305),
        '195914_r2_mix_v2': ('twinkle', 0.174), '113217_r2_mix_v2': ('twinkle', 0.1305),
        '114844_r2_mix_v2': ('twinkle', 0.1305)}


def run(x, sid, rt, first_s):
    E.LI, E.LO = rt - LO, LO
    V.NOTES.clear()
    t0 = time.time()
    eng, _ = E.replay(x, {})
    r = {'want': sid, 'got': eng.song_id, 'id_after_s': None if eng.diag.get('id_time') is None else
         round(eng.diag['id_time'] - first_s, 2), 'wall_s': round(time.time() - t0, 1)}
    js = os.path.join(sorted((os.path.join(E.ROOT, d) for d in os.listdir(E.ROOT)
                              if os.path.isdir(os.path.join(E.ROOT, d))), key=os.path.getmtime)[-1], 'notes.json')
    if os.path.exists(js):
        with open(js) as fh:
            r['block_ms_max'] = round(json.load(fh).get('block_ms_max') or 0, 2)
    if eng.song_id == sid:
        lib = P16.P5.CHOIR
        score = [(float(a), int(m) + (eng.S or 0)) for a, b, m in lib[sid]['melody']]
        truth = E.match(score, E.true_notes(x), lib[sid]['bpm'] if sid == 'v11' else 60.0 / eng.Tb)
        log = [(nt.i0, idx, song) for idx, song, nt in V.NOTES if nt is not None]
        s = E.score_run(eng, log, truth)
        r.update({k: s.get(k) for k in ('chords', 'chord_med_abs_from_target', 'chord_p90_abs_from_target')})
    return r


if __name__ == '__main__':
    E.P3.MODELS.load()
    lib = P16.P5.CHOIR
    assert len(lib) == 100
    pop = sorted(k for k in lib if k.startswith('pop909'))
    picks = ['twinkle', 'v11', 'ode'] + [pop[i] for i in np.linspace(0, len(pop) - 1, 17).astype(int)]
    res = {}
    for sid in picks:
        s = lib[sid]
        sb = 60.0 / (s['bpm'] * 0.95)
        notes = [(3.0 + a * sb, max(0.08, (b - a) * sb - 0.06), m - 3) for a, b, m in s['melody']]
        y = render_voice(notes, int((5.0 + s['K'] * sb) * E.SR))
        y = (y * 0.6 / np.abs(y).max()).astype(np.float32)
        res['synth ' + sid] = r = run(y, sid, 0.174, 3.0 + s['melody'][0][0] * sb)
        print('synth', sid, json.dumps(r), flush=True)
    for name, (sid, rt) in LIVE.items():
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', name, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        sung = E.true_notes(x)
        res[name] = r = run(x, sid, rt, sung[0][0] + E.LI if sung else 0.0)
        print(name, json.dumps(r), flush=True)
    for grp in ('synth', '2'):
        rows = [r for k, r in res.items() if k.startswith(grp) or (grp == '2' and not k.startswith('synth'))]
        ok = [r for r in rows if r['got'] == r['want']]
        print(f'{grp}: right {len(ok)}/{len(rows)}, wrong {sum(1 for r in rows if r["got"] not in (None, r["want"]))},'
              f' id after median {np.median([r["id_after_s"] for r in ok if r["id_after_s"] is not None]):.2f} s, '
              f'chord median {np.median([r["chord_med_abs_from_target"] for r in ok if r.get("chords")]):.0f} ms, '
              f'block max {max(r.get("block_ms_max", 0) for r in rows):.2f} ms', flush=True)
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1)
