"""Vincent's own a cappella Twinkle runs beat by beat (run choir_1102): 095417 (rated 一般) and 110255 (rated 差) are
the only runs of his own voice, and the offline truth match fails on them, so this lists the raw material instead:
his sung notes (offline pyin, minus the input latency), every note the live follower heard (accepted as which beat,
prediction before it, bpm after; or not accepted, and why it looks so), the follower's song / key / start, and when
each chord change sounded. Today's engine (choir_v6 defaults), offline (tripwire).

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py own_trace_v1 600 .venv/bin/python runs_mini/choir_1102/own_trace_v1.py [RUN ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v11 as V  # noqa: E402  (tripwire, page v13, truth, replay, NOTES)

import soundfile as sf  # noqa: E402

E = V.E
RUNS = sys.argv[1:] or ['110255_r2_mix_v2', '095417_r2_mix_v2']
_acc = E.CV6.ChoirStart._accept
ACC = {}


def acc(self, note, c):
    b = self.mel[c][0]
    pred = self.t_of(b)
    _acc(self, note, c)
    ACC[id(note)] = (b, pred, 60 / self.Tb, self.expect(self.mel[c][1]))


E.CV6.ChoirStart._accept = acc

if __name__ == '__main__':
    E.P3.MODELS.load()
    for run in RUNS:
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        sung = E.true_notes(x)
        ACC.clear()
        V.NOTES.clear()
        eng, _ = E.replay(x, {})
        d = {k: v for k, v in eng.diag.items() if not isinstance(v, list)}
        print(f'######## {run}: song {eng.song_id}, shift {eng.S}, final bpm {60 / eng.Tb:.1f}, length {len(x) / E.SR:.1f} s')
        print('diag', d)
        print('sung (offline): onset s, length s, midi')
        print('   ' + '  '.join(f'{a:.2f}/{e - a:.2f}/{p:.1f}' for a, e, p in sung))
        print(f"{'heard t':>8} {'pitch':>6} | {'beat':>5} {'expect':>6} {'live-pred ms':>12} {'bpm after':>9} | nearest sung onset (ms off)")
        for n in eng.hnotes:
            near = min(sung, key=lambda s: abs(s[0] - n['t'])) if sung else (0, 0, 0)
            tail = f"{near[0]:7.2f} ({(n['t'] - near[0]) * 1000:+5.0f}) {near[2]:5.1f}"
            a = ACC.get(id(n))
            if a:
                b, pred, bpm, w = a
                print(f"{n['t']:8.2f} {n['pitch']:6.1f} | {b:5.1f} {w:6.1f} {(n['t'] - pred) * 1000:12.0f} {bpm:9.1f} | {tail}")
            else:
                why = 'junk' if n.get('junk') else '-'
                print(f"{n['t']:8.2f} {n['pitch']:6.1f} | {'':5} {'':6} {why:>12} {'':9} | {tail}")
        seen = set()
        print('chord changes: beat -> sounds at (s)')
        row = []
        for idx, song, nt in V.NOTES:
            if nt is None or song != eng.song_id or idx is None or idx >= len(eng.parts) or nt.done == 0:
                continue
            on = eng.parts[idx][0]
            if on not in seen:
                seen.add(on)
                row.append(f'{on:g}:{nt.i0 / E.SR + eng.Lo:.2f}')
        print('   ' + '  '.join(row))
