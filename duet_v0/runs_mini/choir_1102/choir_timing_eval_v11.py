"""v11 of the choir timing eval (run choir_1102): one voice per part (choir_v6.LEGATO) and how far ahead part notes are
written (AHEAD: the beat map is read then; 0.3 s in choir_v5). Adds, per run, how often a part sings two notes at once
(the next note starts before the previous one ends) and how often it falls silent for > 80 ms between two notes that
are contiguous in the score; chord times are the notes' actual start samples. Same runs and truth as v10.

  cd duet_v0 && .venv/bin/python runs_mini/v4_1101/tools/hbrun.py choir_teval11 900 .venv/bin/python runs_mini/choir_1102/choir_timing_eval_v11.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choir_timing_eval_v10 as E  # noqa: E402  (tripwire, page v13, truth, replay)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

OUT = os.path.join(E.ABS, 'runs_mini/choir_1102/choir_timing_eval_v11.json')
NOTES = []
_init = E.CV6.ChoirStart.__init__


def _init_notes(self, *a, **k):
    _init(self, *a, **k)
    sch = self.choir.schedule

    def schedule(start, dur, midi, tag=None):
        i0 = sch(start, dur, midi, tag=tag)
        NOTES.append((tag[1] if tag else None, self.song_id, self.choir.by_tag.get(tag)))
        return i0
    self.choir.schedule = schedule


E.CV6.ChoirStart.__init__ = _init_notes


def voice_stats(eng):
    """Per part, consecutive notes contiguous in the score: overlap (both sounding, release excluded) and holes."""
    by_part = {}
    for idx, song, nt in NOTES:
        if song != eng.song_id or idx is None or nt is None:
            continue
        on, off, m, p = eng.parts[idx]
        by_part.setdefault(p, []).append((on, off, nt.i0 / E.SR, nt.i0 / E.SR + nt.dur, round(nt.midi)))
    over, holes, clash, n = [], [], 0.0, 0
    for seq in by_part.values():
        seq.sort()
        for (on0, off0, s0, e0, m0), (on1, off1, s1, e1, m1) in zip(seq, seq[1:]):
            if abs(off0 - on1) > 1e-6:
                continue
            n += 1
            if e0 - s1 > 0.0:
                over.append(e0 - s1)
                clash += (e0 - s1) if m0 != m1 else 0.0
            if s1 - e0 > 0.08:
                holes.append(s1 - e0)
    mx = lambda a: round(float(max(a)) * 1000) if a else 0  # noqa: E731
    return {'steps': n, 'two_at_once': len(over), 'two_at_once_max_ms': mx(over), 'clash_s': round(clash, 2),
            'holes': len(holes), 'hole_max_ms': mx(holes)}


VARIANTS = {
    'F now (no legato, ahead 0.3)': dict(LEGATO=False, AHEAD=0.3),
    'L legato, ahead 0.3': dict(LEGATO=True, AHEAD=0.3),
    'M legato, ahead 0.1': dict(LEGATO=True, AHEAD=0.1),
    'N legato, ahead 0.05': dict(LEGATO=True, AHEAD=0.05),
}

if __name__ == '__main__':
    E.P3.MODELS.load()
    lib = E.P13.P12.P5.CHOIR
    res = {}
    for run_name, sid in E.RUNS.items():
        x = sf.read(os.path.join(E.ABS, 'runs_mini/live', run_name, 'run.wav'), dtype='float32', always_2d=True)[0][:, 0]
        sung = E.true_notes(x)
        print(run_name, sid, flush=True)
        for vname, st in VARIANTS.items():
            NOTES.clear()
            eng, _ = E.replay(x, st)
            score = [(float(a), int(m) + (eng.S or 0)) for a, b, m in lib[sid]['melody']]
            truth = E.match(score, sung, lib[sid]['bpm'] if sid == 'v11' else 60.0 / eng.Tb)
            log = [(nt.i0, idx, song) for idx, song, nt in NOTES if nt is not None]
            r = E.score_run(eng, log, truth)
            pe = np.abs(eng.diag['pred_err_ms'])
            r.update({'pred_med': round(float(np.median(pe))) if len(pe) else None,
                      'pred_p90': round(float(np.percentile(pe, 90))) if len(pe) else None})
            r.update(voice_stats(eng))
            res[f'{run_name} | {vname}'] = r
            keep = ('chord_med_abs_from_target', 'chord_p90_abs_from_target', 'pred_med', 'pred_p90', 'two_at_once',
                    'two_at_once_max_ms', 'clash_s', 'holes', 'hole_max_ms')
            print(f'   {vname:30s}', json.dumps({k: r[k] for k in keep}), flush=True)
            print('      per chord (beat, ms):', r['per_chord'], flush=True)
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1)
