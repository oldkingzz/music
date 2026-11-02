"""v2 of choir_follow_test_v1.py (same cases; v1 ran against a choir_v5 whose attribute names clashed with the base
engine, so it heard no notes). Offline test of page v9's choir score follower (choir_v5.ChoirFollow), no sound device (tripwire first).

Synthetic singers (the repo's singer.render: vibrato, glides, +-0.1 st intonation) fed through the page Session with a
simulated microphone (true input delay, the engine told the calibrated round trip), and Vincent's own recordings.
Per case: the song / key / tuning / register it settles on, when Duet comes in, and the timing of every part chord
against her true note on the same beat (sound time - her onset; the target is -LEAD = -35 ms), plus the share of her
singing time where a part is harsh against the pitch she actually sang.

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/choir_follow_test_v1.py
"""
import sys
import types

ABS = '/Users/san/claude_sandbox/music/duet_v0'


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('offline test: no sound device (sounddevice.' + name + ')')


for _m in ('sounddevice', '_sounddevice'):
    sys.modules[_m] = _NoSound(_m)

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import live_panel_v9 as P9  # noqa: E402
import choir_v5 as CV5  # noqa: E402
P3, P5 = P9.P3, P9.P5
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import singer  # noqa: E402
from dsp import SR  # noqa: E402

ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_choir', time.strftime('%H%M%S') + '_v9b')
OUT = os.path.join(ABS, 'runs_mini/choir_1102/choir_follow_test_v2.json')
LOG = []
_init = CV5.ChoirFollow.__init__


def _init_log(self, *a, **k):
    _init(self, *a, **k)
    sch = self.choir.schedule

    def schedule(start, dur, midi, tag=None):
        LOG.append((start, tag[1] if tag else None, float(midi), dur))
        return sch(start, dur, midi, tag=tag)
    self.choir.schedule = schedule


CV5.ChoirFollow.__init__ = _init_log
OK, RES = [], {}


def check(what, ok, detail=''):
    OK.append(bool(ok))
    print(('OK   ' if ok else 'FAIL ') + what, detail, flush=True)


def run(x, li, lo, song='auto', bpm=80.0):
    LOG.clear()
    s = P3.Session(ROOT)
    x = np.concatenate([np.zeros(int(li * SR), np.float32), np.asarray(x, np.float32)])
    s.cals[('m', 'p')] = {'in': 'm', 'out': 'p', 'rt_s': li + lo, 'rt_ms': (li + lo) * 1000, 'n_used': 8,
                          'spread_ms': 5, 'when': 't', 'stream_Li_ms': 60, 'stream_Lo_ms': lo * 1000}
    s.start({'song': song, 'bpm': bpm, 'mode': 'choir', 'click': False, 'input': 'mic'},
            source=x, sim={'devs': ('m', 'p'), 'latency': (0.06, lo)})
    t1 = time.time()
    while s.state != 'await' and time.time() - t1 < 240:
        time.sleep(0.05)
    eng = s.eng
    s.verdict({'run': s.status()['run'], 'verdict': '作废', 'note': 'choir v9 follow test'})
    time.sleep(1.1)
    return eng, list(LOG)


def synth(song, key_shift, octave, bpm_her, start_beat, compress=1.0, tune=0.0, slow_to=None, junk=0, page_bpm=80.0,
          li=0.15, lo=0.01, sel='auto'):
    arr = P5.CHOIR[song]
    T0 = 0.5 + 4 * 60.0 / page_bpm + start_beat * 60.0 / page_bpm
    mel = arr['melody']
    do = mel[0][2]
    K = arr['K']

    def t_at(b):                                         # her clock: tempo may slide from bpm_her to slow_to
        if slow_to is None:
            return T0 + b * 60.0 / bpm_her
        r0, r1 = 60.0 / bpm_her, 60.0 / slow_to
        return T0 + r0 * b + (r1 - r0) * b * b / (2 * K)
    notes = [(t_at(a), t_at(b), do + key_shift + 12 * octave + tune + compress * (m - do)) for a, b, m in mel]
    pre = []
    for k in range(junk):                                 # click-like junk before her start (phone count-in)
        pre.append((T0 - 2.0 + 0.6 * k, T0 - 2.0 + 0.6 * k + 0.25, 83.0))
    x = singer.render(pre + notes)[0] if pre else singer.render(notes)[0]
    x = np.concatenate([x, np.zeros(2 * SR, np.float32)])
    eng, log = run(x, li, lo, sel, page_bpm)
    her_on = {}
    for (a, b, m), n in zip(mel, notes):
        her_on.setdefault(float(a), (n[0], n[2]))
    tm = [(start / SR + eng.Lo) - her_on[eng.parts[idx][0]][0] for start, idx, mm, dur in log
          if eng.parts and eng.parts[idx][0] in her_on and her_on[eng.parts[idx][0]][0] > (eng.diag.get('id_time') or 0) + 0.3]
    # harmony: share of her sung time with a part harsh against her actual pitch (rounded)
    harsh = tot = 0.0
    for start, idx, mm, dur in log:
        a0, a1 = start / SR + eng.Lo, start / SR + eng.Lo + dur
        for (on, off, pm) in notes:
            ov = max(0.0, min(off, a1) - max(on, a0))
            if ov:
                tot += ov
                if int(round(abs(mm - pm))) % 12 in (1, 2, 6, 10, 11):
                    harsh += ov
    first_in = min((st for st, *_ in log), default=None)
    sh = (key_shift % 12)
    return eng, {'song': eng.song_id, 'shift': eng.S, 'expect_shift': sh if sh <= 6 else sh - 12, 'tuning': round(eng.u, 2),
                 'low': eng.low, 'tempo': round(60 / eng.Tb, 1), 'in_at_beat': None if first_in is None else
                 round((first_in / SR + eng.Lo - T0) * bpm_her / 60.0, 2),
                 'timing_median_ms': round(float(np.median(tm)) * 1000) if tm else None,
                 'timing_p90_abs_err_ms': round(float(np.percentile(np.abs(np.array(tm) * 1000 + 35), 90))) if tm else None,
                 'n_chords': len(tm), 'harsh_share': round(harsh / tot, 3) if tot else None,
                 'matched': len(eng.pairs), 'relocations': eng.diag['relocations']}


P3.MODELS.load()
cases = [
    ('a twinkle C, page tempo', dict(song='twinkle', key_shift=0, octave=0, bpm_her=80.0, start_beat=0)),
    ('b twinkle A2 like Vincent: narrow intervals, 30 c flat, 88 bpm, 1 beat late',
     dict(song='twinkle', key_shift=-3, octave=-1, bpm_her=88.0, start_beat=1, compress=0.85, tune=-0.3)),
    ('c v11 in G an octave low, 90 bpm vs page 100', dict(song='v11', key_shift=-2, octave=-1, bpm_her=90.0,
                                                          start_beat=0, page_bpm=100.0)),
    ('d twinkle D with 3 click-like junk notes first', dict(song='twinkle', key_shift=2, octave=0, bpm_her=80.0,
                                                            start_beat=2, junk=3)),
    ('e twinkle F slowing from 84 to 68 bpm', dict(song='twinkle', key_shift=5, octave=-1, bpm_her=84.0, start_beat=0,
                                                   slow_to=68.0)),
    ('f twinkle E, AirPods both ways (Li 150, Lo 180)', dict(song='twinkle', key_shift=4, octave=-1, bpm_her=80.0,
                                                             start_beat=0, lo=0.18)),
]
for name, kw in cases:
    eng, r = synth(**kw)
    RES[name] = r
    print(name, json.dumps(r, ensure_ascii=False), flush=True)
    check(name + ': right song and key', r['song'] == kw['song'] and r['shift'] == r['expect_shift'])
    check(name + ': chords on her beat (median within 40 ms of the -35 ms target)',
          r['timing_median_ms'] is not None and abs(r['timing_median_ms'] + 35) <= 40, r['timing_median_ms'])
    check(name + ': harmony fits what she sings (harsh <= 8 %)', r['harsh_share'] is not None and r['harsh_share'] <= 0.08,
          r['harsh_share'])
for run_name, li in (('095417_r2_mix_v2', 0.17), ('094919_r2_mix_v2', 0.17), ('095341_r2_mix_v2', 0.17)):
    w, sr = sf.read(os.path.join(ABS, 'runs_mini/live', run_name, 'run.wav'), dtype='float32', always_2d=True)
    eng, log = run(w[:, 0], 0.0, 0.0421)
    pe = eng.diag['pred_err_ms']
    r = {'song': eng.song_id, 'shift': eng.S, 'tuning': round(eng.u, 2), 'low': eng.low, 'tempo': round(60 / eng.Tb, 1),
         'id_time_s': eng.diag.get('id_time'), 'matched': len(eng.pairs), 'relocations': eng.diag['relocations'],
         'pred_err_median_abs_ms': round(float(np.median(np.abs(pe)))) if pe else None, 'pred_err_ms': pe[:30],
         'her_first_notes': [(round(n['t'], 2), round(n['pitch'], 2)) for n in eng.notes[:12]]}
    RES['replay ' + run_name] = r
    print('replay', run_name, json.dumps(r, ensure_ascii=False), flush=True)
check('replay 09:54 (his a cappella): Twinkle in A, low singer',
      RES['replay 095417_r2_mix_v2']['song'] == 'twinkle' and RES['replay 095417_r2_mix_v2']['shift'] == -3
      and RES['replay 095417_r2_mix_v2']['low'])
check('replay 09:49 (phone, count-in clicks first): Twinkle in C', RES['replay 094919_r2_mix_v2']['song'] == 'twinkle'
      and RES['replay 094919_r2_mix_v2']['shift'] == 0)
print('ALL PASS' if all(OK) else 'SOME FAILED', sum(OK), '/', len(OK))
with open(OUT, 'x') as fh:
    json.dump({'results': RES, 'pass': all(OK), 'n': len(OK)}, fh, indent=1, ensure_ascii=False)
