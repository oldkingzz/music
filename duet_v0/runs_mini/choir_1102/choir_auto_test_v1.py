"""Offline test of page v8's choir 'auto' (choir_v4.ChoirAuto), no sound device (tripwire first).
Synthetic singers (repo singer.render) and Vincent's own recorded runs, fed through the page Session (simulated mic
with the true input delay, engine told the same). For each case: which song / key it recognises, when the parts come
in, and where each part chord sounds against her melody note on the same beat after recognition
(sound time - her onset; Duet writes PRE + LEAD = 55 ms early on purpose, so about -55 ms is on time).

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/choir_auto_test_v1.py
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

import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import live_panel_v8 as P8  # noqa: E402
import choir_v4 as CV4  # noqa: E402
P3, P5 = P8.P3, P8.P5
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import singer  # noqa: E402
from dsp import SR  # noqa: E402

ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_choir', time.strftime('%H%M%S') + '_v8')
LOG = []
_init = CV4.ChoirAuto.__init__


def _init_log(self, *a, **k):
    _init(self, *a, **k)
    sch = self.choir.schedule

    def schedule(start, dur, midi, tag=None):
        LOG.append((start, tag[1] if tag else None))
        return sch(start, dur, midi, tag=tag)
    self.choir.schedule = schedule


CV4.ChoirAuto.__init__ = _init_log
OK = []


def check(what, ok, detail=''):
    OK.append(bool(ok))
    print(('OK   ' if ok else 'FAIL ') + what, detail, flush=True)


def run(x, li, lo, bpm=80.0):
    LOG.clear()
    s = P3.Session(ROOT)
    x = np.concatenate([np.zeros(int(li * SR), np.float32), np.asarray(x, np.float32)])
    s.cals[('m', 'p')] = {'in': 'm', 'out': 'p', 'rt_s': li + lo - 0.020, 'rt_ms': (li + lo - 0.02) * 1000,
                          'n_used': 8, 'spread_ms': 5, 'when': 't', 'stream_Li_ms': 60, 'stream_Lo_ms': lo * 1000}
    s.start({'song': 'auto', 'bpm': bpm, 'mode': 'choir', 'click': False, 'input': 'mic'},
            source=x, sim={'devs': ('m', 'p'), 'latency': (0.06, lo)})
    t1 = time.time()
    while s.state != 'await' and time.time() - t1 < 180:
        time.sleep(0.05)
    eng = s.eng
    s.verdict({'run': s.status()['run'], 'verdict': '作废', 'note': 'choir v8 auto test'})
    time.sleep(1.1)
    return eng, list(LOG)


def timing(eng, log, her_on):
    """her_on: {song beat: her true onset time (engine clock)}; chord sound time - her onset, per part note on a beat."""
    out = []
    for start, idx in log:
        on = eng.parts[idx][0]
        if on in her_on and on > 0:
            out.append((start / SR + eng.Lo) - her_on[on])
    return out


def sing(song, shift, octave, bpm_her, start_beat, li=0.15, lo=0.01, page_bpm=80.0):
    arr = P5.CHOIR[song]
    T_b = 60.0 / bpm_her
    T0 = 0.5 + 4 * 60.0 / page_bpm + start_beat * 60.0 / page_bpm
    mel = arr['melody']
    notes = [(T0 + a * T_b, T0 + b * T_b, m + shift + 12 * octave) for a, b, m in mel]
    x = np.concatenate([singer.render(notes)[0], np.zeros(2 * SR, np.float32)])
    eng, log = run(x, li, lo, page_bpm)
    her_on = {}
    for a, b, m in mel:
        her_on.setdefault(a, T0 + a * T_b)
    tm = timing(eng, log, her_on)
    first_in = min((s for s, i in log), default=None)
    print(f'{song} shift {shift:+d} octave {octave:+d} her {bpm_her} bpm (page {page_bpm}), starts beat {start_beat}: '
          f'got {eng.song_id} shift {eng.shift}, parts in at {None if first_in is None else round((first_in / SR - T0) / T_b, 2)} '
          f'beats, chord - her onset ms: median {np.median(tm) * 1000 if tm else None:.0f}, p90 abs '
          f'{np.percentile(np.abs(np.array(tm) * 1000 + 55), 90) if tm else None:.0f} from on-time, tempo {60 / eng.Tb:.1f} bpm')
    return eng, tm


P3.MODELS.load()
e, tm = sing('twinkle', 0, 0, 80.0, 0)
check('twinkle C at page tempo: recognised', e.song_id == 'twinkle' and e.shift == 0)
check('  chords within 40 ms of on time (median)', abs(np.median(tm) * 1000 + 55) <= 40)
e, tm = sing('twinkle', 2, -1, 72.0, 2)
check('twinkle in D, an octave low, 10 % slower, 2 beats late: recognised in D', e.song_id == 'twinkle' and e.shift == 2)
check('  follows her tempo: chords within 60 ms of on time (median)', abs(np.median(tm) * 1000 + 55) <= 60)
e, tm = sing('v11', -2, -1, 90.0, 0, page_bpm=90.0)
check('v11 in G, an octave low: recognised', e.song_id == 'v11' and e.shift == -2)
check('  chords within 40 ms of on time (median)', abs(np.median(tm) * 1000 + 55) <= 40)
for name, li in (('094919_r2_mix_v2', 0.194), ('095417_r2_mix_v2', 0.194)):
    w, sr = sf.read(os.path.join(ABS, 'runs_mini/live', name, 'run.wav'), dtype='float32', always_2d=True)
    eng, log = run(w[:, 0], 0.0, 0.0421)
    print(f'replay {name}: got {eng.song_id} shift {eng.shift}, her distinct {[m for t, m in eng.her[:8]]}, '
          f'tempo {60 / eng.Tb:.1f} bpm, moves {eng.moves[:12]}')
print('ALL PASS' if all(OK) else 'SOME FAILED', len(OK))
