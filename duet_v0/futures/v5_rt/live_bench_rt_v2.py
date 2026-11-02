"""live_bench_rt_v2: live bench v1 (futures/v5_rt/live_bench_rt.py + voice_v1.py, frozen in round 3, sha256 in
runs_mini/v5rt_1101/LIVE_BENCH_SHA.txt, never edited) with the round-4 latency settings (run v5rt_1101, round 4,
phase 1; this file is frozen after phase 1, sha256 in runs_mini/v5rt_1101/LIVE_BENCH_V2_SHA.txt). Val split only (17 val
Bach chorales + Twinkle); systems are tuned on val; this bundle has no test split.

What this file does: it installs the _NoSound tripwire before any other import, checks that live_bench_rt.py, voice_v1.py
and bench_rt.py are the frozen files (sha256 below; it refuses to run otherwise), imports live_bench_rt (which installs
its own tripwire and imports live_rt, live_rt_eval_v2 and voice_v1), adds the settings below to live_bench_rt.LAT at
import time (it never changes or shadows v1's 'wired' and 'bt') and runs live_bench_rt.evaluate unchanged. The JSON is
v1's (same fields, same digest rule: the digest covers the deterministic fields only, so the extra code entry does not
change it) plus code['live_bench_rt_v2.py'] = this file's sha256. Through this file --lat wired and --lat bt give
exactly v1's digests (checked before freezing).

CLI (v1's): <spec> --voice clean|sung|sung_low|phone_twinkle|phone_v11 [--lat <setting>] [--songs N] --out <new .json
under runs_mini/v5rt_1101/live_bench/> (refuses to overwrite). Phone files keep v1's fixed setup (Li 0, Lo 10 ms, told =
true, no count-in); v1 refuses any --lat other than 'wired' for them.

Latency model (v1's, unchanged; futures/v4_selfiter/live_rt.py and live_bench_rt.py's docstring): the engine (RTDuet,
count-in 4, lead 0.5 s) is told Li_told / Lo_told. Clicks and planned notes are written Lo_told early, so both sound
Lo_true - Lo_told after the engine's grid; she sings to the click she HEARS, so planned notes stay aligned with her
whatever Lo_told is. Her audio reaches the tracker Li_true late and the engine subtracts Li_told, so every heard onset is
mapped (Li_true + Lo_true) - (Li_told + Lo_told) late on the engine's grid: the ROUND-TRIP error. At 90 bpm (1 beat =
667 ms, onsets snapped to 0.25 beat = 167 ms) an error above about 83 ms moves heard onsets to the wrong grid point.
A calibration that measures the round trip RT (she sings 'da' on a few clicks; RT = median(detected onset - click write
time)) lets the page tell the engine Li_told = RT - Lo_told and removes that error. What is left is the real lag: a hear
decision sounds about Li_true + detect (~80 ms) + Lo_true after her onset, and the plan for point q is decided
Lo_told + PRE (25 ms) (+ up to 2 blocks) before q in engine time, knowing only the notes heard by then.

Settings (seconds): true Li, Lo; told Li_told, Lo_told; round-trip error = (Li + Lo) - (Li_told + Lo_told).
  wired          Li 0.012 Lo 0.010, told 0.012 / 0.010   (v1) wired mic + wired headphones, told the truth: error 0;
                                                          a hear sounds ~100 ms after her onset
  bt             Li 0.150 Lo 0.180, told 0.063 / 0.042   (v1) AirPods both ways, told PortAudio's numbers: error
                                                          +225 ms = 0.34 beat; heard onsets land on wrong grid points
  micbt          Li 0.150 Lo 0.010, told 0.063 / 0.010   AirPods microphone + wired headphones (the Mac mini has no
                                                          built-in mic; Vincent's likely morning setup), told
                                                          PortAudio's input latency: error +87 ms = 0.13 beat
  micbt_cal      Li 0.150 Lo 0.010, told 0.150 / 0.010   the same after calibration (RT 160 ms, Li_told = 160 - 10):
                                                          error 0; a hear sounds ~240-250 ms after her onset
  bt_cal         Li 0.150 Lo 0.180, told 0.288 / 0.042   AirPods both ways after calibration: Lo_told stays
                                                          PortAudio's 42 ms (the page measures only the round trip),
                                                          Li_told = RT - 42 = 330 - 42: error 0; clicks and planned
                                                          notes sound 138 ms after the engine grid, she follows that, so
                                                          they stay aligned with her; a hear sounds ~410 ms after her
                                                          onset and plans know only notes that began ~0.5-0.7 beat ago
  micbt_cal_m25  micbt_cal with Li_told 0.125: the calibration 25 ms short (error +25 ms)    robustness
  micbt_cal_p25  micbt_cal with Li_told 0.175: the calibration 25 ms long (error -25 ms)    robustness
  bt_cal_m25     bt_cal with Li_told 0.263 (error +25 ms)                                     robustness
  bt_cal_p25     bt_cal with Li_told 0.313 (error -25 ms)                                     robustness
Why: Vincent's microphones in the morning are Bluetooth (AirPods) or the iPhone, his main complaint is latency, and a
mis-told round trip moves every heard onset by a large part of a beat; these settings measure what calibration buys
(round-4 Q1 / C2) and score systems for large real lags (round-4 K1 / K2).

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python futures/v5_rt/live_bench_rt_v2.py rt:r3_X_regtrkst_v1 \
      --voice sung --lat bt_cal --out runs_mini/v5rt_1101/live_bench/<new name>.json
"""
import sys
import types


class _NoSound(types.ModuleType):
    """futures/v4_selfiter/live_rt_eval_v2.py's tripwire (as copied in live_bench_rt.py), in place before any other
    import: private / dunder lookups get the usual AttributeError; any public sound-device call raises at once."""

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('live_bench_rt_v2 is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.join(ABS, 'futures', 'v5_rt')
OUT_ROOT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'live_bench')
FROZEN = {'live_bench_rt.py': '46b43f91bf4bfa03f04c07ecdb543368394b6279f198cc8a21c435a8ab2bd8fd',
          'voice_v1.py': 'ed395a5acb7c4a1dbf341c22f41b3b772f74feaeb492880ff93ae8799220e2f9',
          'bench_rt.py': 'f1eb524aa939b5ba54704532ac7b5eb536659ceaaef32c3701934f9cf7d3e772'}


def _sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


for _f, _h in FROZEN.items():
    if _sha(os.path.join(HERE, _f)) != _h:
        raise SystemExit(f'live_bench_rt_v2: {_f} is not the frozen file (sha256 != {_h[:12]}...); refusing to run')

if HERE not in sys.path:
    sys.path.insert(0, HERE)
import live_bench_rt as LB1  # noqa: E402  (frozen v1: tripwire again, bench_rt, live_rt, live_rt_eval_v2, voice_v1)

NEW_LAT = {'micbt': {'Li': 0.150, 'Lo': 0.010, 'Li_told': 0.063, 'Lo_told': 0.010},
           'micbt_cal': {'Li': 0.150, 'Lo': 0.010, 'Li_told': 0.150, 'Lo_told': 0.010},
           'bt_cal': {'Li': 0.150, 'Lo': 0.180, 'Li_told': 0.288, 'Lo_told': 0.042},
           'micbt_cal_m25': {'Li': 0.150, 'Lo': 0.010, 'Li_told': 0.125, 'Lo_told': 0.010},
           'micbt_cal_p25': {'Li': 0.150, 'Lo': 0.010, 'Li_told': 0.175, 'Lo_told': 0.010},
           'bt_cal_m25': {'Li': 0.150, 'Lo': 0.180, 'Li_told': 0.263, 'Lo_told': 0.042},
           'bt_cal_p25': {'Li': 0.150, 'Lo': 0.180, 'Li_told': 0.313, 'Lo_told': 0.042}}
for _k, _v in NEW_LAT.items():
    if _k in LB1.LAT:
        raise SystemExit(f'live_bench_rt_v2: setting {_k!r} already exists in v1; v1 settings are never shadowed')
    LB1.LAT[_k] = dict(_v)
LAT = LB1.LAT


def round_trip_error(name):
    """(Li + Lo) - (Li_told + Lo_told) in seconds: how late every heard onset is mapped on the engine's grid."""
    s = LAT[name]
    return (s['Li'] + s['Lo']) - (s['Li_told'] + s['Lo_told'])


def evaluate(spec, voice, lat_name='wired', n_songs=None, system=None):
    """live_bench_rt.evaluate unchanged (with the settings above in its LAT), plus code['live_bench_rt_v2.py']."""
    res = LB1.evaluate(spec, voice, lat_name, n_songs, system)
    res['code']['live_bench_rt_v2.py'] = _sha(os.path.abspath(__file__))
    return res


def main():
    ap = argparse.ArgumentParser(description='live_bench_rt_v2: live bench v1 + round-4 latency settings (docstring)')
    ap.add_argument('spec', help='reactive:<v3 system> | rt:<rt_systems name>')
    ap.add_argument('--voice', required=True, choices=LB1.VOICES)
    ap.add_argument('--lat', default='wired', choices=tuple(LAT))
    ap.add_argument('--songs', type=int, default=None, help='quick subset: the first N val chorales')
    ap.add_argument('--out', required=True, help='new .json file under runs_mini/v5rt_1101/live_bench/')
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(OUT_ROOT + os.sep) or not path.endswith('.json'):
        sys.exit(f'--out must be a new .json file under {OUT_ROOT}/')
    if os.path.exists(path):
        sys.exit(f'refusing to overwrite {path}')
    res = evaluate(a.spec, a.voice, a.lat, a.songs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1, default=LB1._js)
    print(LB1.line(res), flush=True)


if __name__ == '__main__':
    main()
