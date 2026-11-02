"""VER4_latcal_check_v1 (verifier, run v5rt_1101, round 4): C1 re-check of futures/v4_selfiter/latcal_v1.estimate, offline.
No sound device: the sounddevice tripwire is installed before any other import; every recording is synthetic
(latcal_v1.make_test, Bach-style sung 'da') and never played. Writes ONE new JSON (mode 'x') under
runs_mini/v5rt_1101/verify/ (name must start with VER4_). Nothing here is tuned; seeds fixed before the first run.

A  the page agent's exact C1 case list (runs_mini/v5rt_1101/tools/latcal_selftest_v1.py's run_case, imported, its
   main() not called): main 20 / 60 / 160 / 330 ms x seeds 0-2, missing response, low voice, refuse x 3, extra x 8;
   C1 recomputed and every case compared with the agent's r4/P/latcal_selftest_v1.json.
B  wider Monte Carlo of the same C1 groups: seeds 100-124 x the 4 round trips x {normal, low voice, one missing}
   (300 cases) plus round trips between the C1 points (120 / 250 / 450 ms, seeds 100-109): how often is the estimate
   ok and within +-15 ms (the C1 chance margin), and refusals on 10 seeds x {silent, no response, bleed}.
C  report-only stand-ins for what C1's signals do not contain (AirPods microphone in hands-free mode), each a
   post-process of make_test's recording, seeds 100-109 x round trips 60 / 160 / 330 ms, normal and low voice:
     lp7k     zero-phase 8th-order low-pass at 7 kHz (16 kHz wideband speech codec band; no added delay)
     lp3k5    the same at 3.5 kHz (8 kHz narrowband codec)
     gate     a noise gate: everything at -20 dB except from each true response onset, where the gain ramps to 0 dB
              over 30 ms and stays open 350 ms (voice processing that opens late on a new sound)
     drop     packet loss: in each response, with probability 0.5, one 10 ms stretch inside its first 150 ms zeroed
     quiet    the whole recording 20 dB quieter (her voice about -50 dBFS RMS)
     offset40 she sings every 'da' 40 ms after the click (not an estimator error: a systematic offset is calibrated in;
              reported to show the estimate follows it)
   For each: ok rate, estimate minus true round trip (mean, max abs), share within +-15 ms.

  .venv/bin/python runs_mini/v5rt_1101/verify/VER4_latcal_check_v1.py --out runs_mini/v5rt_1101/verify/VER4_latcal_check_v1.json
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('VER4 latcal check is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
sys.path.insert(0, os.path.join(RUN, 'tools'))
import latcal_v1 as LC  # noqa: E402
import latcal_selftest_v1 as ST  # noqa: E402  (run_case only; its main() is never called)

TOL = 0.015


def sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def degrade(t, kind, seed):
    x = np.asarray(t['rec'], np.float64).copy()
    sr = t['sr']
    rng = np.random.default_rng([int(seed), 777, len(kind)])
    if kind == 'lp7k':
        x = sosfiltfilt(butter(8, 7000.0, 'lowpass', fs=sr, output='sos'), x)
    elif kind == 'lp3k5':
        x = sosfiltfilt(butter(8, 3500.0, 'lowpass', fs=sr, output='sos'), x)
    elif kind == 'gate':
        g = np.full(len(x), 10.0 ** (-20.0 / 20.0))
        for on in t['truth']:
            if on is None:
                continue
            a = int(round(on * sr))
            na, no = int(0.03 * sr), int(0.35 * sr)
            ramp = 10.0 ** ((-20.0 + 20.0 * np.arange(na) / na) / 20.0)
            e = min(len(x), a + na)
            g[a:e] = np.maximum(g[a:e], ramp[:e - a])
            g[e:min(len(x), a + no)] = 1.0
        x = x * g
    elif kind == 'drop':
        for on in t['truth']:
            if on is None or rng.random() >= 0.5:
                continue
            a = int(round((on + rng.uniform(0.0, 0.14)) * sr))
            x[a:a + int(0.01 * sr)] = 0.0
    elif kind == 'quiet':
        x = x * 0.1
    return x


def one(rt, seed, kind='none', **kw):
    t = LC.make_test(rt, seed, **kw)
    rec = t['rec'] if kind == 'none' else degrade(t, kind, seed)
    r = LC.estimate(rec, t['sr'], t['clicks'])
    err = None if r['rt_s'] is None else (r['rt_s'] - rt)
    return {'rt_ms': round(rt * 1000), 'seed': seed, 'kind': kind, 'opts': kw, 'ok': bool(r['ok']), 'why': r['why'],
            'est_ms': None if r['rt_s'] is None else round(r['rt_s'] * 1000, 2),
            'err_ms': None if err is None else round(err * 1000, 2), 'n_used': r['n_used'],
            'spread_ms': None if r['spread_s'] is None else round(r['spread_s'] * 1000, 2),
            'within': err is not None and abs(err) <= TOL + 1e-9}


def summ(rows, offset=0.0):
    errs = [r['err_ms'] - offset for r in rows if r['ok'] and r['err_ms'] is not None]
    return {'n': len(rows), 'ok': sum(r['ok'] for r in rows),
            'ok_and_within_15ms': sum(r['ok'] and r['err_ms'] is not None and abs(r['err_ms'] - offset) <= 15.0 + 1e-9
                                      for r in rows),
            'mean_err_ms': round(float(np.mean(errs)), 2) if errs else None,
            'max_abs_err_ms': round(float(np.max(np.abs(errs))), 2) if errs else None,
            'refused_why': sorted({r['why'] for r in rows if not r['ok']})[:4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out_path = os.path.abspath(ap.parse_args().out)
    if not out_path.startswith(os.path.join(RUN, 'verify') + os.sep) or not os.path.basename(out_path).startswith(
            'VER4_') or os.path.exists(out_path):
        sys.exit('--out must be a new VER4_*.json under runs_mini/v5rt_1101/verify/')
    t0 = time.time()
    res = {'what': 'verifier re-check of latcal_v1 (C1) on synthetic recordings; nothing played',
           'code': {'latcal_v1.py': sha(os.path.join(ABS, 'futures', 'v4_selfiter', 'latcal_v1.py')),
                    'latcal_selftest_v1.py': sha(os.path.join(RUN, 'tools', 'latcal_selftest_v1.py')),
                    'this': sha(os.path.abspath(__file__))}}
    # A: the page agent's own case list
    rows = []
    for rt in ST.RTS:
        for seed in (0, 1, 2):
            rows.append(ST.run_case('main', rt, seed))
    for rt in ST.RTS:
        rows.append(ST.run_case('missing', rt, 0, missing=2))
        rows.append(ST.run_case('low', rt, 0, low=True))
    rows.append(ST.run_case('refuse', 0.160, 0, silent=True))
    rows.append(ST.run_case('refuse', 0.160, 0, respond=False))
    rows.append(ST.run_case('refuse', 0.160, 0, respond=False, bleed=True))
    for rt in ST.RTS:
        rows.append(ST.run_case('extra', rt, 0, low=True, missing=5))
        rows.append(ST.run_case('extra', rt, 0, early=4))
    req = [r for r in rows if r['group'] in ('main', 'missing', 'low')]
    ref = [r for r in rows if r['group'] == 'refuse']
    with open(os.path.join(RUN, 'r4', 'P', 'latcal_selftest_v1.json')) as fh:
        theirs = json.load(fh)
    same = [(a['group'], a['rt_true_ms'], a['seed'], a['ok'] == b['ok'] and a['rt_ms'] == b['rt_ms']
             and a['n_used'] == b['n_used']) for a, b in zip(rows, theirs['cases'])]
    res['A'] = {'c1_pass': all(r['ok'] and r.get('within_15ms') for r in req) and not any(r['ok'] for r in ref),
                'required': len(req), 'required_ok_within': sum(bool(r['ok'] and r.get('within_15ms')) for r in req),
                'max_abs_err_ms': max(abs(r['err_ms']) for r in req), 'refused': sum(not r['ok'] for r in ref),
                'refuse_why': [r['why'] for r in ref],
                'extra': [(r['rt_true_ms'], r['opts'], r['ok'], r.get('err_ms')) for r in rows if r['group'] == 'extra'],
                'same_as_page_agent_file': all(x[3] for x in same) and len(rows) == len(theirs['cases']),
                'mismatches': [x for x in same if not x[3]],
                'errs_ms': [(r['group'], r['rt_true_ms'], r['seed'], r['err_ms']) for r in req]}
    print('A', json.dumps({k: v for k, v in res['A'].items() if k != 'errs_ms'}, ensure_ascii=False), flush=True)
    # B: wider Monte Carlo
    B = {}
    for name, kw in (('normal', {}), ('low', {'low': True}), ('missing', {'missing': 2})):
        rr = [one(rt, s, **kw) for rt in ST.RTS for s in range(100, 125)]
        B[name] = dict(summ(rr), fails=[(r['rt_ms'], r['seed'], r['ok'], r['err_ms']) for r in rr
                                         if not (r['ok'] and r['within'])])
    rr = [one(rt, s) for rt in (0.120, 0.250, 0.450) for s in range(100, 110)]
    B['between_120_250_450'] = dict(summ(rr), fails=[(r['rt_ms'], r['seed'], r['ok'], r['err_ms']) for r in rr
                                                     if not (r['ok'] and r['within'])])
    for name, kw in (('silent', {'silent': True}), ('no_response', {'respond': False}),
                     ('no_response_bleed', {'respond': False, 'bleed': True})):
        rr = [one(0.160, s, **kw) for s in range(100, 110)]
        B['refuse_' + name] = {'n': len(rr), 'refused': sum(not r['ok'] for r in rr),
                               'why': sorted({r['why'] for r in rr})[:3]}
    res['B'] = B
    print('B', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'fails'} for k, v in B.items()},
                          ensure_ascii=False), flush=True)
    # C: AirPods-like stand-ins (report only)
    C = {}
    for kind in ('lp7k', 'lp3k5', 'gate', 'drop', 'quiet'):
        for vname, kw in (('normal', {}), ('low', {'low': True})):
            rr = [one(rt, s, kind, **kw) for rt in (0.060, 0.160, 0.330) for s in range(100, 110)]
            C[f'{kind}_{vname}'] = dict(summ(rr), fails=[(r['rt_ms'], r['seed'], r['ok'], r['err_ms'], r['why'][:30])
                                                         for r in rr if not (r['ok'] and r['within'])][:8])
    rr = []
    for rt in (0.060, 0.160, 0.330):
        for s in range(100, 110):
            r = one(rt + 0.040, s)
            r['rt_ms'] = round(rt * 1000)
            r['err_ms'] = None if r['est_ms'] is None else round(r['est_ms'] - rt * 1000, 2)
            rr.append(r)
    C['offset40_normal'] = dict(summ(rr, offset=40.0), note='err measured against the device round trip + 40 ms')
    res['C'] = C
    print('C', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'fails'} for k, v in C.items()},
                          ensure_ascii=False), flush=True)
    res['secs'] = round(time.time() - t0, 1)
    with open(out_path, 'x') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False, default=float)
    print('wrote', out_path, res['secs'], 's', flush=True)


if __name__ == '__main__':
    main()
