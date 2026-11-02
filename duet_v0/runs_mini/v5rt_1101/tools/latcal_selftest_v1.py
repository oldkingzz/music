"""C1 offline test of futures/v4_selfiter/latcal_v1.py (run v5rt_1101, round 4, page agent). No sound device: the
sounddevice tripwire is installed first; recordings are synthetic (latcal_v1.make_test) and never played.

Cases (fixed before the first run; seeds 0, 1, 2 chosen in advance, never re-picked):
  main         true round trips 20 / 60 / 160 / 330 ms x seeds 0, 1, 2: normal register, all 8 responses
  missing      the 4 round trips, seed 0, one missing response (click 3 of 8)
  low          the 4 round trips, seed 0, a low voice (around MIDI 50)
  refuse       silent (all zeros), no response (room noise and breaths only), no response with the clicks leaking into
               the microphone (loudspeakers) and breaths: estimate must give ok = False
  extra (reported, not part of C1): low voice + one missing response (click 6), and one early response (click 5 sung
               120 ms early), the 4 round trips, seed 0
C1 passes if every main / missing / low case is ok and within +-15 ms of the true round trip and every refuse case gives
ok = False. Each case also reports the detector's own error: the estimate minus the median of her TRUE offsets on the
clicks the estimator used (her 15 ms timing jitter does not enter that number), and the largest per-response detection
error.

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python runs_mini/v5rt_1101/tools/latcal_selftest_v1.py
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('latcal selftest is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import latcal_v1 as LC  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r4', 'P', 'latcal_selftest_v1.json')
RTS = (0.020, 0.060, 0.160, 0.330)
TOL = 0.015


def sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def run_case(group, rt, seed, **kw):
    t = LC.make_test(rt, seed, **kw)
    c0 = time.perf_counter()
    r = LC.estimate(t['rec'], t['sr'], t['clicks'])
    secs = time.perf_counter() - c0
    row = {'group': group, 'rt_true_ms': round(rt * 1000), 'seed': seed, 'opts': {k: v for k, v in kw.items()},
           'ok': r['ok'], 'why': r['why'], 'rt_ms': None if r['rt_s'] is None else round(r['rt_s'] * 1000, 1),
           'n_used': r['n_used'], 'spread_ms': None if r['spread_s'] is None else round(r['spread_s'] * 1000, 1),
           'n_onsets': r['n_onsets'], 'floor_db': r.get('floor_db'), 'rejected': r.get('rejected'),
           'level_dbfs': t['level_dbfs'], 'offsets_ms': r['offsets_ms'], 'secs': round(secs, 3)}
    if r['rt_s'] is not None and t['kind'] == 'response':
        row['err_ms'] = round((r['rt_s'] - rt) * 1000, 1)
        true_off = [None if on is None else on - c / t['sr'] for on, c in zip(t['truth'], t['clicks'])]
        used = [j for j, o in enumerate(r['offsets_ms']) if o is not None and true_off[j] is not None]
        if used:
            row['true_median_used_ms'] = round(float(np.median([true_off[j] for j in used])) * 1000, 1)
            row['detector_err_ms'] = round(row['rt_ms'] - row['true_median_used_ms'], 1)
            row['max_response_err_ms'] = round(max(abs(r['offsets_ms'][j] - true_off[j] * 1000) for j in used), 1)
        row['her_jitter_median_err_ms'] = round(float(np.median([o for o in true_off if o is not None]) - rt) * 1000, 1)
        row['within_15ms'] = abs(r['rt_s'] - rt) <= TOL + 1e-9
    return row


def main():
    if os.path.exists(OUT):
        sys.exit(f'refusing to overwrite {OUT}')
    rows = []
    for rt in RTS:
        for seed in (0, 1, 2):
            rows.append(run_case('main', rt, seed))
    for rt in RTS:
        rows.append(run_case('missing', rt, 0, missing=2))
        rows.append(run_case('low', rt, 0, low=True))
    rows.append(run_case('refuse', 0.160, 0, silent=True))
    rows.append(run_case('refuse', 0.160, 0, respond=False))
    rows.append(run_case('refuse', 0.160, 0, respond=False, bleed=True))
    for rt in RTS:
        rows.append(run_case('extra', rt, 0, low=True, missing=5))
        rows.append(run_case('extra', rt, 0, early=4))
    req = [r for r in rows if r['group'] in ('main', 'missing', 'low')]
    fails = [r for r in req if not (r['ok'] and r.get('within_15ms'))]
    ref = [r for r in rows if r['group'] == 'refuse']
    ref_fail = [r for r in ref if r['ok']]
    extra = [r for r in rows if r['group'] == 'extra']
    c1 = not fails and not ref_fail
    errs = [r['err_ms'] for r in req]
    det = [r['detector_err_ms'] for r in req if 'detector_err_ms' in r]
    summary = {
        'c1_pass': c1,
        'required_cases': len(req), 'required_fail': len(fails), 'refuse_cases': len(ref), 'refuse_fail': len(ref_fail),
        'err_ms_by_case': {f"{r['group']} rt{r['rt_true_ms']} s{r['seed']}": r['err_ms'] for r in req},
        'max_abs_err_ms': max(abs(e) for e in errs), 'mean_err_ms': round(float(np.mean(errs)), 2),
        'detector_err_ms_mean': round(float(np.mean(det)), 2), 'detector_err_ms_max_abs': max(abs(e) for e in det),
        'max_response_err_ms': max(r['max_response_err_ms'] for r in req if 'max_response_err_ms' in r),
        'refuse_why': [r['why'] for r in ref],
        'extra': {f"{'low_missing' if r['opts'].get('low') else 'early'} rt{r['rt_true_ms']}":
                  {'ok': r['ok'], 'err_ms': r.get('err_ms'), 'n_used': r['n_used']} for r in extra},
        'extra_all_ok_within_15ms': all(r['ok'] and r.get('within_15ms') for r in extra),
    }
    res = {'what': 'C1 offline test of latcal_v1.estimate on latcal_v1.make_test recordings (synthetic, never played)',
           'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'tolerance_ms': TOL * 1000,
           'code': {'latcal_v1.py': sha(os.path.join(ABS, 'futures', 'v4_selfiter', 'latcal_v1.py')),
                    'latcal_selftest_v1.py': sha(os.path.abspath(__file__))},
           'schedule': {k: v for k, v in LC.schedule().items()},
           'summary': summary, 'cases': rows}
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False, default=float)
    for r in rows:
        print(f"{r['group']:8s} rt {r['rt_true_ms']:3d} s{r['seed']} {json.dumps(r['opts'], ensure_ascii=False):32s} "
              f"ok {str(r['ok']):5s} rt {r['rt_ms']} err {r.get('err_ms')} det {r.get('detector_err_ms')} "
              f"maxresp {r.get('max_response_err_ms')} used {r['n_used']} spread {r['spread_ms']} "
              f"onsets {r['n_onsets']} rej {r['rejected']} {r['why']}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
