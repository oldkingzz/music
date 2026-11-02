"""VER4_panel_check_v1 (verifier, run v5rt_1101, round 4): the page agent's offline page selftest re-run with a new output
file, plus two verifier parts with the round-4 pick on the page. NO sound device: the sounddevice tripwire is installed
before any other import (and again by the imported selftest); audio is fed through the page's own Session._cb 10 ms at a
time; nothing is played; the page server is never started; no list file is written. Val only; Twinkle is the only song
rendered (voice_v1 'sung', as the selftest's part c); every run is rated 作废.

1. runs_mini/v5rt_1101/tools/panel_v3_selftest_v1.py imported as a module (not edited); its OUT is pointed at a new
   VER4_*.json under runs_mini/v5rt_1101/verify/ and its main() is run unchanged (parts a v2 parity, b calibration through
   the page, c a calibrated song, d by_latency). Its run folders go to a new runs_mini/live/selftest_v3/<HHMMSS>/.
2. part e (verifier): the pick rt:r4_X_hgeh_v1 built into the page's MODELS (as the page's MODELS.load would for a list
   that names it) and a page Session with an IN-MEMORY list {default: the pick, systems: pick, W}:
   e1  AirPods both ways after calibration: pair BT calibrated through the page on a latcal_v1.make_test recording with
       true round trip 330 ms; Twinkle (voice_v1 'sung', 90 bpm) sung to the page's count-in clicks with true Li 150 /
       Lo 180 ms while the stream reports Li 63 / Lo 42 ms (live bench v2's bt_cal model). Checks: 'auto' runs the pick,
       RTDuet Li = RT - 42 ms and Lo = 42 ms, the pick's lag regimes switched on (LagWrap lag_on_at, core counters), and
       heard onsets on the score onsets. The same audio on an uncalibrated pair for comparison.
   e2  AirPods microphone + wired headphones after calibration (true Li 150 / Lo 10, RT 160 ms, stream 63 / 10 ms): the
       pick and W (explicit choices) on the same audio through the page must give identical events, decisions and output
       audio (the pick = W where its regimes are off), and the pick's regimes must stay off.
3. sha256 of live_panel_v2.py / .html, live_panel_v3.py / .html, latcal_v1.py and every live_rt_systems_v*.json before and
   after.

  .venv/bin/python runs_mini/v5rt_1101/verify/VER4_panel_check_v1.py --out runs_mini/v5rt_1101/verify/VER4_panel_check_v1.json
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('VER4 panel check is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
V4 = os.path.join(ABS, 'futures', 'v4_selfiter')
sys.path.insert(0, os.path.join(RUN, 'tools'))
import panel_v3_selftest_v1 as ST  # noqa: E402  (installs its own tripwire; imports live_panel_v2 / v3, latcal_v1)

P3, LC, VV, SR = ST.P3, ST.LC, ST.VV, ST.SR
PICK = 'r4_X_hgeh_v1'
FILES = ('live_panel_v2.py', 'live_panel_v2.html', 'live_panel_v3.py', 'live_panel_v3.html', 'latcal_v1.py')
CHECKS = []


def check(part, what, ok, detail=None):
    CHECKS.append({'part': part, 'what': what, 'ok': bool(ok), 'detail': detail})
    print(f"[{part}] {'OK  ' if ok else 'FAIL'} {what}" + ('' if detail is None else f'  {detail}'), flush=True)


def shas():
    d = {f: ST.sha_file(os.path.join(V4, f)) for f in FILES}
    d.update({os.path.basename(p): ST.sha_file(p) for p in glob.glob(os.path.join(V4, 'live_rt_systems_v*.json'))})
    return d


def render(Li_true, Lo_true, Lo_told, bpm=90.0):
    song = ST.twinkle()
    T_b = 60.0 / bpm
    t0 = 0.5 + 4 * T_b                                     # RTDuet's grid with count_in 4, lead 0.5
    her, notes, facts = VV.render(song, 'sung', t0 + (Lo_true - Lo_told), T_b)
    x = np.concatenate([np.zeros(int(round(Li_true * SR)), np.float32), her, np.zeros(SR, np.float32)])
    return x, [float(n[0]) for n in song['sop']]


def regime(sysobj):
    core = sysobj.core
    lob = list(getattr(core, 'g_lob', []))
    return {'h_lag_on_at': sysobj.info.get('lag_on_at'), 'h_n_d': sysobj.info.get('n_d'),
            'g_samples': len(lob), 'g_median': float(np.median(lob[-16:])) if len(lob) >= 3 else None,
            'counts': {k: int(core.stats.get(k, 0)) for k in ('lag_plan', 'lag_attack', 'late_keep', 'g_large',
                                                              'g_ehedge', 'g_late_pred', 'g_late_attack')}}


def part_e(root):
    train = P3.bench3.train_sets()
    t0 = time.time()
    P3.MODELS.rt[PICK] = P3.LR.rt_build(PICK, train)
    check('e', 'the pick builds inside the page process', PICK in P3.MODELS.rt, f'{time.time() - t0:.1f} s')
    W = P3.RT['default']
    test_rt = {'default': PICK, 'systems': {PICK: 'VER4 pick (test list in memory)', W: 'W'},
               'file': 'VER4 test list in memory (never written)'}
    s = P3.Session(os.path.join(root, 'e'), rt=test_rt)
    out = {}
    # e1: AirPods both ways after calibration (bt_cal model)
    BT = ('VER4 AirPods mic', 'VER4 AirPods out')
    cl = ST.cal_run(s, LC.make_test(0.330, 31)['rec'], BT, 'e1')
    check('e', 'e1 BT pair calibrated through the page (true round trip 330 ms)',
          cl['ok'] and abs(cl['rt_ms'] - 330.0) <= 15.0, (cl['ok'], cl['rt_ms'], cl['n_used'], cl['why']))
    x, score = render(0.150, 0.180, 0.042)
    p = {'song': 'twinkle', 'mode': 'rt', 'click': True, 'bpm': 90.0, 'input': 'mic'}
    before = dict(P3.MODELS.rt[PICK].core.stats)
    o, eng, rec, st = ST.song_run(s, p, x, sim={'devs': BT, 'latency': (0.063, 0.042)}, note='VER4 (e1) pick bt_cal')
    reg = regime(P3.MODELS.rt[PICK])
    reg['counts'] = {k: v - before.get(k, 0) for k, v in reg['counts'].items()}
    want = s.cals[BT]['rt_s'] - 0.042
    check('e', "e1 'auto' runs the pick (the list default)", o['name'] == PICK, o['name'])
    check('e', 'e1 RTDuet Li = RT - Lo (Lo = the stream output latency 42 ms)',
          abs(eng.Li - want) < 1e-12 and eng.Lo == 0.042, (round(eng.Li, 4), eng.Lo))
    check('e', "e1 run record says lat source 'calibrated'", rec['lat_source'] == 'calibrated', rec['lat_source'])
    check('e', "e1 the pick's lag regimes switched on (H lag mode and G large regime)",
          reg['h_lag_on_at'] is not None and reg['counts']['lag_plan'] > 0 and reg['counts']['g_large'] > 0, reg)
    sa = ST.onset_stats(eng, score)
    o_u, eng_u, rec_u, st_u = ST.song_run(s, p, x, sim={'devs': ('VER4 other mic', BT[1]), 'latency': (0.063, 0.042)},
                                          note='VER4 (e1) pick not calibrated')
    su = ST.onset_stats(eng_u, score)
    check('e', 'e1 calibrated: most heard onsets on the score onset; uncalibrated: far fewer',
          sa['on_score_onset'] >= 0.7 * sa['heard'] and su['on_score_onset'] < sa['on_score_onset'], (sa, su))
    out['e1'] = {'cal': cl, 'Li': eng.Li, 'Lo': eng.Lo, 'regime': reg, 'onsets_calibrated': sa,
                 'onsets_uncalibrated': su, 'run': o, 'uncalibrated_Li': eng_u.Li}
    # e2: AirPods microphone + wired headphones after calibration: pick == W
    MB = ('VER4 AirPods mic (wired out)', 'VER4 wired phones')
    cl2 = ST.cal_run(s, LC.make_test(0.160, 32)['rec'], MB, 'e2')
    check('e', 'e2 MB pair calibrated through the page (true round trip 160 ms)',
          cl2['ok'] and abs(cl2['rt_ms'] - 160.0) <= 15.0, (cl2['ok'], cl2['rt_ms'], cl2['n_used']))
    x2, _ = render(0.150, 0.010, 0.010)
    before = dict(P3.MODELS.rt[PICK].core.stats)
    o_p = ST.song_run(s, dict(p, system=PICK), x2, sim={'devs': MB, 'latency': (0.063, 0.010)}, note='VER4 (e2) pick')
    reg2 = regime(P3.MODELS.rt[PICK])
    reg2['counts'] = {k: v - before.get(k, 0) for k, v in reg2['counts'].items()}
    o_w = ST.song_run(s, dict(p, system=W), x2, sim={'devs': MB, 'latency': (0.063, 0.010)}, note='VER4 (e2) W')
    keys = ('on_time', 'switched', 'answered', 'sung', 'cons', 'events_sha', 'decisions_sha', 'out_sha', 'Li', 'Lo')
    diff = [k for k in keys if o_p[0][k] != o_w[0][k]]
    check('e', 'e2 the pick and W give identical events, decisions and output audio through the page', not diff,
          diff or {k: o_p[0][k] for k in ('on_time', 'switched', 'answered', 'sung', 'cons', 'Li', 'Lo')})
    check('e', "e2 the pick's regimes stayed off", reg2['h_lag_on_at'] is None and reg2['counts']['lag_plan'] == 0
          and reg2['counts']['g_large'] == 0, reg2)
    out['e2'] = {'cal': cl2, 'pick': o_p[0], 'W': o_w[0], 'diff': diff, 'regime': reg2}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out_path = os.path.abspath(ap.parse_args().out)
    if not out_path.startswith(os.path.join(RUN, 'verify') + os.sep) or not os.path.basename(out_path).startswith(
            'VER4_') or os.path.exists(out_path):
        sys.exit('--out must be a new VER4_*.json under runs_mini/v5rt_1101/verify/')
    sel_out = out_path[:-len('.json')] + '_selftest.json'
    if os.path.exists(sel_out):
        sys.exit(f'refusing to overwrite {sel_out}')
    sh0 = shas()
    ST.OUT = sel_out
    t0 = time.time()
    res = {'what': 'verifier re-run of panel_v3_selftest_v1 (offline, tripwire) + part e with the round-4 pick',
           'selftest_out': os.path.relpath(sel_out, ABS), 'root': os.path.relpath(ST.ROOT, ABS), 'errors': []}
    try:
        ST.main()
    except SystemExit as e:
        res['errors'].append({'part': 'selftest', 'exit': str(e)})
    with open(sel_out) as fh:
        sel = json.load(fh)
    res['selftest'] = {'all_pass': sel['all_pass'], 'pass': sel['pass'], 'n_checks': len(sel['checks']),
                       'n_ok': sum(c['ok'] for c in sel['checks']), 'errors': len(sel['errors']),
                       'failed': [c for c in sel['checks'] if not c['ok']], 'seconds': sel['seconds']}
    try:
        res['e'] = part_e(ST.ROOT)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        res['errors'].append({'part': 'e', 'traceback': tb})
        check('e', 'part e ran without an exception', False)
    sh1 = shas()
    check('all', 'page files, latcal_v1 and every list file unchanged; no new list file', sh0 == sh1, sorted(sh1))
    res['checks'] = CHECKS
    res['all_pass'] = res['selftest']['all_pass'] and all(c['ok'] for c in CHECKS) and not res['errors']
    res['shas'] = sh1
    res['seconds'] = round(time.time() - t0, 1)
    with open(out_path, 'x') as fh:
        json.dump(res, fh, indent=1, default=float, ensure_ascii=False)
    print(json.dumps({'selftest_all_pass': res['selftest']['all_pass'], 'selftest_ok': f"{res['selftest']['n_ok']}/"
                      f"{res['selftest']['n_checks']}", 'verifier_checks': f"{sum(c['ok'] for c in CHECKS)}/{len(CHECKS)}",
                      'all_pass': res['all_pass'], 'seconds': res['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
