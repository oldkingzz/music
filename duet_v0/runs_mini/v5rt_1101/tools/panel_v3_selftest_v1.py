"""Offline self-test of futures/v4_selfiter/live_panel_v3.py with NO sound device (Vincent asleep, night of 2026-11-01).

The sounddevice tripwire (live_rt_eval_v2's _NoSound) is installed before anything else is imported: any public
sounddevice call raises. Audio is fed through the pages' own Session code 10 ms at a time (Session._cb, the callback a
live stream drives); nothing is played. Every song run is rated 作废. Outputs: runs_mini/live/selftest_v3/<HHMMSS>/
(a_v2, a_v3, b, c, d), summary runs_mini/v5rt_1101/r4/P/panel_v3_selftest_v1.json.

(a) v2 parity: v2's own selftest runs (every rt system of the list on Twinkle, the default on v11, the rt A/B blind
    test, reactive r2_mix_v2 on Twinkle) through live_panel_v2.Session and live_panel_v3.Session in this process, each
    page with its own freshly built models, the A/B arm drawn after the same random.seed. Equal: system, mode, engine,
    label, on time, switched, answered, sung, consonant with heard, rows, acts, the full event lists and RTDuet decisions
    (timing fields 'us' / 'step_ms' left out) and Duet's output audio (sha256 of the float32 samples).
(b) calibration through the page: a latcal_v1.make_test recording (true round trip 160 ms, seed 11) fed to
    Session.calibrate (CalRun on Session._cb) for the device pair A: ok, within 15 ms of 160, exactly latcal_v1.estimate
    of the same samples, the recording equals the input, every click starts in Duet's output channel exactly one sample
    after its schedule write sample (the click is a sine starting at phase 0), one latcal_log.jsonl line per attempt, a
    cal folder with cal.wav and cal.json; then a no-response recording on the same pair: not ok, a reason, the earlier
    calibration kept; calibrating with input 'file' is refused.
(c) a song with that calibration: Twinkle rendered by voice_v1 ('sung', 90 bpm) singing to the page's count-in clicks
    with the micbt latency (true Li 150 ms, Lo 10 ms, so the true round trip is 160 ms; the stream reports Li 63 ms, Lo
    10 ms) and fed to Session.start (rt 'auto', click mode) on pair A: RTDuet gets Li = RT - Lo and Lo = 10 ms, the run
    and notes.json record lat source 'calibrated'. The same audio on pair B (not calibrated) gets PortAudio's 63 ms.
    Reported (not a gate): how many heard onsets land exactly on a score onset in each case.
(d) by_latency with a test list in memory (no list file is ever written; the list files' sha256 are checked before and
    after): {"threshold_ms": 120, "system": <a non-default system of the list>}; pairs A (160 ms) and B (60 ms)
    calibrated through the page. pick_rt at 119.9 / 120 ms / none; 'auto' (no system given) on A runs the by_latency
    system and says why; 'auto' on B runs the default; an explicit system is honored on A; the rt blind A/B on A only
    ever plays reactive r2_mix_v2 or the by_latency system (4 runs, seeds 0-3).

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python runs_mini/v5rt_1101/tools/panel_v3_selftest_v1.py
"""
import sys
import types

ABS = '/Users/san/claude_sandbox/music/duet_v0'


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('panel selftest is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

V4 = os.path.join(ABS, 'futures', 'v4_selfiter')
sys.path.insert(0, V4)
import live_panel_v2 as P2  # noqa: E402
import live_panel_v3 as P3  # noqa: E402
import latcal_v1 as LC  # noqa: E402
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import voice_v1 as VV  # noqa: E402  (frozen renderer; Bach chorales and Twinkle only)
import soundfile as sf  # noqa: E402
from songs import twinkle  # noqa: E402
from dsp import SR  # noqa: E402

OUT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r4', 'P', 'panel_v3_selftest_v1.json')
ROOT = os.path.join(P3.LB.OUT_ROOT, 'selftest_v3', time.strftime('%H%M%S'))
PHONE = {'twinkle': 'twinkle_C_80bpm_countin.wav', 'v11': 'test_her_v11_countin.wav'}
STREAM = (0.063, 0.010)                      # what PortAudio reports for an AirPods mic + wired headphones (live bench)
PAIR_A = ('selftest mic A (AirPods)', 'selftest phones (wired)')
PAIR_B = ('selftest mic B', 'selftest phones (wired)')
CHECKS = []


def check(part, what, ok, detail=None):
    CHECKS.append({'part': part, 'what': what, 'ok': bool(ok), 'detail': detail})
    print(f"[{part}] {'OK  ' if ok else 'FAIL'} {what}" + ('' if detail is None else f'  {detail}'), flush=True)
    return bool(ok)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    with open(path, 'rb') as fh:
        return sha_bytes(fh.read())


def sha_json(o):
    return sha_bytes(json.dumps(o, sort_keys=True, default=float, ensure_ascii=False).encode())


def strip(d):
    return {k: v for k, v in d.items() if k not in ('us', 'step_ms')}


def wait(s, states, timeout=300.0):
    t0 = time.time()
    while s.state not in states and time.time() - t0 < timeout:
        time.sleep(0.02)
    if s.state not in states:
        raise RuntimeError(f'state {s.state} after {timeout} s, wanted {states}')


def song_run(s, p, x, sim=None, note='selftest v3'):
    """One run fed offline through a page Session (v2 or v3), rated 作废. Returns the facts compared."""
    st = s.start(p, source=x) if sim is None else s.start(p, source=x, sim=sim)
    assert st['state'] == 'running', st['state']
    wait(s, ('await',))
    st = s.status()
    eng = s.eng
    rec = s.verdict({'run': st['run'], 'verdict': '作废', 'note': note})
    ev = eng.events
    cons = sum(e['duet'] >= 0 and abs(e['her_heard'] - e['duet']) % 12 in P3.CONS for e in ev)
    acts = sorted({r['act'] for r in st['rows'] if r['act']})
    out = {'run': rec['run'], 'system': rec['system'], 'mode': rec['mode'], 'engine': rec['engine'],
           'label': rec['label'], 'on_time': rec['on_time'], 'switched': rec['switched'],
           'answered': rec['notes_answered'], 'sung': rec['sung'], 'cons': cons, 'rows': len(st['rows']), 'acts': acts,
           'events_sha': sha_json([strip(e) for e in ev]),
           'decisions_sha': sha_json([strip(d) for d in getattr(eng, 'decisions', [])]),
           'out_sha': sha_bytes(np.concatenate(s.rec_out).astype(np.float32).tobytes()),
           'Li': float(eng.Li), 'Lo': float(eng.Lo), 'name': s.name}
    time.sleep(1.1)                                        # run folders are named by the second
    return out, eng, rec, st


def cal_run(s, rec, pair, note):
    st = s.calibrate({'input': 'mic'}, source=rec, sim={'devs': pair, 'latency': STREAM})
    assert st['state'] == 'calibrating', st['state']
    wait(s, ('idle',))
    time.sleep(1.1)
    return s.cal_last


def phone(song, seconds=None):
    x, sr = sf.read(os.path.join(ABS, 'futures', 'live_test', 'phone', PHONE[song]), dtype='float32', always_2d=True)
    assert sr == SR
    x = x.mean(1)
    return x if seconds is None else x[:int(seconds * SR)]


def part_a():
    P2.MODELS.load()
    assert P2.MODELS.err is None, P2.MODELS.err
    s2 = P2.Session(os.path.join(ROOT, 'a_v2'))
    s3 = P3.Session(os.path.join(ROOT, 'a_v3'))
    runs = [(name, 'twinkle', 'rt') for name in P2.RT['systems']]
    runs += [(P2.RT['default'], 'v11', 'rt'), ('blind', 'twinkle', 'rt'), ('r2_mix_v2', 'twinkle', 'reactive')]
    check('a', 'v2 and v3 read the same rt list', P2.RT == {k: v for k, v in P3.RT.items()}, P3.RT['file'])
    keys = ('system', 'mode', 'engine', 'label', 'on_time', 'switched', 'answered', 'sung', 'cons', 'rows', 'acts',
            'events_sha', 'decisions_sha', 'out_sha', 'Li', 'Lo')
    rows = []
    for i, (choice, song, mode) in enumerate(runs):
        x = phone(song)
        p = {'system': choice, 'song': song, 'mode': mode}
        random.seed(1000 + i)
        o2 = song_run(s2, p, x, note='selftest v3 (a) through v2')[0]
        random.seed(1000 + i)
        o3 = song_run(s3, p, x, note='selftest v3 (a) through v3')[0]
        diff = [k for k in keys if o2[k] != o3[k]]
        check('a', f'{choice} {song} {mode}: v3 == v2', not diff,
              f"{o3['mode']} {o3['engine']} {o3['system']}: on time {o3['on_time']} switched {o3['switched']} answered "
              f"{o3['answered']} sung {o3['sung']} consonant {o3['cons']} rows {o3['rows']}" + (f' DIFF {diff}' if diff else ''))
        rows.append({'run': [choice, song, mode], 'v2': o2, 'v3': o3, 'diff': diff})
    return rows


def part_b():
    s = P3.Session(os.path.join(ROOT, 'b'))
    sched = P3.CalRun().sched
    t = LC.make_test(0.160, 11, sched=sched)
    try:
        s.calibrate({'input': 'file'})
        refused = False
    except ValueError as e:
        refused = '麦克风' in str(e)
    check('b', "calibrate with input 'file' is refused", refused)
    cl = cal_run(s, t['rec'], PAIR_A, 'b')
    cal = s.eng
    recd = cal.recording()
    direct = LC.estimate(recd, SR, t['clicks'])
    check('b', 'the CalRun recording equals the fed input sample for sample',
          len(recd) == len(t['rec']) // P3.HOP * P3.HOP and np.array_equal(recd, t['rec'][:len(recd)]), len(recd))
    check('b', 'calibration ok', cl['ok'], cl['why'])
    check('b', 'round trip within 15 ms of the true 160 ms', cl['rt_ms'] is not None and abs(cl['rt_ms'] - 160.0) <= 15.0,
          f"{cl['rt_ms']} ms, used {cl['n_used']}, spread {cl['spread_ms']} ms")
    check('b', 'page result == latcal_v1.estimate on the same samples',
          direct['rt_s'] is not None and cl['rt_ms'] == round(direct['rt_s'] * 1000, 1) and cl['n_used'] == direct['n_used'],
          f"direct {direct['rt_s']}")
    out = np.concatenate(s.rec_out)
    starts = []
    for c in sched['count_in'] + sched['clicks']:
        nz = np.nonzero(out[max(c - 200, 0):c + 200])[0]
        starts.append(int(nz[0]) + max(c - 200, 0) - c if len(nz) else None)
    check('b', 'each click starts in the output channel exactly 1 sample after its write sample (sin phase 0)',
          all(v == 1 for v in starts), starts)
    kept = s.cals.get(PAIR_A)
    check('b', 'the result is kept for device pair A', kept is not None and kept['rt_s'] == direct['rt_s'], kept)
    folder = os.path.join(s.root, cl['folder'])
    check('b', 'cal folder has cal.wav and cal.json', os.path.exists(os.path.join(folder, 'cal.wav'))
          and os.path.exists(os.path.join(folder, 'cal.json')), cl['folder'])
    t0 = LC.make_test(0.160, 12, respond=False)
    bad = cal_run(s, t0['rec'], PAIR_A, 'b')
    check('b', 'a no-response recording is not ok and says why', (not bad['ok']) and bool(bad['why']), bad['why'])
    check('b', 'the earlier calibration of pair A is kept after the failed retry',
          s.cals.get(PAIR_A) is not None and s.cals[PAIR_A]['rt_ms'] == cl['rt_ms'])
    with open(os.path.join(s.root, 'latcal_log.jsonl')) as fh:
        log = [json.loads(line) for line in fh if line.strip()]
    check('b', 'latcal_log.jsonl has one line per attempt (append only)', len(log) == 2 and log[0]['ok'] and not log[1]['ok'],
          [(r['ok'], r['rt_ms'], r['in']) for r in log])
    st = s.status()
    check('b', 'status lists the calibration for the page', any(c['in'] == PAIR_A[0] and c['out'] == PAIR_A[1]
                                                                  for c in st['cals']) and st['cal_last'] is not None)
    return s, cl, {'make_test': {'rt_s': 0.160, 'seed': 11, 'level_dbfs': t['level_dbfs']}, 'result': cl,
                   'direct': {k: direct[k] for k in ('rt_s', 'n_used', 'spread_s', 'ok', 'offsets_ms')},
                   'click_start_offsets': starts, 'failed_retry': bad}


def onset_stats(eng, score_onsets):
    ev = eng.events
    on = np.asarray(score_onsets)
    exact, dev = 0, []
    for e in ev:
        d = float(np.min(np.abs(on - e['onset_beats'])))
        exact += d < 1e-6
        dev.append(float(np.min(np.abs(on - e['onset_raw']))))
    return {'heard': len(ev), 'on_score_onset': int(exact), 'mean_abs_raw_dev_beats': round(float(np.mean(dev)), 3)
            if dev else None, 'median_detect_ms': float(np.median([e['detect_ms'] for e in ev])) if ev else None}


def part_c(s, cl):
    song = twinkle()
    bpm = 90.0
    T_b = 60.0 / bpm
    t0 = 0.5 + 4 * T_b                                     # RTDuet's grid with count_in 4, lead 0.5
    Li_true, Lo_true = 0.150, 0.010
    her, notes, facts = VV.render(song, 'sung', t0 + (Lo_true - STREAM[1]), T_b)
    x = np.concatenate([np.zeros(int(round(Li_true * SR)), np.float32), her, np.zeros(SR, np.float32)])
    p = {'song': 'twinkle', 'mode': 'rt', 'click': True, 'bpm': bpm, 'input': 'mic'}
    score = [float(n[0]) for n in song['sop']]
    o_a, eng_a, rec_a, st_a = song_run(s, p, x, sim={'devs': PAIR_A, 'latency': STREAM}, note='selftest v3 (c) calibrated')
    want = max(0.0, s.cals[PAIR_A]['rt_s'] - STREAM[1])
    check('c', 'calibrated run: RTDuet Li = RT - Lo', abs(eng_a.Li - want) < 1e-12, f'Li {eng_a.Li:.4f} = {want:.4f}')
    check('c', 'calibrated run: RTDuet Lo = the stream output latency', eng_a.Lo == STREAM[1], eng_a.Lo)
    with open(os.path.join(s.root, rec_a['run'], 'notes.json')) as fh:
        nj = json.load(fh)
    check('c', "notes.json records Li, Lo and lat source 'calibrated'", nj['lat']['source'] == 'calibrated'
          and abs(nj['Li'] - want) < 1e-12 and nj['Lo'] == STREAM[1] and rec_a['lat_source'] == 'calibrated', nj['lat'])
    check('c', "the run used the list default through 'auto' (no by_latency in the list)",
          o_a['name'] == P3.RT['default'] if not P3.RT.get('by_latency') else True, o_a['name'])
    o_b, eng_b, rec_b, st_b = song_run(s, p, x, sim={'devs': PAIR_B, 'latency': STREAM}, note='selftest v3 (c) not calibrated')
    check('c', "uncalibrated pair: PortAudio's Li 63 ms and Lo 10 ms, source 'portaudio'",
          eng_b.Li == STREAM[0] and eng_b.Lo == STREAM[1] and rec_b['lat_source'] == 'portaudio', (eng_b.Li, eng_b.Lo))
    sa, sb = onset_stats(eng_a, score), onset_stats(eng_b, score)
    print(f'[c] heard onsets on the score onset: calibrated {sa}  not calibrated {sb}', flush=True)
    return {'audio': {'song': 'twinkle (voice_v1 sung)', 'bpm': bpm, 'true_Li': Li_true, 'true_Lo': Lo_true,
                      'stream_reports': STREAM, 'cal_rt_ms': s.cals[PAIR_A]['rt_ms']},
            'calibrated': {'Li': eng_a.Li, 'Lo': eng_a.Lo, 'lat': nj['lat'], 'onsets': sa, 'run': o_a},
            'not_calibrated': {'Li': eng_b.Li, 'Lo': eng_b.Lo, 'onsets': sb, 'run': o_b}}


def part_d():
    names = [n for n in P3.RT['systems'] if n != P3.RT['default'] and n in P3.MODELS.rt]
    by_sys = 'r3_X_regtrkstsafe_v1' if 'r3_X_regtrkstsafe_v1' in names else names[0]
    test_rt = {k: v for k, v in P3.RT.items() if k != 'file'}
    test_rt['by_latency'] = {'threshold_ms': 120, 'system': by_sys}
    test_rt['file'] = 'test list in memory (panel_v3_selftest_v1; never written)'
    s = P3.Session(os.path.join(ROOT, 'd'), rt=test_rt)
    default = test_rt['default']
    check('d', 'pick_rt: no calibration -> default', P3.pick_rt(test_rt, None)[0] == default)
    check('d', 'pick_rt: 119.9 ms -> default', P3.pick_rt(test_rt, {'rt_ms': 119.9})[0] == default)
    check('d', 'pick_rt: 120 ms -> by_latency system', P3.pick_rt(test_rt, {'rt_ms': 120.0})[0] == by_sys)
    check('d', 'pick_rt: list without by_latency -> default', P3.pick_rt(P3.RT, {'rt_ms': 400.0})[0] == P3.RT['default']
          if not P3.RT.get('by_latency') else True)
    ca = cal_run(s, LC.make_test(0.160, 21)['rec'], PAIR_A, 'd')
    cb = cal_run(s, LC.make_test(0.060, 22)['rec'], PAIR_B, 'd')
    check('d', 'pairs calibrated through the page: A >= 120 ms, B < 120 ms', ca['ok'] and cb['ok'] and ca['rt_ms'] >= 120
          and cb['rt_ms'] < 120, (ca['rt_ms'], cb['rt_ms']))
    x = phone('twinkle', 14.0)
    simA, simB = {'devs': PAIR_A, 'latency': STREAM}, {'devs': PAIR_B, 'latency': STREAM}
    oa = song_run(s, {'song': 'twinkle', 'mode': 'rt'}, x, sim=simA, note='selftest v3 (d) auto A')
    check('d', "'auto' (no system given) on pair A runs the by_latency system and says why",
          oa[0]['name'] == by_sys and s.pick['auto'] == by_sys and '≥' in s.pick['why'], (oa[0]['name'], s.pick['why']))
    check('d', 'status shows the pick', s.status()['pick']['auto'] == by_sys)
    ob = song_run(s, {'song': 'twinkle', 'mode': 'rt', 'system': 'auto'}, x, sim=simB, note='selftest v3 (d) auto B')
    check('d', "'auto' on pair B runs the default", ob[0]['name'] == default and '<' in s.pick['why'],
          (ob[0]['name'], s.pick['why']))
    oe = song_run(s, {'song': 'twinkle', 'mode': 'rt', 'system': default}, x, sim=simA, note='selftest v3 (d) explicit')
    check('d', 'an explicit system on pair A is honored', oe[0]['name'] == default, oe[0]['name'])
    arms = []
    for k in range(4):
        random.seed(k)
        o = song_run(s, {'song': 'twinkle', 'mode': 'rt', 'system': 'blind'}, x, sim=simA, note=f'selftest v3 (d) blind {k}')
        arms.append((o[0]['mode'], o[0]['name']))
    allowed = {('reactive', 'r2_mix_v2'), ('rt', by_sys)}
    check('d', 'the rt blind A/B on pair A plays only reactive r2_mix_v2 or the by_latency system',
          all(a in allowed for a in arms), arms)
    return {'test_list_by_latency': test_rt['by_latency'], 'cal_A_ms': ca['rt_ms'], 'cal_B_ms': cb['rt_ms'],
            'auto_A': oa[0]['name'], 'auto_B': ob[0]['name'], 'explicit_A': oe[0]['name'], 'blind_arms': arms,
            'why_A': oa[3]['pick'], 'why_B': ob[3]['pick']}


def main():
    if os.path.exists(OUT):
        sys.exit(f'refusing to overwrite {OUT}')
    lists0 = {os.path.basename(p): sha_file(p) for p in glob.glob(os.path.join(V4, 'live_rt_systems_v*.json'))}
    pages0 = {f: sha_file(os.path.join(V4, f)) for f in ('live_panel_v2.py', 'live_panel_v2.html')}
    t_start = time.time()
    res = {'what': 'offline selftest of live_panel_v3 (no sound device; tripwire)', 'root': ROOT.replace(ABS + '/', ''),
           'written': None, 'parts': {}, 'errors': []}
    P3.MODELS.load()
    assert P3.MODELS.err is None, P3.MODELS.err
    s_b = cl = None
    for part, fn in (('a', part_a), ('b', part_b), ('c', None), ('d', part_d)):
        try:
            if part == 'b':
                s_b, cl, res['parts']['b'] = part_b()
            elif part == 'c':
                if s_b is None:
                    raise RuntimeError('part c needs part b')
                res['parts']['c'] = part_c(s_b, cl)
            else:
                res['parts'][part] = fn()
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            res['errors'].append({'part': part, 'traceback': tb})
            check(part, 'part ran without an exception', False)
    lists1 = {os.path.basename(p): sha_file(p) for p in glob.glob(os.path.join(V4, 'live_rt_systems_v*.json'))}
    pages1 = {f: sha_file(os.path.join(V4, f)) for f in ('live_panel_v2.py', 'live_panel_v2.html')}
    check('all', 'no list file written or changed', lists0 == lists1, sorted(lists1))
    check('all', 'live_panel_v2.py / .html unchanged', pages0 == pages1)
    res['checks'] = CHECKS
    res['pass'] = {p: all(c['ok'] for c in CHECKS if c['part'] == p) for p in ('a', 'b', 'c', 'd', 'all')}
    res['all_pass'] = all(c['ok'] for c in CHECKS) and not res['errors']
    res['code'] = {'live_panel_v3.py': sha_file(os.path.join(V4, 'live_panel_v3.py')),
                   'live_panel_v3.html': sha_file(os.path.join(V4, 'live_panel_v3.html')),
                   'latcal_v1.py': sha_file(os.path.join(V4, 'latcal_v1.py')),
                   'panel_v3_selftest_v1.py': sha_file(os.path.abspath(__file__)), 'lists': lists1}
    res['seconds'] = round(time.time() - t_start, 1)
    res['written'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(OUT, 'x') as fh:
        json.dump(res, fh, indent=1, default=float, ensure_ascii=False)
    print(json.dumps({'pass': res['pass'], 'all_pass': res['all_pass'], 'errors': len(res['errors']),
                      'seconds': res['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
