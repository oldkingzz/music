"""Offline block-by-block comparison of the live Duet engines on the phone files (run v5rt_1101).

NO sound device is opened and nothing is played: each file is fed to engine.process() 10 ms at a time, as
live_predict.run_file does, and Duet's notes are read back from the engine's own voice. These numbers come from two
files (Twinkle and one real singer), not from the val set; the bench_rt systems were tuned on val (no test split).

Engines (each on the same audio):
  reactive     live_bench3.Bench3Duet(r2_mix_v2): one step() per heard note, sung at once
  predict      live_predict.PredictDuet(r2_mix_v2) with its class defaults (mimic, consonant, W_CUR 0.5): the
               look-ahead Vincent heard on 2026-11-01
  rt r0        live_rt.RTDuet(r0_predict_live_v1): that look-ahead ported to bench_rt (W_CUR 0)
  rt champion  live_rt.RTDuet(r2_copyhmbold_v1): the verified bench_rt champion
Files:
  twinkle  futures/live_test/phone/twinkle_C_80bpm_countin.wav, C major, 80 bpm; her true onsets 4.0 s + beat * 0.75 s,
           true notes and chords from songs.twinkle()
  v11      futures/live_test/phone/test_her_v11_countin.wav, a real singer, A major, 122.45 bpm, 28 beats (the page's
           settings); no ground truth
  clicks   a rendered Twinkle (audio.render_her 'vibrato', 80 bpm, mic latency Li 12 ms) through RTDuet(r2_copyhmbold_v1)
           with count_in = 4 (click mode), plus reactive and RTDuet without a count-in on the same audio for reference
Timeline as live_predict.run_file: Li = 0 for the phone files, Lo = 10 ms. The note Duet's voice sounds at the end of each
10 ms block is taken as heard 20 ms + Lo after that block started (live_predict.score_twinkle's convention).

Metrics with ground truth (Twinkle; her singing time from beat J = 4 sampled every 10 ms as score_twinkle samples it):
  fit            live_predict.score_twinkle: share of that time Duet's sounding note is consonant with her TRUE note, not
                 her unison/octave, and a tone of the TRUE chord (late or silent = miss)
  harsh          share of that time Duet sounds a 2nd, tritone or 7th (bench3.HARSH) against her true note
  harsh100       the same, only inside the first 100 ms of each of her notes
  fresh          Duet STARTS a new pitch (from silence or from another pitch; exact start taken from the voice) that is
                 harsh against her true note at that moment; an attack less than 60 ms before one of her onsets is
                 judged against that note (a planned note is written 25 ms early); count and per 100 of her notes from
                 beat 4. wrong_att = attacks that are not a hit (consonant, not unison/octave, chord tone), per 100 notes
  on_time_right  share of her notes from beat 4 where the note Duet sounds at her true onset is still the note it sounds
                 200 ms later and is a hit at the onset (bench_rt's on_time_right with 200 ms for L)
  changes        share where Duet sounds a note at her onset and a different note 200 ms later
  late           share where Duet is silent at her onset and sounds 200 ms later
  blips          notes that sounded less than 30 ms (cut right after they were written; their 50 ms release is heard)
Without ground truth (v11; also computed on Twinkle to show how they track the truth there): her notes = the notes the
segmenter closed (the same for every engine), her pitch = the heard pitch, all of her notes (no beat-4 rule):
  answered, sung (answers after which Duet sounds), attacks (new pitches Duet starts), cons_heard (answers after which
  Duet's note is consonant with her heard pitch, not unison/octave), changes, on_time_cons (the note at her detected
  onset is kept for 200 ms and is consonant with her heard pitch), fresh_heard (attacks harsh against her heard pitch),
  harsh_heard (share of her detected singing time Duet is harsh against her heard pitch).

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python futures/v4_selfiter/live_rt_eval.py \
      --out runs_mini/v5rt_1101/live/<new name>.json [--smoke]
"""
import argparse
import json
import os
import sys
import time
import types


class _NoSound(types.ModuleType):
    """Tripwire: this evaluation is offline; any use of a sound-device library raises instead of touching a device."""

    def __getattr__(self, name):
        raise RuntimeError('live_rt_eval is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_bench3 as LB  # noqa: E402  (puts duet_beat, live, panel and v3 on sys.path)
import live_predict as LP  # noqa: E402
import live_rt as LR  # noqa: E402
from live_bench3 import HOP, SR  # noqa: E402
import bench3  # noqa: E402
from audio import render_her  # noqa: E402
from songs import twinkle  # noqa: E402
from theory import CT  # noqa: E402

ABS = LB.ABS
OUT_ROOT = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
PHONE = os.path.join(ABS, 'futures', 'live_test', 'phone')
GOOD = frozenset({3, 4, 7, 8, 9})          # consonant, not unison/octave (rescore_nd's music_nd)
HARSH = bench3.HARSH                       # 2nds, tritone, 7ths
J = 4
LO = 0.010                                 # output latency, as live_predict.run_file
W_AFTER = 0.20                             # s after her onset: the note Duet has settled on (the reactive answer is in)
W_PRE = 0.06                               # s: an attack this close before her onset is judged against that note
BLIP = 0.03                                # s: a note that sounds less than this is a blip
TW = {'file': 'twinkle_C_80bpm_countin.wav', 'tonic': 0, 'M': 4, 'K': 48, 'bpm': 80.0, 'beat1': 4.0}
V11 = {'file': 'test_her_v11_countin.wav', 'tonic': 9, 'M': 4, 'K': 28, 'bpm': 122.45}
ENGINES = (('reactive', 'reactive r2_mix_v2', 'r2_mix_v2'),
           ('predict', 'PredictDuet r2_mix_v2', 'r2_mix_v2'),
           ('rt', 'RTDuet r0_predict_live_v1', 'r0_predict_live_v1'),
           ('rt', 'RTDuet r2_copyhmbold_v1', 'r2_copyhmbold_v1'))


def quiet(s):
    return None


def make(kind, sysm, name, song, count_in=0):
    args = (sysm, name, song['tonic'], song['M'], song['K'], song['bpm'])
    if kind == 'reactive':
        return LB.Bench3Duet(*args, log=quiet)
    if kind == 'predict':
        return LP.PredictDuet(*args, log=quiet)
    return LR.RTDuet(*args, log=quiet, count_in=count_in)


def read(name):
    x, sr = sf.read(os.path.join(PHONE, name), dtype='float32', always_2d=True)
    assert sr == SR, (name, sr)
    return x.mean(1)


def run(eng, x, Lo=LO, Li=0.0):
    """Feed x block by block. Returns Duet's sounding note per block (live_predict.sounding), every note the voice was
    asked to sing (its _Note objects, read after the run), Duet's output signal and the seconds it took."""
    eng.Lo, eng.Li = Lo, Li
    log = []
    orig = eng.voice.schedule

    def schedule(start, dur, midi, tag=None):
        i0 = orig(start, dur, midi, tag)
        log.append(eng.voice.notes[-1])
        return i0

    eng.voice.schedule = schedule
    track, out = [], []
    c0 = time.perf_counter()
    for b in range(len(x) // HOP):
        out.append(np.asarray(eng.process(x[b * HOP:(b + 1) * HOP]), np.float64))
        track.append(LP.sounding(eng, b))
    return np.array(track, dtype=int), log, np.concatenate(out), time.perf_counter() - c0


def voice_notes(log, Lo):
    """Duet's notes as heard (start s, end s, midi), Lo after they were written; notes RTDuet dropped before they
    started (level 0) never sounded and are left out; notes shorter than BLIP are blips."""
    notes, blips = [], []
    for nt in log:
        if nt.level <= 0:
            continue
        n = int(nt.dur * SR)
        row = (nt.i0 / SR + Lo, (nt.i0 + n) / SR + Lo, int(round(nt.midi)))
        (notes if n >= BLIP * SR else blips).append(row)
    return sorted(notes), sorted(blips)


def attacks(notes):
    """(time, midi) where Duet starts a new pitch: from silence, or the note sounding just before it differs."""
    out = []
    for j, (a, e, m) in enumerate(notes):
        prev = None
        for p in notes[:j]:
            if p[0] < a <= p[1]:
                prev = p
        if prev is None or prev[2] != m:
            out.append((a, m))
    return out


def her_at(x, her):
    """Her note (on s, off s, midi, onset beat) an event at time x belongs to: the latest with on - W_PRE <= x < off."""
    best = None
    for n in her:
        if n[0] - W_PRE <= x < n[1] and (best is None or n[0] > best[0]):
            best = n
    return best


def at_time(track, t_out, x):
    i = int(np.searchsorted(t_out, x))
    return int(track[min(i, len(track) - 1)])


def hit(h, m, ton, zk):
    return h >= 0 and abs(m - h) % 12 in GOOD and (h - ton) % 12 in CT[zk]


def truth(track, notes, blips, Lo, beat1, T_b):
    """Twinkle metrics against her TRUE notes and chords (songs.twinkle(), onsets beat1 + beat * T_b)."""
    song = twinkle()
    z = [b['z'] for b in song['beats']]
    ton = song['tonic']
    true = sorted(song['sop'])
    fit, wrong100 = LP.score_twinkle(track, Lo=Lo, beat1=beat1, T_b=T_b, J=J)
    t_out = (np.arange(len(track)) + 2) * HOP / SR + Lo
    tot = bad = f100 = bad100 = 0
    for on, off, m in true:
        a, e = beat1 + on * T_b, beat1 + off * T_b - 0.03
        for idx in np.where((t_out >= a) & (t_out < e))[0]:
            k = int(np.floor((t_out[idx] - beat1) / T_b))
            if k < J or k >= len(z):
                continue
            h = int(track[idx])
            x = h >= 0 and abs(m - h) % 12 in HARSH
            tot += 1
            bad += x
            if t_out[idx] - a < 0.10:
                f100 += 1
                bad100 += x
    per_note, n_notes, otr, chg, late = [], 0, 0, 0, 0
    for on, off, m in true:
        k0 = int(on)
        if on < J or k0 >= len(z) or z[k0] < 0:
            continue
        a, e = beat1 + on * T_b, beat1 + off * T_b - 0.03
        v0, v1 = at_time(track, t_out, a), at_time(track, t_out, min(a + W_AFTER, e))
        ok = v0 == v1 and hit(v0, m, ton, z[k0])
        n_notes += 1
        otr += ok
        chg += v0 != v1 and v0 >= 0 and v1 >= 0
        late += v0 < 0 and v1 >= 0
        per_note.append({'beat': on, 'her': m, 'at_onset': v0, 'after_200ms': v1, 'on_time_right': bool(ok)})
    her = [(beat1 + on * T_b, beat1 + off * T_b, m, on) for on, off, m in true]
    fresh = miss = n_att = in_rest = 0
    clashes = []
    for x, h in attacks(notes):
        nt = her_at(x, her)
        if nt is None:
            in_rest += 1
            continue
        xb = max((x - beat1) / T_b, nt[3])
        kb = int(np.floor(xb))
        if xb < J or kb >= len(z) or z[kb] < 0:
            continue
        n_att += 1
        iv = abs(nt[2] - h) % 12
        if iv in HARSH:
            fresh += 1
            clashes.append({'t': round(x, 3), 'beat': round((x - beat1) / T_b, 2), 'her': nt[2], 'duet': h, 'iv': iv})
        miss += not hit(h, nt[2], ton, z[kb])
    n_bl = bl_clash = 0
    for a, e, h in blips:
        nt = her_at(a, her)
        if nt is None or max((a - beat1) / T_b, nt[3]) < J:
            continue
        n_bl += 1
        bl_clash += abs(nt[2] - h) % 12 in HARSH
    return {'fit': fit, 'wrong_first100': wrong100, 'harsh': bad / tot, 'harsh100': bad100 / max(f100, 1),
            'fresh': fresh, 'fresh_per100': 100.0 * fresh / n_notes, 'wrong_att_per100': 100.0 * miss / n_notes,
            'attacks_counted': n_att, 'attacks_in_rest': in_rest,
            'on_time_right': otr / n_notes, 'changes': chg / n_notes, 'late': late / n_notes, 'notes': n_notes,
            'blips': n_bl, 'blip_clashes': bl_clash, 'clashes': clashes, 'per_note': per_note}


def heard_proxy(eng, track, notes, blips, Lo):
    """No ground truth: her notes = the notes the segmenter closed, her pitch = the heard pitch."""
    t_out = (np.arange(len(track)) + 2) * HOP / SR + Lo
    her = [(float(on), float(off), int(m), None) for on, off, m in eng.notes]
    ev = eng.events
    chg = otc = late = 0
    for on, off, m, _ in her:
        v0, v1 = at_time(track, t_out, on), at_time(track, t_out, min(on + W_AFTER, off))
        chg += v0 != v1 and v0 >= 0 and v1 >= 0
        otc += v0 == v1 and v0 >= 0 and abs(m - v0) % 12 in GOOD
        late += v0 < 0 and v1 >= 0
    att = attacks(notes)
    fresh = n_att = 0
    for x, h in att:
        nt = her_at(x, her)
        if nt is None:
            continue
        n_att += 1
        fresh += abs(nt[2] - h) % 12 in HARSH
    tot = bad = 0
    for on, off, m, _ in her:
        for i in np.where((t_out >= on) & (t_out < off))[0]:
            tot += 1
            bad += track[i] >= 0 and abs(m - int(track[i])) % 12 in HARSH
    nh = max(len(her), 1)
    return {'answered': len(ev), 'sung': int(sum(e['duet'] >= 0 for e in ev)),
            'cons_heard': int(sum(e['duet'] >= 0 and abs(e['her_heard'] - e['duet']) % 12 in GOOD for e in ev)),
            'her_notes': len(her), 'attacks': len(att), 'attacks_on_her': n_att,
            'changes': chg, 'on_time_cons': otc, 'late': late, 'fresh_heard': fresh,
            'fresh_heard_per100': 100.0 * fresh / nh, 'harsh_heard': bad / max(tot, 1), 'blips': len(blips)}


def eng_stats(eng, secs, n_blocks):
    s = eng.summary()
    keep = ('notes_answered', 'sung', 'detect_ms_median', 'block_ms_median', 'block_ms_max', 't0', 'mode', 'kept',
            'corrected', 'late_start', 'decision_counts', 'late_decisions', 'us_plan_mean', 'us_hear_mean', 'us_mean',
            'us_max', 'count_in')
    out = {k: s[k] for k in keep if k in s}
    out['run_seconds'] = round(secs, 2)
    out['realtime_factor'] = round(secs / (n_blocks * HOP / SR), 4)
    out['late_plan_decisions'] = len(eng.late)
    return out, s


def one(kind, label, sysm, name, song, x, *, beat1=None, Li=0.0, count_in=0, T_b=None):
    eng = make(kind, sysm, name, song, count_in=count_in)
    track, log, y, secs = run(eng, x, Li=Li)
    notes, blips = voice_notes(log, LO)
    res = {'engine': label, 'system': name}
    if beat1 is not None:
        res['truth'] = truth(track, notes, blips, LO, beat1, T_b or 60.0 / song['bpm'])
    res['heard'] = heard_proxy(eng, track, notes, blips, LO)
    res['stats'], full = eng_stats(eng, secs, len(track))
    res['events'] = full.get('events')
    if kind == 'rt':
        res['plans'] = full.get('plans')
        res['decisions'] = full.get('decisions')
    elif kind == 'predict':
        res['plans'] = full.get('plans')
    return res, eng, track, y


def click_check(eng, y, Lo):
    """Where RTDuet wrote its count-in clicks, how loud they are, and that nothing else sounds before her first note."""
    L = int(0.03 * SR)
    mask = np.zeros(len(y), bool)
    clicks = []
    for k in range(-eng.count_in, 0):
        i0 = int(round((eng.t0 + k * eng.T_b - Lo) * SR))
        seg = y[max(i0, 0):i0 + L]
        mask[max(i0, 0):i0 + L] = True
        clicks.append({'beat': k, 'heard_s': round(i0 / SR + Lo, 3), 'peak': round(float(np.max(np.abs(seg))), 3)})
    ev0 = eng.decisions[0] if eng.decisions else None
    pre_end = int(round((eng.t0 - Lo - 0.03) * SR))
    other = np.abs(y[:pre_end][~mask[:pre_end]])
    first_hear = next((d for d in eng.decisions if d['kind'] == 'hear'), None)
    return {'t0': eng.t0, 'clicks': clicks, 'max_other_before_beat0': round(float(other.max()) if len(other) else 0.0, 4),
            'first_decision': ev0, 'first_hear': first_hear,
            'plans_before_first_hear': sum(1 for d in eng.decisions if d['kind'] == 'plan'
                                           and (first_hear is None or d['t'] < first_hear['t']))}


def f3(x):
    return '  n/a' if x is None else f'{x:.3f}'


def table(res):
    lines = []
    tw = res['twinkle']
    lines.append('TWINKLE phone file (C major, 80 bpm, truth = songs.twinkle(); her %d notes from beat 4; 10 ms sampling)'
                 % next(iter(tw.values()))['truth']['notes'])
    lines.append(f"{'engine':27s} {'fit':>5s} {'harsh':>6s} {'harsh100':>8s} {'fresh n (/100)':>14s} "
                 f"{'wrong att/100':>13s} {'on_time_right':>13s} {'changes':>7s} {'late':>5s} {'blips':>5s} "
                 f"{'answered':>8s}")
    for label, r in tw.items():
        t = r['truth']
        lines.append(f"{label:27s} {t['fit']:5.3f} {t['harsh']:6.3f} {t['harsh100']:8.3f} "
                     f"{t['fresh']:5d} ({t['fresh_per100']:5.1f}) {t['wrong_att_per100']:13.1f} "
                     f"{t['on_time_right']:13.3f} {t['changes']:7.3f} {t['late']:5.3f} {t['blips']:5d} "
                     f"{r['heard']['answered']:8d}")
    for key, title in (('v11', 'V11 phone file (real singer, A major, 122.45 bpm, 28 beats; NO ground truth: counts, '
                                'and proxies against the notes the tracker found)'),
                       ('twinkle', 'Same proxies on the Twinkle phone file (to compare with its truth above)')):
        lines.append('')
        lines.append(title)
        lines.append(f"{'engine':27s} {'answered':>8s} {'sung':>4s} {'attacks':>7s} {'cons_heard':>10s} "
                     f"{'changes':>7s} {'on_time_cons':>12s} {'fresh_heard n (/100)':>20s} {'harsh_heard':>11s} "
                     f"{'blips':>5s}")
        for label, r in res[key].items():
            h = r['heard']
            lines.append(f"{label:27s} {h['answered']:8d} {h['sung']:4d} {h['attacks']:7d} {h['cons_heard']:10d} "
                         f"{h['changes']:7d} {h['on_time_cons']:12d} {h['fresh_heard']:11d} ({h['fresh_heard_per100']:5.1f})"
                         f" {h['harsh_heard']:11.3f} {h['blips']:5d}")
    if 'clicks' in res:
        c = res['clicks']
        ck = c['check']
        lines.append('')
        lines.append('CLICK MODE: rendered Twinkle (render_her vibrato, 80 bpm, Li 12 ms), RTDuet(r2_copyhmbold_v1) '
                     'count_in = 4; the two reference rows take beat 0 from her first note')
        lines.append('  clicks heard at ' + ', '.join(f"{k['heard_s']:.2f} s (peak {k['peak']:.2f})" for k in ck['clicks'])
                     + f"; loudest other output before beat 0: {ck['max_other_before_beat0']:.4f}; "
                     f"plan points before her first note: {ck['plans_before_first_hear']}")
        lines.append(f"{'engine':46s} {'fit':>5s} {'harsh':>6s} {'harsh100':>8s} {'fresh n (/100)':>14s} "
                     f"{'on_time_right':>13s} {'changes':>7s} {'late':>5s} {'late plans':>10s}")
        for label, r in c['runs'].items():
            t = r['truth']
            lines.append(f"{label:46s} {t['fit']:5.3f} {t['harsh']:6.3f} {t['harsh100']:8.3f} "
                         f"{t['fresh']:5d} ({t['fresh_per100']:5.1f}) {t['on_time_right']:13.3f} {t['changes']:7.3f} "
                         f"{t['late']:5.3f} {r['stats']['late_plan_decisions']:10d}")
    return '\n'.join(lines)


def _js(o):
    if hasattr(o, 'item'):
        return o.item()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description='offline live-engine comparison on the phone files (no sound device)')
    ap.add_argument('--out', required=True, help='new .json under runs_mini/v5rt_1101/')
    ap.add_argument('--smoke', action='store_true', help='only RTDuet(r2_copyhmbold_v1) on Twinkle, and click mode')
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(OUT_ROOT + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit(f'--out must be a new .json file under {OUT_ROOT}/')
    c0 = time.perf_counter()
    train = bench3.train_sets()
    need = {name for _, _, name in ENGINES} if not a.smoke else {'r2_copyhmbold_v1', 'r2_mix_v2'}
    systems = {}
    for name in sorted(need):
        systems[name] = LB.build(name, train) if name == 'r2_mix_v2' else LR.rt_build(name, train)
    engines = ENGINES if not a.smoke else (ENGINES[0], ENGINES[3])
    print(f'built {sorted(systems)} in {time.perf_counter() - c0:.1f} s', flush=True)
    res = {'split_note': 'phone files only (not val); the systems were tuned on val, this bundle has no test split',
           'twinkle': {}, 'v11': {}, 'settings': {'Lo': LO, 'W_AFTER': W_AFTER, 'W_PRE': W_PRE, 'BLIP': BLIP, 'J': J,
                                                   'twinkle': TW, 'v11': V11}}
    x_tw = read(TW['file'])
    x_11 = read(V11['file'])
    tracks = {}
    for kind, label, name in engines:
        r, eng, track, _ = one(kind, label, systems[name], name, TW, x_tw, beat1=TW['beat1'])
        res['twinkle'][label] = r
        tracks[label] = (track, [d for d in eng.decisions] if kind == 'rt' else None)
        t = r['truth']
        print(f"twinkle {label:27s} fit {t['fit']:.3f} harsh {t['harsh']:.3f} harsh100 {t['harsh100']:.3f} "
              f"fresh {t['fresh']} ({t['fresh_per100']:.1f}/100) on_time_right {t['on_time_right']:.3f} "
              f"changes {t['changes']:.3f} late {t['late']:.3f} blips {t['blips']} "
              f"({r['stats']['run_seconds']:.1f} s)", flush=True)
        r, eng, track, _ = one(kind, label, systems[name], name, V11, x_11)
        res['v11'][label] = r
        h = r['heard']
        print(f"v11     {label:27s} answered {h['answered']} sung {h['sung']} attacks {h['attacks']} "
              f"cons_heard {h['cons_heard']} changes {h['changes']} on_time_cons {h['on_time_cons']} "
              f"fresh_heard {h['fresh_heard']} harsh_heard {h['harsh_heard']:.3f}", flush=True)
    # the same champion run again: the engine and system keep no state across runs (same decisions, same notes)
    label = 'RTDuet r2_copyhmbold_v1'
    r2, eng2, track2, _ = one('rt', label, systems['r2_copyhmbold_v1'], 'r2_copyhmbold_v1', TW, x_tw, beat1=TW['beat1'])
    res['repeat_check'] = {'same_track': bool(np.array_equal(track2, tracks[label][0])),
                           'same_decisions': [(d['kind'], d['t'], d['result'], d['action']) for d in eng2.decisions]
                           == [(d['kind'], d['t'], d['result'], d['action']) for d in tracks[label][1]]}
    print(f"repeat check (champion on Twinkle twice): {res['repeat_check']}", flush=True)
    # click mode: rendered Twinkle with a count-in
    song = twinkle()
    bpm, Li = 80.0, 0.012
    T_b = 60.0 / bpm
    probe = LR.RTDuet(systems['r2_copyhmbold_v1'], 'r2_copyhmbold_v1', 0, 4, 48, bpm, log=quiet, count_in=4)
    her = render_her(song, probe.t0, T_b, 'vibrato')
    x_r = np.concatenate([np.zeros(int(round(Li * SR))), her, np.zeros(SR)])
    meta = dict(TW, bpm=bpm)
    runs = {}
    r, eng, track, y = one('rt', 'RTDuet r2_copyhmbold_v1 count_in=4', systems['r2_copyhmbold_v1'],
                           'r2_copyhmbold_v1', meta, x_r, beat1=probe.t0, Li=Li, count_in=4, T_b=T_b)
    check = click_check(eng, y, LO)
    runs['RTDuet r2_copyhmbold_v1 count_in=4'] = r
    for kind, lab, name in ((ENGINES[0]), (ENGINES[3])):
        rr, _, _, _ = one(kind, 'reference: ' + lab + ' (no click)', systems[name], name, meta, x_r, beat1=probe.t0,
                          Li=Li, T_b=T_b)
        runs['reference: ' + lab + ' (no click)'] = rr
    res['clicks'] = {'check': check, 'runs': runs, 'render': {'style': 'vibrato', 'bpm': bpm, 'Li': Li, 't0': probe.t0}}
    for lab, rr in runs.items():
        t = rr['truth']
        print(f"clicks  {lab:45s} fit {t['fit']:.3f} harsh {t['harsh']:.3f} fresh {t['fresh']} "
              f"on_time_right {t['on_time_right']:.3f} late plans {rr['stats']['late_plan_decisions']}", flush=True)
    print(f"click check: {json.dumps(check['clicks'])} other before beat 0 {check['max_other_before_beat0']}", flush=True)
    res['table'] = table(res)
    res['seconds'] = round(time.perf_counter() - c0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1, default=_js)
    print('\n' + res['table'], flush=True)
    print(f'\nwrote {path} ({res["seconds"]:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
