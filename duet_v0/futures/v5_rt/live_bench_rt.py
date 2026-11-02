"""live_bench_rt: the tracker-in-the-loop live benchmark (run v5rt_1101, round 3). Val split only: no test split.

LIVE_SPEC (tracker-in-the-loop live benchmark; implemented in futures/v5_rt/live_bench_rt.py with renderers in
futures/v5_rt/voice_v1.py; both frozen after phase 1, sha256 in runs_mini/v5rt_1101/LIVE_BENCH_SHA.txt):
- Purpose: score a WHOLE live engine (pitch tracker, note segmenter, onset snapping, decision code, Duet's voice) against
  GROUND TRUTH, offline: audio is fed to engine.process() 10 ms (HOP) at a time; nothing is played; the _NoSound
  tripwire is installed before any import.
- Engines: live_rt.RTDuet(system, ...) with system = bench_rt.load(spec, bench3.train_sets()) for spec
  'reactive:<v3 name>' or 'rt:<rt_systems name>' (built once per process). Song facts given to the engine are what
  Vincent sets on the page: tonic, meter M, beats K, bpm.
- Main condition: count_in = 4 (Duet's click count-in, the page's 'free' mode: the grid t0 = lead + count_in * T_b is
  known in advance); she follows the click: her true onset i is at t0 + onset_i * T_b plus human timing jitter (sd 15 ms,
  seeded). Tempo 90 bpm for every rendered song (1 beat = 0.667 s, so the ~70 ms live hearing lag is about bench_rt's
  L = 0.1).
- Songs: the 17 val Bach chorales (headline, pooled by time like a bench_rt dataset) and Twinkle (reported separately).
- Voices (deterministic, seeded from crc32 of the song id): 'clean' = render_clean (tracker sanity check); 'sung' =
  singer.render-based voice plus onset scoops on about half of the non-legato notes (start 50-150 cents flat, reach the
  pitch in 60-120 ms), per-note detune (sd 15 cents) and room noise at about -40 dB; 'sung_low' = 'sung' one octave
  lower (a low voice). Phone files, no rendering: 'phone_twinkle' = twinkle_C_80bpm_countin.wav with live_rt_eval_v2's
  setup (no engine count-in, beat 0 from her first note, Li 0, Lo 10 ms; truth = songs.twinkle() at 80 bpm, her first
  onset at 4.0 s); 'phone_v11' = test_her_v11_countin.wav with the page's settings (A major, 122.45 bpm, 28 beats,
  first-note sync): no truth, live_rt_eval_v2's proxies (answered, sung, attacks, cons_heard, changes, on_time_cons,
  fresh_heard count, harsh_heard share).
- Latency: 'wired' = Li 12 ms, Lo 10 ms, told to the engine; 'bt' = true Li 150 ms, Lo 180 ms while the engine is told
  Core Audio's 63 / 42 ms (AirPods stress test, report only). Her audio reaches the tracker the true Li late; Duet's
  note is heard the true Lo after it is written.
- Truth metrics (her TRUE notes from beat J = 4 on labelled beats z >= 0; time to beats with the true grid; exact
  integration over Duet's actual sounding intervals read from the voice, like bench_rt): rt_nd_live (hit share of her
  singing time: h >= 0, |m - h| mod 12 in theory.CONSONANT minus {0}, (h - tonic) mod 12 in CT[z]; silence = miss),
  harsh ({1, 2, 6, 10, 11}), double, first_100ms hit and harsh shares, on_time_right (the note sounding at her true
  onset still sounds 200 ms later and is a hit at the onset), late_entry, fresh_per100 (Duet STARTS a new pitch that is
  harsh against her true note at that moment; an attack less than 60 ms before one of her onsets is judged against that
  note), wrong_att_per100 (attacks that are not hits), stumbles_per100 (an attack from 60 ms before to 150 ms after her
  onset that is replaced by another pitch or by silence within 250 ms), blips (notes < 30 ms).
- Tracker report per run: each true note matched to the tracker's answered notes: heard - true pitch in semitones,
  missed, split, detect lag (ms after the true onset), snapped-onset error in beats.
- CLI: .venv/bin/python futures/v5_rt/live_bench_rt.py <spec> --voice clean|sung|sung_low|phone_twinkle|phone_v11
  [--lat wired|bt] [--songs N quick subset] --out <new json under runs_mini/v5rt_1101/live_bench/>; prints one summary
  line; one engine on all 17 val chorales must fit in 600 s.

How this file reads LIVE_SPEC where it leaves room (decided in phase 1, frozen with the file):
- Her true notes (voice_v1.true_notes, the same for all three rendered voices): onset t0_her + onset * T_b + N(0, 15 ms)
  (rng from crc32(id)); a legato note (the next one starts at its offset in the score) ends at the next jittered onset,
  any other note at its nominal offset. Her true pitch is the pitch rendered ('sung_low': the score - 12). The 'clean'
  renderer leaves audio.render_her's 30 ms gap at each note's end; that gap still counts as her singing time (the truth
  is the same for every voice). Only Bach chorales and Twinkle are ever rendered.
- Latency: the engine (RTDuet, count_in 4, lead 0.5) is told Li_told / Lo_told. She sings to the click she HEARS, which
  sounds Lo_true - Lo_told after the engine's grid, so her grid is t0_her = t0 + (Lo_true - Lo_told) (= t0 for 'wired').
  Her audio enters the engine Li_true late (leading zeros). Duet's notes, written at output sample i0, are heard at
  i0 / SR + Lo_true. 'wired': Li 12 ms, Lo 10 ms, told = true. Phone files use live_rt_eval_v2's setup (Li 0, Lo 10 ms,
  told = true, no count-in, beat 0 from her first note); --lat bt is refused for them.
- Song facts to the engine: tonic, M, K, bpm only, so RTDuet starts the system with pos = k % M (the page has no pickup
  setting). 14 of the 17 val chorales start with a pickup (pos[0] != 0): there the bar position the systems get is not
  the song's (bench_rt gives them the true pos). Her beat 0 = the song's beat 0 = the engine's beat 0.
- Duet's sounding notes come from the voice's own note log (live_rt_eval_v2.run / voice_notes): [i0 / SR + Lo_true,
  (i0 + int(dur * SR)) / SR + Lo_true), notes RTDuet dropped before they started never sounded; the 50 ms release is
  not counted. Blips (< 30 ms) DID sound: they are in the step function and in the attack metrics, and counted apart.
- Times to beats with her true grid: x_beats = (x - t0_her) / T_b. Counted time = her true notes from beat J = 4 on
  grid beats with z >= 0, integrated exactly against Duet's step function (bench_rt.score_song's loop with
  bench_rt.pieces / hit / HARSH). first_100ms = counted time in [true onset, true onset + 100 ms), cut at her offset.
- Per-note shares (on_time_right, changes, late_entry; the denominator of every *_per100) are over her notes whose
  NOMINAL (score) onset is >= J on a labelled beat k0 = int(onset); "a hit at the onset" uses z[k0] (jitter never moves
  a note onto the previous beat's chord). Duet's note "at her onset" / "200 ms later" = bench_rt.at at her TRUE onset and
  at true onset + 200 ms (also when her note is shorter than 200 ms, as bench_rt judges at onset + L).
- fresh / wrong_att: live_rt_eval_v2.attacks (a new pitch, from silence or from another pitch) and her_at (her note
  with on - 60 ms <= x < off, the latest); judged at beat floor(max(attack beat, her note's nominal onset beat)) as
  live_rt_eval_v2.truth does; attacks while she rests or before beat J or on unlabelled beats are not counted.
- stumbles: an attack within [onset - 60 ms, onset + 150 ms] of one of her counted notes whose pitch stops sounding
  (another pitch or silence) less than 250 ms after the attack. Also stumbles_own: the same, replaced before her next
  onset - 60 ms (a correction inside her note, not an answer to her next note).
- Headline = the 17 val chorales pooled (time and notes summed, like a bench_rt dataset); Twinkle (rendered at 90 bpm
  with the same voice and latency) is reported separately. --songs N = the first N val chorales (bench3 order).
- Tracker report: the answered notes are RTDuet's heard notes (segment start c0 in engine time, her time =
  c0 + Li_told - Li_true). An answered note belongs to the latest true note with onset <= its start + 80 ms if it
  starts before that note's offset + 30 ms, else it is 'extra'. Per true note: missed (none), split (>= 2), the first
  one's heard - true pitch, detect lag = her time of the newest audio the engine had when it answered ((block + 1) *
  HOP / SR - Li_true) minus her true onset (input latency not included; add Li_true + Lo_true for when Duet's answer is
  heard), raw onset error = segment start - true onset (ms), snapped-onset error = the onset the system got (engine
  beats) - the nominal onset.
- phone_twinkle truth: her notes are the full nominal spans 4.0 + [on, off) * 0.75 s (no jitter known, no 30 ms cut);
  live_rt_eval_v2.truth's own 10 ms-sampled numbers are reported next to the exact ones ('sampled').
- Determinism: all draws are seeded (voice_v1); r2_mix_v2 sets torch to 1 thread; 'digest' = sha256 of every
  deterministic field (no timings), equal for equal runs.

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python futures/v5_rt/live_bench_rt.py rt:r2_copyhmbold_v1 \
      --voice sung --lat wired --out runs_mini/v5rt_1101/live_bench/<new name>.json
"""
import sys
import types


class _NoSound(types.ModuleType):
    """futures/v4_selfiter/live_rt_eval_v2.py's tripwire, copied so it is in place before any other import: private /
    dunder lookups get the usual AttributeError; any public sound-device call raises at once."""

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('live_bench_rt is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import bisect  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.join(ABS, 'futures', 'v5_rt')
V4 = os.path.join(ABS, 'futures', 'v4_selfiter')
OUT_ROOT = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'live_bench')
for _p in (V4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402
import bench_rt as BR  # noqa: E402  (bench3, theory and the systems / rt_systems paths)
import live_rt_eval_v2 as EV  # noqa: E402  (installs its own identical tripwire; run, voice_notes, attacks, her_at)
import live_rt as LR  # noqa: E402
import voice_v1 as VV  # noqa: E402
from live_bench3 import HOP, SR  # noqa: E402
from songs import twinkle  # noqa: E402

bench3 = BR.bench3
SPLIT = 'val'
J = BR.J
HARSH = BR.HARSH
BPM = 90.0
COUNT_IN = 4
LAT = {'wired': {'Li': 0.012, 'Lo': 0.010, 'Li_told': 0.012, 'Lo_told': 0.010},
       'bt': {'Li': 0.150, 'Lo': 0.180, 'Li_told': 0.063, 'Lo_told': 0.042}}
PHONE_LAT = {'Li': 0.0, 'Lo': EV.LO, 'Li_told': 0.0, 'Lo_told': EV.LO}
W_AFTER = 0.20          # s: on_time_right / changes / late_entry look 200 ms after her true onset
FIRST = 0.10            # s: first_100ms shares
W_PRE = EV.W_PRE        # s: an attack this close before her onset is judged against that note (0.06)
ST_PRE, ST_POST, ST_WIN = 0.06, 0.15, 0.25   # s: stumble window around her onset, and 'replaced within'
BLIP = EV.BLIP          # s: 0.03
MATCH_PRE, MATCH_POST = 0.08, 0.03           # s: tracker report matching tolerances
VOICES = VV.VOICES + ('phone_twinkle', 'phone_v11')
ACC_KEYS = ('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes', 'otr', 'changes', 'late',
            'att', 'fresh', 'wrong', 'att_rest', 'stumbles', 'stumbles_own', 'blips', 'blips_her')


def quiet(s):
    return None


def _js(o):
    if hasattr(o, 'item'):
        return o.item()
    return str(o)


def file_sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_system(spec):
    kind = spec.partition(':')[0]
    if kind not in ('reactive', 'rt'):
        raise SystemExit(f"bad spec {spec!r}: use reactive:<v3 system> | rt:<rt_systems name>")
    return BR.load(spec, bench3.train_sets())


# ---------------------------------------------------------------------------------------------------- one engine run
def make_engine(system, name, facts, count_in):
    return LR.RTDuet(system, name, facts['tonic'], facts['M'], facts['K'], facts['bpm'], count_in=count_in, log=quiet)


def run_engine(eng, x, lat):
    """Feed x 10 ms at a time (live_rt_eval_v2.run: the engine is told Li_told / Lo_told); Duet's notes as heard
    (Lo_true after they were written): sounding notes >= 30 ms and blips."""
    track, log, _, secs = EV.run(eng, x, Lo=lat['Lo_told'], Li=lat['Li_told'])
    notes, blips = EV.voice_notes(log, lat['Lo'])
    return track, notes, blips, secs


def step_fn(notes):
    """Duet's sounding notes [(a, e, midi)] sorted by start -> (T, H): from T[j] on Duet sounds H[j]; -1 = silent."""
    T, H = [], []
    for j, (a, e, m) in enumerate(notes):
        if e <= a:
            continue
        if T and T[-1] == a:
            H[-1] = int(m)
        else:
            T.append(a)
            H.append(int(m))
        nxt = notes[j + 1][0] if j + 1 < len(notes) else math.inf
        if e < nxt:
            T.append(e)
            H.append(-1)
    return T, H


# ---------------------------------------------------------------------------------------------------- truth metrics
def new_acc():
    return dict.fromkeys(ACC_KEYS, 0.0)


def score(her, duet, z, ton, t0h, T_b, acc):
    """Add one song to acc. her = true notes [(on_s, off_s, midi, nominal_on, nominal_off)]; duet = Duet's sounding
    notes [(a_s, e_s, midi)] (blips included). Returns the per-note rows (all of her notes)."""
    b = lambda x: (x - t0h) / T_b  # noqa: E731
    Ts, H = step_fn(duet)
    Tb = [b(x) for x in Ts]
    fb, wb = FIRST / T_b, W_AFTER / T_b
    counted = []
    rows = []
    for on_s, off_s, m, non, noff in her:
        on, off = b(on_s), b(off_s)
        k0 = int(math.floor(non + 1e-9))
        c = non >= J and k0 < len(z) and z[k0] >= 0
        counted.append(c)
        v0, v1 = BR.at(Tb, H, on), BR.at(Tb, H, on + wb)
        if c:
            acc['notes'] += 1
            acc['otr'] += v0 == v1 and BR.hit(v0, m, ton, z[k0])
            acc['changes'] += v0 != v1 and v0 >= 0 and v1 >= 0
            acc['late'] += v0 < 0 and v1 >= 0
        rows.append([non, round(on, 4), m, int(c), v0, v1, int(bool(c and v0 == v1 and BR.hit(v0, m, ton, z[k0])))])
        e1 = on + fb
        for k in range(max(J, int(math.floor(on))), int(math.floor(off)) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            acc['d'] += ov
            for x0, x1, h in BR.pieces(Tb, H, max(on, k), min(off, k + 1)):
                w, f = x1 - x0, min(x1, e1) - x0
                g = BR.hit(h, m, ton, z[k])
                bad = h >= 0 and abs(m - h) % 12 in HARSH
                acc['hit'] += w * g
                acc['harsh'] += w * bad
                acc['double'] += w * (h >= 0 and abs(m - h) % 12 == 0)
                if f > 0:
                    acc['d_first'] += f
                    acc['hit_first'] += f * g
                    acc['harsh_first'] += f * bad
    # attacks: fresh clashes and wrong attacks (live_rt_eval_v2's rules), stumbles
    hv = [(on_s, off_s, m, non) for on_s, off_s, m, non, _ in her]
    ons = [n[0] for n in her]
    clashes = []
    for x, h in EV.attacks(duet):
        nt = EV.her_at(x, hv)
        if nt is None:
            acc['att_rest'] += 1
        else:
            xb = max(b(x), nt[3])
            kb = int(math.floor(xb))
            if not (xb < J or kb >= len(z) or z[kb] < 0):
                acc['att'] += 1
                iv = abs(nt[2] - h) % 12
                if iv in HARSH:
                    acc['fresh'] += 1
                    clashes.append([round(b(x), 3), nt[2], h, iv])
                acc['wrong'] += not BR.hit(h, nt[2], ton, z[kb])
        j = bisect.bisect_right(ons, x + ST_PRE) - 1          # her latest onset with on - 60 ms <= x
        if j >= 0 and counted[j] and x <= ons[j] + ST_POST:
            q = bisect.bisect_right(Ts, x) - 1
            while q + 1 < len(Ts) and H[q + 1] == h:
                q += 1
            t_change = Ts[q + 1] if q + 1 < len(Ts) else math.inf
            if t_change - x < ST_WIN:
                acc['stumbles'] += 1
                nxt = ons[j + 1] if j + 1 < len(ons) else math.inf
                acc['stumbles_own'] += t_change < nxt - ST_PRE
    for a, e, h in duet:
        if e - a < BLIP:
            acc['blips'] += 1
            nt = EV.her_at(a, hv)
            if nt is not None:
                k = ons.index(nt[0])
                acc['blips_her'] += counted[k]
    return rows, clashes


def metrics(acc):
    r = lambda a, b: a / b if b > 0 else None  # noqa: E731
    n = acc['notes']
    p = lambda a: 100.0 * a / n if n > 0 else None  # noqa: E731
    return {'rt_nd_live': r(acc['hit'], acc['d']), 'harsh': r(acc['harsh'], acc['d']),
            'double': r(acc['double'], acc['d']), 'first_100ms_hit': r(acc['hit_first'], acc['d_first']),
            'first_100ms_harsh': r(acc['harsh_first'], acc['d_first']), 'on_time_right': r(acc['otr'], n),
            'changes': r(acc['changes'], n), 'late_entry': r(acc['late'], n),
            'fresh_per100': p(acc['fresh']), 'fresh': int(acc['fresh']), 'wrong_att_per100': p(acc['wrong']),
            'attacks_per100': p(acc['att']), 'attacks_in_rest': int(acc['att_rest']),
            'stumbles_per100': p(acc['stumbles']), 'stumbles_own_per100': p(acc['stumbles_own']),
            'blips': int(acc['blips']), 'blips_on_her': int(acc['blips_her']), 'notes': int(n),
            'counted_beats': acc['d']}


# ---------------------------------------------------------------------------------------------------- tracker report
def tracker_rows(eng, her, lat, t0h, T_b):
    """One row per true note: [nominal onset, nominal duration (beats), true midi, interval from her previous true note
    (None for the first), answered notes, heard midi of the first (None = missed), detect lag ms, raw onset error ms,
    snapped-onset error beats]; and the number of answered notes that match no true note."""
    starts = {i: c0 for c0, i in eng.by_start.items()}
    dec = {}
    for d in eng.decisions:
        if d['kind'] == 'hear' and d['i'] not in dec:
            blk = int(round((d['t'] - lat['Lo_told']) * SR / HOP))
            dec[d['i']] = (blk + 1) * HOP / SR - lat['Li']
    ons = [n[0] for n in her]
    per = [[] for _ in her]
    extra = 0
    for i, (on_b, _, midi) in enumerate(eng.heard_notes):
        tau = starts[i] + lat['Li_told'] - lat['Li']
        k = bisect.bisect_right(ons, tau + MATCH_PRE) - 1
        if k >= 0 and tau <= her[k][1] + MATCH_POST:
            per[k].append((tau, int(midi), float(on_b), dec.get(i)))
        else:
            extra += 1
    rows = []
    for k, (on_s, off_s, m, non, noff) in enumerate(her):
        iv = None if k == 0 else int(m - her[k - 1][2])
        got = per[k]
        if got:
            tau, hm, onb, td = got[0]
            rows.append([non, round(noff - non, 4), int(m), iv, len(got), hm,
                         None if td is None else round(1000 * (td - on_s), 1), round(1000 * (tau - on_s), 1),
                         round(onb - non, 4)])
        else:
            rows.append([non, round(noff - non, 4), int(m), iv, 0, None, None, None, None])
    return rows, extra


def _q(a, p):
    return float(np.percentile(a, p)) if len(a) else None


def tracker_summary(rows, extra=0):
    """Pooled tracker report over tracker_rows rows."""
    n = len(rows)
    ans = [r for r in rows if r[4] > 0]
    err = Counter(int(r[5] - r[2]) for r in ans)
    lag = [r[6] for r in ans if r[6] is not None]
    raw = [r[7] for r in ans]
    snap = [r[8] for r in ans]
    f = lambda a, b: a / b if b else None  # noqa: E731
    return {'true_notes': n, 'answered': len(ans), 'missed': f(n - len(ans), n), 'split': f(sum(r[4] >= 2 for r in rows), n),
            'extra_answered': int(extra),
            'pc_right': f(sum((r[5] - r[2]) % 12 == 0 for r in ans), n),
            'pitch_right': f(err.get(0, 0), n),
            'pitch_right_of_answered': f(err.get(0, 0), len(ans)),
            'octave_err_of_answered': f(sum(v for d, v in err.items() if d != 0 and d % 12 == 0), len(ans)),
            'semitone_err_of_answered': f(err.get(1, 0) + err.get(-1, 0), len(ans)),
            'heard_minus_true': {str(d): v for d, v in sorted(err.items())},
            'detect_ms': {'median': _q(lag, 50), 'mean': float(np.mean(lag)) if lag else None, 'p10': _q(lag, 10),
                          'p90': _q(lag, 90)},
            'onset_err_ms': {'median': _q(raw, 50), 'mean_abs': float(np.mean(np.abs(raw))) if raw else None,
                             'p10': _q(raw, 10), 'p90': _q(raw, 90)},
            'snapped_err_beats': {'exact': f(sum(abs(s) < 1e-6 for s in snap), len(snap)),
                                  'mean_abs': float(np.mean(np.abs(snap))) if snap else None,
                                  'hist': {str(k): v for k, v in sorted(Counter(round(s, 2) for s in snap).items())
                                           if v >= 1}}}


# ---------------------------------------------------------------------------------------------------- songs
def engine_stats(eng, secs, n_blocks):
    s = eng.summary()
    return {'us': list(eng.us_plan) + list(eng.us_hear), 'decision_counts': s['decision_counts'],
            'late_plan_decisions': len(eng.late), 'notes_answered': s['notes_answered'],
            'detect_ms_median': s['detect_ms_median'], 'block_ms': list(eng.proc_ms), 'secs': secs,
            'audio_s': n_blocks * HOP / SR}


def rendered_song(system, name, song, voice, lat, bpm=BPM):
    """One rendered song through RTDuet with the count-in: (score acc, result dict, tracker rows, extra, stats)."""
    T_b = 60.0 / bpm
    facts = {'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'bpm': bpm}
    eng = make_engine(system, name, facts, COUNT_IN)
    t0h = eng.t0 + (lat['Lo'] - lat['Lo_told'])
    y, her, vf = VV.render(song, voice, t0h, T_b)
    x = np.concatenate([np.zeros(int(round(lat['Li'] * SR))), y.astype(np.float64), np.zeros(SR)])
    track, notes, blips, secs = run_engine(eng, x, lat)
    duet = sorted(notes + blips)
    z = [bt['z'] for bt in song['beats']]
    acc = new_acc()
    rows, clashes = score(her, duet, z, song['tonic'], t0h, T_b, acc)
    trows, extra = tracker_rows(eng, her, lat, t0h, T_b)
    st = engine_stats(eng, secs, len(track))
    b = lambda v: round((v - t0h) / T_b, 4)  # noqa: E731
    res = {'id': song['id'], 'K': song['K'], 'M': song['M'], 'tonic': song['tonic'], 'pickup': int(song['beats'][0]['pos']),
           'voice_facts': vf, 'metrics': metrics(acc), 'tracker': tracker_summary(trows, extra),
           'decision_counts': st['decision_counts'], 'late_plan_decisions': st['late_plan_decisions'],
           'notes_answered': st['notes_answered'],
           'duet': [[b(a), b(e), int(m)] for a, e, m in duet], 'her': rows, 'tracker_rows': trows,
           'fresh_clashes': clashes}
    return acc, res, trows, extra, st


def phone_run(system, name, which):
    cfg = EV.TW if which == 'phone_twinkle' else EV.V11
    x = EV.read(cfg['file']).astype(np.float64)
    facts = {'tonic': cfg['tonic'], 'M': cfg['M'], 'K': cfg['K'], 'bpm': cfg['bpm']}
    eng = make_engine(system, name, facts, 0)
    track, notes, blips, secs = run_engine(eng, x, PHONE_LAT)
    st = engine_stats(eng, secs, len(track))
    prox = EV.heard_proxy(eng, track, notes, blips, EV.LO)
    out = {'file': cfg['file'], 'facts': facts, 'proxies': prox, 't0': eng.t0,
           'decision_counts': st['decision_counts'], 'late_plan_decisions': st['late_plan_decisions'],
           'notes_answered': st['notes_answered'],
           'answered': [[e['onset_beats'], e['her_heard'], e['duet'], e['action']] for e in eng.events],
           'duet_s': [[round(a, 4), round(e, 4), int(m)] for a, e, m in sorted(notes + blips)]}
    if which == 'phone_twinkle':
        song = twinkle()
        T_b, t0h = 60.0 / cfg['bpm'], cfg['beat1']
        her = [(t0h + on * T_b, t0h + off * T_b, int(m), float(on), float(off))
               for on, off, m in sorted(song['sop'], key=lambda n: n[0])]
        acc = new_acc()
        rows, clashes = score(her, sorted(notes + blips), [bt['z'] for bt in song['beats']], song['tonic'], t0h, T_b, acc)
        trows, extra = tracker_rows(eng, her, PHONE_LAT, t0h, T_b)
        samp = EV.truth(track, notes, blips, EV.LO, t0h, T_b)
        out.update({'metrics': metrics(acc), 'her': rows, 'fresh_clashes': clashes, 'tracker': tracker_summary(trows, extra),
                    'tracker_rows': trows,
                    'sampled': {k: v for k, v in samp.items() if k not in ('per_note', 'clashes')}})
    return out, st


# ---------------------------------------------------------------------------------------------------- a run
def pooled_stats(sts):
    us = [u for s in sts for u in s['us']]
    blk = [u for s in sts for u in s['block_ms']]
    secs = sum(s['secs'] for s in sts)
    audio = sum(s['audio_s'] for s in sts)
    return {'us_mean': float(np.mean(us)) if us else None, 'us_max': float(np.max(us)) if us else None,
            'us_p99': _q(us, 99), 'calls': len(us), 'block_ms_median': _q(blk, 50), 'block_ms_max': float(np.max(blk)),
            'engine_secs': round(secs, 2), 'realtime_factor': round(secs / audio, 4) if audio else None}


def evaluate(spec, voice, lat_name='wired', n_songs=None, system=None):
    c0 = time.perf_counter()
    if voice not in VOICES:
        raise SystemExit(f'voice must be one of {VOICES}')
    if voice.startswith('phone') and lat_name != 'wired':
        raise SystemExit('phone files have their own latency setup (Li 0, Lo 10 ms); --lat bt is for rendered voices')
    if system is None:
        system = load_system(spec)
    built = time.perf_counter() - c0
    det = {'spec': spec, 'voice': voice, 'lat': lat_name, 'split': SPLIT}
    if voice in VV.VOICES:
        lat = LAT[lat_name]
        songs = bench3.eval_sets(SPLIT)['bach']
        if n_songs:
            songs = songs[:n_songs]
        acc, rows, trows, extra, sts = new_acc(), [], [], 0, []
        for s in songs:
            a, r, tr, ex, st = rendered_song(system, spec, s, voice, lat)
            for k in ACC_KEYS:
                acc[k] += a[k]
            rows.append(r)
            trows += tr
            extra += ex
            sts.append(st)
        a, tw, tr_tw, ex_tw, st_tw = rendered_song(system, spec, twinkle(), voice, lat)
        dc = Counter()
        for r in rows:
            dc.update(r['decision_counts'])
        det.update({'latency': lat, 'bpm': BPM, 'count_in': COUNT_IN, 'songs_n': len(songs),
                    'headline': metrics(acc), 'twinkle': tw['metrics'],
                    'tracker': {'bach': tracker_summary(trows, extra), 'twinkle': tw['tracker']},
                    'decision_counts': dict(sorted(dc.items())),
                    'late_plan_decisions': sum(r['late_plan_decisions'] for r in rows),
                    'songs': rows, 'twinkle_song': tw})
        timing = pooled_stats(sts + [st_tw])
    else:
        out, st = phone_run(system, spec, voice)
        det.update({'latency': PHONE_LAT, 'phone': out})
        timing = pooled_stats([st])
    det['settings'] = {'W_AFTER': W_AFTER, 'FIRST': FIRST, 'W_PRE': W_PRE, 'stumble': [ST_PRE, ST_POST, ST_WIN],
                       'BLIP': BLIP, 'match': [MATCH_PRE, MATCH_POST], 'J': J, 'jitter_s': VV.JITTER,
                       'scoop_p': VV.SCOOP_P, 'detune_st': VV.DETUNE_ST, 'noise_db': VV.NOISE_DB}
    digest = hashlib.sha256(json.dumps(det, sort_keys=True, default=_js).encode()).hexdigest()
    res = dict(det)
    res.update({'digest': digest, 'timing': dict(timing, build_s=round(built, 2),
                                                  total_s=round(time.perf_counter() - c0, 2)),
                'code': {'live_bench_rt.py': file_sha(os.path.abspath(__file__)),
                         'voice_v1.py': file_sha(os.path.join(HERE, 'voice_v1.py')),
                         'bench_rt.py': file_sha(os.path.join(HERE, 'bench_rt.py'))},
                'note': 'val only (17 Bach chorales) + Twinkle; systems tuned on val; this bundle has no test split',
                'time': time.strftime('%Y-%m-%d %H:%M:%S')})
    return res


def line(res):
    f = lambda x, d=4: 'n/a' if x is None else f'{x:.{d}f}'  # noqa: E731
    head = f"{res['spec']} {res['voice']} {res['lat']}"
    tm = res['timing']
    tail = f"| digest {res['digest'][:12]} | {tm['total_s']:.0f} s, us/decision {f(tm['us_mean'], 0)}"
    if res['voice'] == 'phone_v11':
        p = res['phone']['proxies']
        return (f"{head}: answered {p['answered']} sung {p['sung']} attacks {p['attacks']} cons_heard {p['cons_heard']} "
                f"changes {p['changes']} on_time_cons {p['on_time_cons']} fresh_heard {p['fresh_heard']} "
                f"harsh_heard {p['harsh_heard']:.4f} blips {p['blips']} {tail}")
    if res['voice'] == 'phone_twinkle':
        m, s, t = res['phone']['metrics'], res['phone']['sampled'], res['phone']['tracker']
        p = res['phone']['proxies']
        return (f"{head}: rt_nd_live {f(m['rt_nd_live'])} harsh {f(m['harsh'])} fresh/100 {f(m['fresh_per100'], 2)} "
                f"stumbles/100 {f(m['stumbles_per100'], 2)} on_time_right {f(m['on_time_right'])} wrong_att/100 "
                f"{f(m['wrong_att_per100'], 1)} | sampled fit {s['fit']:.3f} harsh {s['harsh']:.3f} fresh {s['fresh']} | "
                f"proxies fresh_heard {p['fresh_heard']} harsh_heard {p['harsh_heard']:.4f} | tracker pc_right "
                f"{f(t['pc_right'], 3)} missed {f(t['missed'], 3)} {tail}")
    m, tw, t = res['headline'], res['twinkle'], res['tracker']['bach']
    return (f"{head}: rt_nd_live {f(m['rt_nd_live'])} harsh {f(m['harsh'])} fresh/100 {f(m['fresh_per100'], 2)} "
            f"stumbles/100 {f(m['stumbles_per100'], 2)} on_time_right {f(m['on_time_right'])} wrong_att/100 "
            f"{f(m['wrong_att_per100'], 1)} late {f(m['late_entry'])} | twinkle rt_nd_live {f(tw['rt_nd_live'])} "
            f"fresh/100 {f(tw['fresh_per100'], 2)} | tracker pc_right {f(t['pc_right'], 3)} missed {f(t['missed'], 3)} "
            f"split {f(t['split'], 3)} detect {f(t['detect_ms']['median'], 0)} ms {tail}")


def main():
    ap = argparse.ArgumentParser(description='live_bench_rt: tracker-in-the-loop live benchmark (LIVE_SPEC = docstring)')
    ap.add_argument('spec', help='reactive:<v3 system> | rt:<rt_systems name>')
    ap.add_argument('--voice', required=True, choices=VOICES)
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
        json.dump(res, fh, indent=1, default=_js)
    print(line(res), flush=True)


if __name__ == '__main__':
    main()
