"""Duet benchmark v2: does Duet's note sound bad against what she actually sings, live, with a noisy ear?

A system is a module bench_systems/<name>.py with
    build(train) -> obj            train = {'bach': [...], 'folk': [...], 'pop': [...], ...}: every
                                   chord-labelled dataset present in bench/data, plus 'essen' (melody
                                   only, no chords) when present. Training songs are clean.
    obj.run(song, cfg) -> [Beat]   planner.Beat per beat, at least k, z_hat (chord Duet plays, -1 none)
                                   and h (Duet's MIDI note, -1 silent)
    obj.note_guess (optional)      dict k -> guessed pitch class of her beat k (relative to the tonic)

What the system is given at evaluation (built here, so it cannot cheat on these two points):
  * her melody as HEARD: every note independently, with probability NOISE = 10 %, is misheard as one
    of the other 11 pitch classes (uniform, nearest octave), fixed seed per song;
  * no chord labels: every beat is reduced to {'pos', 'z': -1, 'onset_pcs': []}.
Scores are computed against her TRUE melody.

Rules (checked by review):
  causal    the choice for beat k may use the heard melody only up to the decision time k - D,
            D = (cfg.L + cfg.delta) / beat length (ScoreSource.partial for the beat being heard)
  no test   tune on --split val only; the test split is run at checkpoints; every run is logged
  real time mean compute well under 1 ms per beat on this laptop (M4); us/beat is reported, and a
            system over 5000 us/beat is marked SLOW and not ranked
  streaming the system must be able to run beat by beat (no look-ahead over the song)

Scores per dataset, over her melody time from beat cfg.J on (Duet listens first):
  good      share of her melody time where Duet sings a consonant interval against her true note:
            unison/octave, minor/major third, perfect fifth, minor/major sixth (two-voice counterpoint
            convention). Silence counts as not good, so Duet must sing.
  harsh     share in sharp dissonance: minor/major second, tritone, minor/major seventh (1,2,6,10,11)
  fourth    share at a perfect fourth (5): mild, counted as not good in two voices
  silent    share where Duet is silent
  double    share at unison/octave. A system above 25 % is marked DOUBLING (echoing her is not harmony)
            and not ranked.
  four      share of her melody whose pitch class is in the chord Duet plays (stage metric)
  note      her next note guessed right (if the system exposes note_guess)
Headline = mean of 'good' over the chord-labelled datasets except twinkle.

  .venv/bin/python duet_beat/bench.py --system hmm --split val
  .venv/bin/python duet_beat/bench.py --board
"""
import argparse
import importlib
import json
import os
import random
import sys
import time
import zlib
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load  # noqa: E402
from metrics import song_counts, summarize  # noqa: E402
from model import soprano_obs  # noqa: E402
from planner import Config  # noqa: E402
from songs import twinkle  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402

BENCH = os.path.join(HERE, 'bench')
BOARD = os.path.join(BENCH, 'leaderboard_v2.jsonl')
EXTRA = ('folk', 'pop', 'pop909')                    # bench/data/<name>.json, each {'train','val','test'}
NOISE = 0.10
MAX_EVAL = 300                                       # songs per dataset per split (first N, fixed order)
HARSH = frozenset({1, 2, 6, 10, 11})
SLOW_US, DOUBLE_MAX = 5000.0, 0.25


def config():
    res = json.load(open(os.path.join(HERE, 'out', 'results.json')))
    base = dict(res['tuning']['chosen']['main'])
    base['alto'] = tuple(base['alto'])
    return Config(**base)


def _extra(name):
    p = os.path.join(BENCH, 'data', f'{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def train_sets():
    out = {'bach': load()['split']['train']}
    for name in EXTRA + ('essen',):
        x = _extra(name)
        if x:
            out[name] = x['train']
    return out


def eval_sets(split):
    out = {'bach': load()['split'][split]}
    for name in EXTRA:
        x = _extra(name)
        if x and x.get(split):
            out[name] = x[split][:MAX_EVAL]
    out['twinkle'] = [twinkle()]
    return out


def heard(song, p=NOISE):
    """What Duet is given: her melody with mishearings, and no chord labels."""
    rng = random.Random(zlib.crc32(str(song.get('id', song.get('title', ''))).encode()) ^ 5160)
    sop = []
    for on, off, m in song['sop']:
        if rng.random() < p:
            pc = (m + rng.randint(1, 11)) % 12           # one of the other 11 pitch classes
            m = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        sop.append([on, off, m])
    beats = [{'pos': b['pos'], 'z': -1, 'onset_pcs': []} for b in song['beats']]
    return dict(song, sop=sop, beats=beats)


def interval_shares(song, beats, J):
    """Time shares of her TRUE melody (from beat J) by what Duet sings against it."""
    tot = dict(good=0.0, harsh=0.0, fourth=0.0, silent=0.0, double=0.0, four_n=0.0, four_d=0.0, d=0.0)
    by_k = {b.k: b for b in beats}
    for on, off, m in song['sop']:
        for k in range(max(J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9:
                continue
            tot['d'] += ov
            b = by_k.get(k)
            if b is None or b.h < 0:
                tot['silent'] += ov
                continue
            iv = abs(m - b.h) % 12
            tot['good'] += ov * (iv in CONSONANT)
            tot['harsh'] += ov * (iv in HARSH)
            tot['fourth'] += ov * (iv == 5)
            tot['double'] += ov * (iv == 0)
            if b.z_hat >= 0:
                tot['four_d'] += ov
                tot['four_n'] += ov * ((m - song['tonic']) % 12 in CT[b.z_hat])
    return tot


def score(system, songs, cfg):
    acc = dict(good=0.0, harsh=0.0, fourth=0.0, silent=0.0, double=0.0, four_n=0.0, four_d=0.0, d=0.0)
    cs, nn, nd, beats_n, secs = [], 0, 0, 0, 0.0
    for s in songs:
        c = replace(cfg, bpm=s['bpm']) if 'bpm' in s else cfg
        h = heard(s)
        if hasattr(system, 'note_guess'):
            system.note_guess = {}
        t0 = time.perf_counter()
        beats = system.run(h, c)
        secs += time.perf_counter() - t0
        beats_n += s['K']
        cs.append(song_counts(s, beats, cfg.J))
        for key, v in interval_shares(s, beats, cfg.J).items():
            acc[key] += v
        for k, pc in (getattr(system, 'note_guess', None) or {}).items():
            mid = soprano_obs(s['sop'], k)[1]
            if k >= cfg.J and mid is not None:
                nd += 1
                nn += int(pc == (mid - s['tonic']) % 12)
    d = acc['d'] or float('nan')
    sm = summarize(cs)
    return {'good': acc['good'] / d, 'harsh': acc['harsh'] / d, 'fourth': acc['fourth'] / d,
            'silent': acc['silent'] / d, 'double': acc['double'] / d,
            'four': acc['four_n'] / acc['four_d'] if acc['four_d'] else float('nan'),
            'chord': sm['m1'], 'note': nn / nd if nd else None,
            'us_per_beat': 1e6 * secs / beats_n, 'songs': len(songs)}


def evaluate(name, split, note=''):
    mod = importlib.import_module(f'bench_systems.{name}')
    cfg = config()
    system = mod.build(train_sets())
    rows = {ds: score(system, songs, cfg) for ds, songs in eval_sets(split).items()}
    main = [ds for ds in rows if ds != 'twinkle']
    flags = []
    if max(rows[ds]['us_per_beat'] for ds in rows) > SLOW_US:
        flags.append('SLOW')
    if max(rows[ds]['double'] for ds in main) > DOUBLE_MAX:
        flags.append('DOUBLING')
    entry = {'system': name, 'split': split, 'headline': sum(rows[ds]['good'] for ds in main) / len(main),
             'datasets': main, 'flags': flags, 'rows': rows, 'note': note, 'noise': NOISE,
             'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    os.makedirs(BENCH, exist_ok=True)
    with open(BOARD, 'a') as fh:
        fh.write(json.dumps(entry, default=float) + '\n')
    return entry


def show(entry):
    fl = f"  {' '.join(entry['flags'])}" if entry['flags'] else ''
    print(f"{entry['system']}  [{entry['split']}, noise {entry['noise']:.0%}]  headline good "
          f"{entry['headline']:.3f} (mean of {', '.join(entry['datasets'])}){fl}")
    for ds, r in entry['rows'].items():
        note = f"  note {r['note']:.3f}" if r['note'] is not None else ''
        print(f"  {ds:8s} good {r['good']:.3f}  harsh {r['harsh']:.3f}  fourth {r['fourth']:.3f}  "
              f"silent {r['silent']:.3f}  double {r['double']:.3f}  four {r['four']:.3f}{note}  "
              f"{r['us_per_beat']:.0f} us/beat  ({r['songs']} songs)")


def board(split):
    if not os.path.exists(BOARD):
        print('empty')
        return
    best = {}
    for line in open(BOARD):
        e = json.loads(line)
        if e['split'] == split:
            best[e['system']] = e                         # latest run of each system
    for e in sorted(best.values(), key=lambda e: (bool(e['flags']), -e['headline'])):
        ds = '  '.join(f"{k} {v['good']:.3f}/{v['harsh']:.3f}" for k, v in e['rows'].items())
        us = max(v['us_per_beat'] for v in e['rows'].values())
        print(f"{e['headline']:.3f}  {e['system']:22s} {ds}  {us:.0f}us {' '.join(e['flags'])}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--system')
    ap.add_argument('--split', default='val', choices=('val', 'test'))
    ap.add_argument('--note', default='')
    ap.add_argument('--board', action='store_true')
    a = ap.parse_args()
    if a.board:
        board(a.split)
    else:
        show(evaluate(a.system, a.split, a.note))
