"""Duet benchmark v3: no latency, so the question is only how good the harmony sounds.

Setting: Duet hears each of her notes the instant it starts (pitch known at the onset, with the same
10 % mishearing as bench v2), and answers with one note that it holds until her next note starts.
It still cannot hear the future: at her note i it knows the pitches and onsets of notes 0..i and
the durations of notes 0..i-1, plus the key (tonic), the meter and the beat grid. No chord labels.
The interface makes that the only information a system can get:

    systems/<name>.py:  build(train) -> obj      train = {'bach': [...], 'folk': [...], ...} clean
                                                 songs WITH chord labels (beats[k]['z'])
                        obj.start(meta)           meta = {'tonic', 'M', 'pos': bar position per beat, 'K'}
                        obj.step(i, onset, midi, prev_dur) -> Duet's MIDI note (or -1 = silent)

Scores, against her TRUE melody and the song's TRUE chords, over her singing time from beat 4 on:
  cons    Duet's note is consonant with hers (unison/octave, 3rds, 5th, 6ths)
  fit     Duet's note is a tone of the song's real chord on that beat (beats with a chord label)
  music   HEADLINE = consonant AND chord tone: it agrees with her and with the song's harmony
  harsh   minor/major 2nd, tritone, 7ths;  inkey  Duet's note is in the major scale
  double  unison/octave with her (> 25 % = DOUBLING, not ranked: echoing is not harmony)
  leap    mean jump of Duet's line in semitones;  big = share of jumps over 7 semitones
  par     share of moves in parallel 5ths or octaves (both voices move the same way into a 5th/8ve
          from the same interval): the classic voice-leading fault
  ROUGH flag when big > 10 % or par > 5 %.  SLOW flag above 2000 us per note.
Headline = mean of 'music' over bach, folk, pop, pop909 (val: up to 300 songs each).

  .venv/bin/python futures/v3_nolatency/bench3.py --system third_below
  .venv/bin/python futures/v3_nolatency/bench3.py --board
"""
import argparse
import importlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DUET = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(DUET, 'duet_beat'))
sys.path.insert(0, HERE)
import bench as B2  # noqa: E402
from theory import CONSONANT, CT, MAJOR  # noqa: E402

BOARD = os.path.join(HERE, 'leaderboard_v3.jsonl')
HARSH = frozenset({1, 2, 6, 10, 11})
J = 4                                     # scoring starts at beat 4 (Duet may listen first)
SLOW_US = 2000.0


def train_sets():
    return {k: v for k, v in B2.train_sets().items() if k != 'essen'}


def eval_sets(split):
    out = B2.eval_sets(split)
    out.pop('twinkle', None)
    return out


def run_song(system, song):
    """Feed the HEARD notes one by one; return Duet's note per her note index."""
    h = B2.heard(song)
    notes = sorted(h['sop'], key=lambda n: n[0])
    system.start({'tonic': song['tonic'], 'M': song['M'], 'K': song['K'],
                  'pos': [b['pos'] for b in song['beats']]})
    out, prev = [], None
    for i, (on, off, m) in enumerate(notes):
        out.append(int(system.step(i, float(on), int(m), None if prev is None else float(prev[1] - prev[0]))))
        prev = (on, off)
    return out


def song_stats(song, hs):
    true = sorted(song['sop'], key=lambda n: n[0])
    t = song['tonic']
    z = [b['z'] for b in song['beats']]
    a = dict(d=0.0, cons=0.0, harsh=0.0, double=0.0, inkey=0.0, fit_d=0.0, fit=0.0, music_d=0.0, music=0.0,
             moves=0, leap=0.0, big=0, par=0, pairs=0)
    for i, (on, off, m) in enumerate(true):
        h = hs[i]
        for k in range(max(J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z):
                continue
            a['d'] += ov
            if h < 0:
                if z[k] >= 0:
                    a['music_d'] += ov
                continue
            iv = abs(m - h) % 12
            c = iv in CONSONANT
            a['cons'] += ov * c
            a['harsh'] += ov * (iv in HARSH)
            a['double'] += ov * (iv == 0)
            a['inkey'] += ov * ((h - t) % 12 in MAJOR)
            if z[k] >= 0:
                f = (h - t) % 12 in CT[z[k]]
                a['fit_d'] += ov
                a['fit'] += ov * f
                a['music_d'] += ov
                a['music'] += ov * (c and f)
        if i >= 1 and hs[i] >= 0 and hs[i - 1] >= 0 and on >= J:
            a['pairs'] += 1
            dh = hs[i] - hs[i - 1]
            if dh != 0:
                a['moves'] += 1
                a['leap'] += abs(dh)
                a['big'] += abs(dh) > 7
            dm = m - true[i - 1][2]
            i0, i1 = abs(true[i - 1][2] - hs[i - 1]) % 12, abs(m - hs[i]) % 12
            if dm != 0 and dh != 0 and (dm > 0) == (dh > 0) and i1 in (0, 7) and i0 == i1:
                a['par'] += 1
    return a


def score(system, songs):
    tot, secs, n = {}, 0.0, 0
    for s in songs:
        t0 = time.perf_counter()
        hs = run_song(system, s)
        secs += time.perf_counter() - t0
        n += len(hs)
        for k, v in song_stats(s, hs).items():
            tot[k] = tot.get(k, 0) + v
    d = tot['d'] or float('nan')
    return {'music': tot['music'] / (tot['music_d'] or float('nan')), 'cons': tot['cons'] / d,
            'fit': tot['fit'] / (tot['fit_d'] or float('nan')), 'harsh': tot['harsh'] / d,
            'double': tot['double'] / d, 'inkey': tot['inkey'] / d,
            'leap': tot['leap'] / max(tot['moves'], 1), 'big': tot['big'] / max(tot['moves'], 1),
            'par': tot['par'] / max(tot['pairs'], 1), 'us_per_note': 1e6 * secs / max(n, 1), 'songs': len(songs)}


def evaluate(name, split='val', note=''):
    mod = importlib.import_module(f'systems.{name}')
    system = mod.build(train_sets())
    rows = {ds: score(system, songs) for ds, songs in eval_sets(split).items()}
    flags = []
    if max(r['us_per_note'] for r in rows.values()) > SLOW_US:
        flags.append('SLOW')
    if max(r['double'] for r in rows.values()) > 0.25:
        flags.append('DOUBLING')
    if max(r['big'] for r in rows.values()) > 0.10 or max(r['par'] for r in rows.values()) > 0.05:
        flags.append('ROUGH')
    e = {'system': name, 'split': split, 'headline': sum(r['music'] for r in rows.values()) / len(rows),
         'flags': flags, 'rows': rows, 'note': note, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(BOARD, 'a') as fh:
        fh.write(json.dumps(e, default=float) + '\n')
    return e


def show(e):
    print(f"{e['system']} [{e['split']}] music {e['headline']:.3f} {' '.join(e['flags'])}")
    for ds, r in e['rows'].items():
        print(f"  {ds:7s} music {r['music']:.3f} cons {r['cons']:.3f} fit {r['fit']:.3f} harsh {r['harsh']:.3f} "
              f"double {r['double']:.3f} inkey {r['inkey']:.3f} leap {r['leap']:.1f} big {r['big']:.3f} "
              f"par {r['par']:.3f} {r['us_per_note']:.0f}us ({r['songs']})")


def board(split='val'):
    best = {}
    for line in open(BOARD):
        e = json.loads(line)
        if e['split'] == split and (e['system'] not in best or e['headline'] >= best[e['system']]['headline']):
            best[e['system']] = e
    for e in sorted(best.values(), key=lambda e: (bool(e['flags']), -e['headline'])):
        r = e['rows']
        print(f"{e['headline']:.3f}  {e['system']:24s} " + '  '.join(
            f"{k} {v['music']:.3f}" for k, v in r.items()) + f"  {' '.join(e['flags'])}  {e['note']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--system')
    ap.add_argument('--split', default='val', choices=('val', 'test'))
    ap.add_argument('--note', default='')
    ap.add_argument('--board', action='store_true')
    a = ap.parse_args()
    board(a.split) if a.board else show(evaluate(a.system, a.split, a.note))
