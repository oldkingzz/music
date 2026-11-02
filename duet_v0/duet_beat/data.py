"""Bach chorales on a beat grid: the training and test data for duet_beat (spec section 8, E16).

For every major-key chorale in the repository's tune-grouped split (harmony/data/split.json) we
keep, beat by beat:
  * the soprano (her melody) as notes in beat units, so any tempo can be applied later;
  * the beat's position in the bar (E7);
  * a chord label z* = the diatonic triad that best explains all four voices in that beat (Q1:
    chords outside the key are folded into the closest triad), and how well it fits;
  * the pitch classes Bach actually sounds at the beat onset (to compare with the repo's metric).
All pitch classes are relative to the tonic. Run once:  .venv/bin/python duet_beat/data.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from theory import CT, N_CHORDS, ROOT, TIE_ORDER  # noqa: E402

CACHE = os.path.join(HERE, 'data', 'bach_beats.json')
SPLIT = os.path.join(REPO, 'harmony', 'data', 'split.json')
KEYS = os.path.join(REPO, 'harmony', 'data', 'bach_satb.json')
VOICES = ('Soprano', 'Alto', 'Tenor', 'Bass')
BASS_BONUS = 0.3        # a triad whose root is in the bass at the beat onset wins ties and near-ties


def _events(part, top=True):
    """[(on_q, off_q, midi)] for one voice: ties merged, rests and grace notes dropped."""
    out = []
    for n in part.stripTies().flatten().notes:
        ql = float(n.quarterLength)
        if ql <= 0:
            continue
        ps = [p.midi for p in n.pitches]
        out.append((float(n.offset), float(n.offset) + ql, int(max(ps) if top else min(ps))))
    out.sort()
    return out


def label_beat(weights, bass_pc):
    """Best diatonic triad for one beat. weights[pc] = how long pc sounds in the beat (all voices)."""
    total = float(weights.sum())
    if total <= 0:
        return -1, 0.0
    score = np.array([sum(weights[pc] for pc in CT[z]) + (BASS_BONUS if bass_pc == ROOT[z] else 0.0)
                      for z in range(N_CHORDS)])
    best = max(TIE_ORDER, key=lambda z: (round(score[z], 9), -TIE_ORDER.index(z)))
    return int(best), float(sum(weights[pc] for pc in CT[best]) / total)


def extract(score, tonic):
    parts = {p.id: p for p in score.parts}
    if not all(v in parts for v in VOICES):
        if len(score.parts) != 4:
            return None
        parts = dict(zip(VOICES, score.parts))
    ev = {v: _events(parts[v], top=(v != 'Bass')) for v in VOICES}
    if len(ev['Soprano']) < 8:
        return None
    ts = score.recurse().getElementsByClass('TimeSignature').first()
    beat_q = float(ts.beatDuration.quarterLength) if ts is not None else 1.0
    M = int(ts.beatCount) if ts is not None else 4
    meas = list(parts['Soprano'].getElementsByClass('Measure'))
    m0 = meas[0]
    if m0.number == 0 and len(meas) > 1:              # pickup bar: the grid's downbeats start at bar 1
        origin = float(meas[1].offset)
    else:
        origin = float(m0.offset) - float(getattr(m0, 'paddingLeft', 0.0) or 0.0)
    first_on = min(e[0][0] for e in ev.values() if e)
    end = max(e[-1][1] for e in ev.values() if e)
    q0 = origin - beat_q * math.ceil((origin - first_on) / beat_q - 1e-9)
    K = int(math.ceil((end - q0) / beat_q - 1e-9))

    beats = []
    for k in range(K):
        a, b = q0 + k * beat_q, q0 + (k + 1) * beat_q
        w = np.zeros(12)
        onset = set()
        for v in VOICES:
            for on, off, m in ev[v]:
                ov = min(off, b) - max(on, a)
                if ov > 1e-9:
                    w[(m - tonic) % 12] += ov / beat_q
                if on <= a + 1e-6 < off:
                    onset.add((m - tonic) % 12)
        bass = [m for on, off, m in ev['Bass'] if on <= a + 1e-6 < off]
        if not bass:
            bass = [m for v in VOICES for on, off, m in ev[v] if on <= a + 1e-6 < off]
        bass_pc = (min(bass) - tonic) % 12 if bass else None
        z, fit = label_beat(w, bass_pc)
        pos = int(round((a - origin) / beat_q)) % M
        beats.append({'pos': pos, 'z': z, 'fit': round(fit, 4), 'onset_pcs': sorted(onset)})
    sop = [[round((on - q0) / beat_q, 6), round((off - q0) / beat_q, 6), m] for on, off, m in ev['Soprano']]
    return {'M': M, 'beat_q': beat_q, 'K': K, 'beats': beats, 'sop': sop}


def build(verbose=True):
    from music21 import corpus
    with open(SPLIT) as fh:
        split = json.load(fh)
    with open(KEYS) as fh:
        keys = {r['id']: r for r in json.load(fh)['chorales']}
    out, skipped = {'train': [], 'val': [], 'test': []}, []
    t0 = time.time()
    n = 0
    for part in ('train', 'val', 'test'):
        for cid in split[part]:
            kr = keys.get(cid)
            if kr is None or kr['mode'] != 'major':
                skipped.append((cid, 'minor' if kr else 'no key'))
                continue
            try:
                rec = extract(corpus.parse(cid), kr['tonic'])
            except Exception as e:                       # noqa: BLE001 - report and continue
                skipped.append((cid, repr(e)))
                continue
            if rec is None:
                skipped.append((cid, 'not four SATB parts'))
                continue
            rec.update({'id': cid, 'title': kr.get('title'), 'tonic': kr['tonic']})
            out[part].append(rec)
            n += 1
            if verbose and n % 25 == 0:
                print(f'{n} major chorales parsed, {time.time() - t0:.0f} s', flush=True)
    return out, skipped


def load():
    with open(CACHE) as fh:
        return json.load(fh)


if __name__ == '__main__':
    data, skipped = build()
    with open(CACHE, 'w') as fh:
        json.dump({'split': data, 'skipped': skipped,
                   'source': 'music21 corpus via harmony/data/split.json (tune-grouped split), major keys only'}, fh)
    from collections import Counter
    for part, recs in data.items():
        beats = [b for r in recs for b in r['beats']]
        zs = Counter(b['z'] for b in beats)
        fits = [b['fit'] for b in beats if b['z'] >= 0]
        print(part, len(recs), 'chorales', len(beats), 'beats', dict(sorted(zs.items())),
              'mean fit', round(float(np.mean(fits)), 3),
              'meters', dict(Counter((r['M'], r['beat_q']) for r in recs)))
    print('skipped', len(skipped), Counter(why if why in ('minor', 'no key') else 'error' for _, why in skipped))
