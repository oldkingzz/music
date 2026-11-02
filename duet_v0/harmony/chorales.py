"""Extract soprano-onset SATB slices from the Bach chorales in the music21 corpus.

For every soprano note (ties merged) we record the pitch that the alto, tenor and bass
are sounding at the soprano onset, the pitch classes of the full four-voice chord there,
and whether the soprano note carries a fermata (phrase end).

The result is cached as JSON (harmony/data/bach_satb.json, ~2 MB) so training does not
re-parse MusicXML. Run:  .venv/bin/python harmony/chorales.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'data', 'bach_satb.json')
VOICES = ('Soprano', 'Alto', 'Tenor', 'Bass')


def _events(part, top=True):
    """[(on_q, off_q, midi, fermata)] for the notes of one part (ties merged, rests dropped)."""
    out = []
    for n in part.stripTies().flatten().notes:
        ql = float(n.quarterLength)
        if ql <= 0:                                   # grace notes
            continue
        ps = [p.midi for p in n.pitches]
        midi = max(ps) if top else min(ps)
        ferm = any(type(e).__name__ == 'Fermata' for e in n.expressions)
        out.append((float(n.offset), float(n.offset) + ql, int(midi), bool(ferm)))
    out.sort()
    return out


def _at(events, q):
    """MIDI pitch sounding at quarter offset q, or None (rest)."""
    for on, off, m, _ in events:
        if on <= q + 1e-6 < off:
            return m
        if on > q + 1e-6:
            break
    return None


def extract_one(score):
    parts = {p.id: p for p in score.parts}
    if not all(v in parts for v in VOICES):
        if len(score.parts) != 4:
            return None
        parts = dict(zip(VOICES, score.parts))
    ev = {v: _events(parts[v], top=(v != 'Bass')) for v in VOICES}
    if len(ev['Soprano']) < 8:
        return None
    k = score.analyze('key')
    rows = []
    for on, off, m, ferm in ev['Soprano']:
        a, t, b = (_at(ev[v], on) for v in VOICES[1:])
        chord = sorted({x % 12 for x in (m, a, t, b) if x is not None})
        rows.append([on, off - on, m, int(ferm), a, t, b, chord])
    return {'tonic': int(k.tonic.pitchClass), 'mode': k.mode,
            'key_corr': float(k.correlationCoefficient), 'notes': rows}


def build(limit=None, verbose=True):
    from music21 import corpus
    files = list(corpus.chorales.Iterator(numberingSystem='riemenschneider', returnType='filename'))
    if limit:
        files = files[:limit]
    data, skipped = [], []
    seen = set()
    t0 = time.time()
    for i, f in enumerate(files):
        if f in seen:                                # the iterator maps several Riemenschneider
            skipped.append((f, 'duplicate file'))    # numbers to the same file; keep it once
            continue
        seen.add(f)
        try:
            s = corpus.parse(f)
            rec = extract_one(s)
        except Exception as e:                       # noqa: BLE001 - report and continue
            skipped.append((f, repr(e)))
            continue
        if rec is None:
            skipped.append((f, 'not four SATB parts'))
            continue
        title = s.metadata.title if s.metadata is not None else None
        rec.update({'id': f, 'riemenschneider': i + 1, 'title': title})
        data.append(rec)
        if verbose and (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(files)} parsed, {time.time() - t0:.0f} s', flush=True)
    return data, skipped


def load():
    with open(CACHE) as fh:
        return json.load(fh)


def to_notes(rec, sec_per_q=0.6, breath=0.2):
    """Soprano line as Duet rehearsal notes [{'on','off','midi','kind'}] (seconds).

    A fermata becomes a breath: the note is shortened by `breath` seconds, which is how
    offline_notes() sees a phrase end in a real recording (a gap before the next note).
    """
    out = []
    for on, dur, m, ferm, *_ in rec['notes']:
        off = (on + dur) * sec_per_q - (breath if ferm else 0.0)
        out.append({'on': on * sec_per_q, 'off': max(off, on * sec_per_q + 0.08), 'midi': int(m),
                    'kind': 'attack'})
    return out


if __name__ == '__main__':
    data, skipped = build()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, 'w') as fh:
        json.dump({'chorales': data, 'skipped': skipped,
                   'source': 'music21 corpus, chorales.Iterator(numberingSystem="riemenschneider")'}, fh)
    print(f'kept {len(data)} chorales, skipped {len(skipped)}', file=sys.stderr)
    for f, why in skipped:
        print('  skipped', f, why, file=sys.stderr)
