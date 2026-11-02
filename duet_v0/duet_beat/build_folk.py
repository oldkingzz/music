"""Chord-labelled folk tunes on a beat grid, in the same song format as data.load()['split'] songs.

Sources are ABC files (chord symbols in quotes, e.g. "G" "D7" "Em", parsed by music21 into
harmony.ChordSymbol objects) and MusicXML lead sheets with <harmony> elements. For every tune in a
MAJOR key whose chord symbols cover most beats we keep, beat by beat:
  * her melody = the top note of the tune, in beat units (pickups handled as in data.extract:
    the grid's downbeats sit on the bar lines);
  * z = the diatonic triad (theory.CT, relative to the tonic) that best matches the chord symbol
    sounding over most of the beat: 7ths and extensions reduced to root/3rd/5th, non-diatonic
    chords folded to the triad sharing the most tones (ties: the triad on the same root, then
    theory.TIE_ORDER), -1 when nothing shares >= 2 tones or no chord is sounding yet;
  * onset_pcs = sorted(CT[z]) (as songs.twinkle), fit = 1.0 for labelled beats, 0.0 otherwise.
Meters use music21's beat: 6/8 -> beat_q 1.5, M 2; 2/2 -> beat_q 2.0, M 2. Repeats are NOT
expanded (same as the Bach chorales in data.py).

FINDING (2026-10-29): the local music21 corpus has almost no chord-labelled folk tunes. The quoted
tokens in ryansMammoth are fingerings ("4", "0", "1") and directions ("Segno", "D.C."), not chords;
oneills1850 / airdsAirs / miscFolk likewise. nottingham-dataset ships only reelsa-c.abc (2 tunes,
one in A minor); leadSheet has 2 songs. So the default run yields a 3-tune PREVIEW, written to
bench/data/folk_preview.json, a name bench.py does not read. Point --src at a local copy of the
full Nottingham ABC set (or any chord-symbol ABC/MusicXML) and pass --out bench/data/folk.json to
make the dataset live.

  ../.venv/bin/python build_folk.py                  # build the preview from the local corpus
  ../.venv/bin/python build_folk.py --src DIR_OR_FILE ... --out bench/data/folk.json
  ../.venv/bin/python build_folk.py --selftest       # pickup / 6/8 / label-folding checks
  ../.venv/bin/python build_folk.py --check FILE     # hmm baseline on FILE's train+val, no leaderboard
"""
import argparse
import glob
import json
import math
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from theory import CHORDS, CT, MAJOR, PC_NAMES, ROOT, TIE_ORDER, chord_name  # noqa: E402

DATA = os.path.join(HERE, 'bench', 'data')
PREVIEW = os.path.join(DATA, 'folk_preview.json')
MIN_COVER = 0.8          # share of beats that must carry a chord label
MIN_NOTES = 8
SEED = 5160


def corpus_dir():
    import music21
    return os.path.join(os.path.dirname(music21.__file__), 'corpus')


def default_sources():
    c = corpus_dir()
    pats = ['ryansMammoth/*.abc', 'nottingham-dataset/*.abc', 'oneills1850/*.abc', 'airdsAirs/*.abc',
            'leadSheet/*.mxl']
    return [p for pat in pats for p in sorted(glob.glob(os.path.join(c, pat)))]


# ---------------------------------------------------------------------- chord symbol -> triad index
def triad_pcs(cs):
    """Absolute pitch classes of a ChordSymbol's root, 3rd and 5th (7ths etc. dropped)."""
    out = {cs.root().pitchClass}
    for f in ('third', 'fifth'):
        p = getattr(cs, f, None)
        if p is not None:
            out.add(p.pitchClass)
    if len(out) < 3:                                   # sus / power chords: keep what is there
        out |= {p.pitchClass for p in cs.pitches[:3]}
    return out


def fold(pcs_rel, root_rel):
    """Diatonic triad index for a chord given relative pitch classes, or -1 (see module doc)."""
    share = [len(pcs_rel & CT[z]) for z in range(7)]
    best = max(share)
    if best < 2:
        return -1
    cands = [z for z in range(7) if share[z] == best]
    return min(cands, key=lambda z: (ROOT[z] != root_rel, TIE_ORDER.index(z)))


def label_symbol(cs, tonic):
    try:
        if cs.figure in ('N.C.', 'NC') or not cs.pitches:
            return -1
        pcs = {(p - tonic) % 12 for p in triad_pcs(cs)}
        return fold(pcs, (cs.root().pitchClass - tonic) % 12)
    except Exception:                                  # noqa: BLE001 - unparseable symbol
        return -1


# ---------------------------------------------------------------------- one tune
def key_of(score):
    """(tonic pc, mode) from the written key (ABC K: field / MusicXML key), else analysed."""
    from music21 import key
    k = score.recurse().getElementsByClass(key.Key).first()
    if k is not None:
        return k.tonic.pitchClass, k.mode
    ks = score.recurse().getElementsByClass(key.KeySignature).first()
    a = score.analyze('key')
    if ks is not None and a.mode == 'major' and a.sharps == ks.sharps:
        return a.tonic.pitchClass, 'major'
    return a.tonic.pitchClass, a.mode


def extract(score):
    """Song dict without id/title, or (None, reason)."""
    from music21 import harmony, meter
    tonic, mode = key_of(score)
    if mode != 'major':
        return None, f'mode {mode}'
    part = score.parts[0] if score.parts else score
    flat = part.stripTies().flatten()
    mel = []
    for n in flat.notes:
        if isinstance(n, harmony.Harmony):
            continue
        ql = float(n.quarterLength)
        if ql <= 0:                                    # grace notes
            continue
        mel.append((float(n.offset), float(n.offset) + ql, int(max(p.midi for p in n.pitches))))
    mel.sort()
    if len(mel) < MIN_NOTES:
        return None, 'too few notes'
    syms = sorted(((float(h.offset), h) for h in part.flatten().getElementsByClass(harmony.ChordSymbol)),
                  key=lambda x: x[0])
    if len(syms) < 4:
        return None, 'no chord symbols'
    tss = list(part.recurse().getElementsByClass(meter.TimeSignature))
    if len({ts.ratioString for ts in tss}) > 1:
        return None, 'meter changes'
    ts = tss[0] if tss else None
    beat_q = float(ts.beatDuration.quarterLength) if ts is not None else 1.0
    M = int(ts.beatCount) if ts is not None else 4
    bar_q = float(ts.barDuration.quarterLength) if ts is not None else 4.0
    meas = list(part.getElementsByClass('Measure'))
    if not meas:
        return None, 'no measures'
    m0 = meas[0]
    pickup = len(meas) > 1 and (m0.number == 0 or float(m0.duration.quarterLength) < bar_q - 1e-6)
    if pickup:                                         # pickup bar: downbeats start at the next bar
        origin = float(meas[1].offset)
    else:
        origin = float(m0.offset) - float(getattr(m0, 'paddingLeft', 0.0) or 0.0)
    first_on, end = mel[0][0], mel[-1][1]
    q0 = origin - beat_q * math.ceil((origin - first_on) / beat_q - 1e-9)
    K = int(math.ceil((end - q0) / beat_q - 1e-9))

    spans = []                                         # (on, off, z, figure)
    for i, (o, h) in enumerate(syms):
        off = syms[i + 1][0] if i + 1 < len(syms) else end
        if off > o + 1e-9:
            spans.append((o, off, label_symbol(h, tonic), h.figure))
    beats, labelled = [], 0
    for k in range(K):
        a, b = q0 + k * beat_q, q0 + (k + 1) * beat_q
        best, best_ov = None, 0.0
        for on, off, z, fig in spans:
            if on >= b:
                break
            ov = min(off, b) - max(on, a)
            if ov > best_ov + 1e-9:
                best, best_ov = (z, fig), ov
        z = best[0] if best else -1
        labelled += z >= 0
        pos = int(round((a - origin) / beat_q)) % M
        beats.append({'pos': pos, 'z': int(z), 'fit': 1.0 if z >= 0 else 0.0,
                      'onset_pcs': sorted(CT[z]) if z >= 0 else [],
                      'sym': best[1] if best else None})
    if labelled < MIN_COVER * K:
        return None, f'chord cover {labelled}/{K}'
    sop = [[round((on - q0) / beat_q, 6), round((off - q0) / beat_q, 6), m] for on, off, m in mel]
    return {'tonic': int(tonic), 'M': M, 'beat_q': beat_q, 'K': K, 'beats': beats, 'sop': sop,
            'meter': ts.ratioString if ts is not None else '?'}, None


def tunes_in(path):
    from music21 import converter, stream
    rel = os.path.relpath(path, corpus_dir()) if path.startswith(corpus_dir()) else path
    obj = converter.parse(path)
    scores = list(obj.scores) if isinstance(obj, stream.Opus) else [obj]
    for i, s in enumerate(scores):
        md = s.metadata
        num = getattr(md, 'number', None) if md is not None else None
        title = (md.title if md is not None else None) or os.path.basename(path)
        yield f'{rel}#{num if num is not None else i}', title, s


def build(paths, verbose=True):
    songs, skipped, why = [], [], Counter()
    for p in paths:
        try:
            items = list(tunes_in(p))
        except Exception as e:                         # noqa: BLE001
            skipped.append((p, repr(e)[:120]))
            why['parse error'] += 1
            continue
        for sid, title, s in items:
            try:
                rec, reason = extract(s)
            except Exception as e:                     # noqa: BLE001
                rec, reason = None, 'error ' + repr(e)[:80]
            if rec is None:
                skipped.append((sid, reason))
                why[reason.split(' ')[0] if reason.startswith(('mode', 'chord', 'error')) else reason] += 1
                continue
            rec.update({'id': sid, 'title': title})
            songs.append(rec)
            if verbose:
                print(f'  kept {sid}  {title!r}  {rec["meter"]}  K={rec["K"]}', flush=True)
    return songs, skipped, why


def split(songs, seed=SEED):
    ids = sorted(s['id'] for s in songs)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    a, b = round(0.6 * n), round(0.6 * n) + round(0.2 * n)
    part = {i: ('train' if r < a else 'val' if r < b else 'test') for r, i in enumerate(ids)}
    out = {'train': [], 'val': [], 'test': []}
    for s in songs:
        out[part[s['id']]].append(s)
    return out


# ---------------------------------------------------------------------- reporting
def describe(song, n=8):
    t = song['tonic']
    lines = [f"{song['id']}  {song['title']!r}  key {PC_NAMES[t]} major  {song['meter']}  "
             f"M={song['M']} beat_q={song['beat_q']}  K={song['K']}"]
    for k in range(min(n, song['K'])):
        notes = [f"{PC_NAMES[m % 12]}{m // 12 - 1}" for on, off, m in song['sop']
                 if min(off, k + 1) - max(on, k) > 1e-9]
        b = song['beats'][k]
        lab = chord_name(b['z'], t) if b['z'] >= 0 else '-1 (none)'
        lines.append(f"  beat {k:2d} pos {b['pos']}  melody {' '.join(notes) or '(rest)':24s} "
                     f"symbol {str(b['sym']):6s} -> {lab}")
    return '\n'.join(lines)


def selftest():
    from music21 import converter, harmony
    # fold(): D major tonic 2. "C" (bVII) shares D,F? no: C E G rel {10,2,5} -> ii or vii°, ii wins
    t = 2
    assert label_symbol(harmony.ChordSymbol('A7'), t) == 4            # V7 -> V
    assert label_symbol(harmony.ChordSymbol('Em'), t) == 1            # ii
    assert label_symbol(harmony.ChordSymbol('G'), t) == 3             # IV
    assert label_symbol(harmony.ChordSymbol('Bm'), t) == 5            # vi
    assert label_symbol(harmony.ChordSymbol('C'), t) == 1             # bVII folds to ii (tie, TIE_ORDER)
    assert label_symbol(harmony.ChordSymbol('E7'), t) == 1            # V/V (E G# B) -> ii (E,B shared)
    assert label_symbol(harmony.ChordSymbol('B-'), t) == -1           # shares 1 tone at most
    # 6/8 with a one-eighth pickup: beats are dotted quarters, the pickup beat is beat 0 of the grid
    abc = 'X:1\nT:t68\nM:6/8\nL:1/8\nK:G\nD|"G"GAB "D"ABc|"G"B2A "C"G2E|"D"D3 "G"G3|]\n'
    s = converter.parse(abc, format='abc')
    rec, why = extract(s)
    assert rec is not None, why
    assert (rec['M'], rec['beat_q']) == (2, 1.5), (rec['M'], rec['beat_q'])
    assert [b['pos'] for b in rec['beats']] == [1, 0, 1, 0, 1, 0, 1], [b['pos'] for b in rec['beats']]
    assert [b['z'] for b in rec['beats']] == [-1, 0, 4, 0, 3, 4, 0], [b['z'] for b in rec['beats']]
    assert abs(rec['sop'][1][0] - 1.0) < 1e-5 and abs(rec['sop'][0][0] - (1 - 1 / 3)) < 1e-5, rec['sop'][:2]
    # 4/4, no pickup
    abc = 'X:2\nT:t44\nM:4/4\nL:1/4\nK:C\n"C"cdef|"F"f2"G7"g2|"C"cdec|]\n'
    rec, why = extract(converter.parse(abc, format='abc'))
    assert rec is not None, why
    assert [b['pos'] for b in rec['beats']][:5] == [0, 1, 2, 3, 0]
    assert [b['z'] for b in rec['beats']] == [0, 0, 0, 0, 3, 3, 4, 4, 0, 0, 0, 0]
    print('selftest ok: fold rules, 6/8 pickup grid, 4/4 grid')


def check(path):
    """hmm baseline on a folk file's train+val tunes (unseen by the Bach-trained HMM), not logged."""
    import bench
    import importlib
    x = json.load(open(path))
    cfg = bench.config()
    system = importlib.import_module('bench_systems.hmm').build(bench.train_sets())
    for part in ('train', 'val'):
        if x[part]:
            r = bench.score(system, x[part], cfg)
            print(f"  folk[{part}] two {r['two']:.3f}  four {r['four']:.3f}  chord {r['chord']:.3f}  "
                  f"{r['us_per_beat']:.0f} us/beat  ({r['songs']} songs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', nargs='*', help='ABC / MusicXML files or directories (default: local corpus)')
    ap.add_argument('--out', default=PREVIEW)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--check')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.check:
        check(a.check)
        return
    paths = []
    for p in a.src or default_sources():
        paths += sorted(glob.glob(os.path.join(p, '*.abc')) + glob.glob(os.path.join(p, '*.mxl'))) \
            if os.path.isdir(p) else [p]
    songs, skipped, why = build(paths)
    parts = split(songs)
    src = ('music21 local corpus: ' + ', '.join(sorted({os.path.relpath(os.path.dirname(p), corpus_dir())
                                                        if p.startswith(corpus_dir()) else os.path.dirname(p)
                                                        for p in paths})))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump({**parts, 'source': src, 'skipped_reasons': dict(why), 'n_skipped': len(skipped)}, fh)
    beats = [b for s in songs for b in s['beats']]
    print(f'\nwrote {a.out}  ({os.path.getsize(a.out) / 1e3:.1f} kB)')
    print('tunes', len(songs), {k: len(v) for k, v in parts.items()}, '| skipped', len(skipped), dict(why))
    print('meters', dict(Counter(s['meter'] for s in songs)))
    print('beats', len(beats), 'unlabelled', sum(b['z'] < 0 for b in beats),
          'labels', {CHORDS[z] if z >= 0 else 'none': c for z, c in sorted(Counter(b['z'] for b in beats).items())})
    for s in songs[:2]:
        print('\n' + describe(s))


if __name__ == '__main__':
    main()
