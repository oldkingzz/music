"""Nottingham Music Database (jukedeck/nottingham-dataset, ABC_cleaned/*.abc) -> bench/data/folk.json.

A thin wrapper around build_folk.py (extract / split / describe are reused unchanged). Three fixes
the Nottingham ABC needs, found 2026-10-29:
  * chord-symbol spelling. Nottingham writes the bass in lower case with '+' / '-' for sharp / flat
    ("D/f+" = D/F#, "Gm/bb" = Gm/Bb) and 'd' / 'a' suffixes for diminished / augmented ("Gd",
    "D#d", "Ea"). music21 reads "D/f+" as an AUGMENTED D over F natural and drops "Gd" entirely, so
    every chord token is rewritten to standard spelling before parsing ("D/F#", "Gdim", "Eaug").
    Stray spaces / parentheses ("D m", " B7", "(E7)") are stripped; quoted strings that are still
    not chord symbols after that (" ") are removed.
  * mid-tune key changes. music21 keeps only the first K: of a tune, so the later sections would be
    labelled against the wrong tonic. Tunes whose K: (or M:) changes after the music starts are
    skipped ('key change' / 'meter changes'). Repeated identical K: lines are fine.
  * one tune at a time. Each X: block is parsed on its own, so a malformed tune skips only itself.
The downloaded ABC files are data: they are only read as text here, never executed.

  ../.venv/bin/python -I build_folk_nmd.py --src ../data/external/nottingham/nottingham-dataset-master/ABC_cleaned \
      --out bench/data/folk.json
  ../.venv/bin/python -I build_folk_nmd.py --selftest
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_folk as bf  # noqa: E402
from theory import CHORDS  # noqa: E402

CHORD_RE = re.compile(r'^([A-G])([#b]?)(m7|m6|m|maj7|7b9|7|6|d|a7|a)?(?:/([a-gA-G])([+\-#b]?))?$')
SUFFIX = {'d': 'dim', 'a': 'aug', 'a7': 'aug7'}
QUOTED = re.compile(r'"([^"\n]*)"')
FIELD = re.compile(r'^([A-Za-z]):\s*(.*?)\s*$')


def fix_symbol(tok):
    """Nottingham chord token -> music21 spelling, or None when it is not a chord symbol."""
    t = tok.replace(' ', '').strip('()')
    m = CHORD_RE.match(t)
    if not m:
        return None
    root, acc, qual, bass, bacc = m.groups()
    out = root + acc + SUFFIX.get(qual, qual or '')
    if bass:
        out += '/' + bass.upper() + {'+': '#', '#': '#', '-': 'b', 'b': 'b'}.get(bacc, '')
    return out


def fix_body_line(line):
    return QUOTED.sub(lambda m: f'"{fix_symbol(m.group(1))}"' if fix_symbol(m.group(1)) else '', line)


def split_tunes(text):
    """[(X number, block text)] for every X: block of an ABC file."""
    blocks, cur = [], None
    for line in text.splitlines():
        if line.startswith('X:'):
            cur = [line]
            blocks.append(cur)
        elif cur is not None:
            cur.append(line)
    return [(b[0][2:].strip(), b) for b in blocks]


def clean_tune(lines):
    """(abc text, title, None) or (None, title, reason). Header = field lines before the first music."""
    title, key0, met0, started, out = None, None, None, False, []
    for line in lines:
        f = FIELD.match(line)
        if f and not line.startswith('|'):
            tag, val = f.groups()
            if tag == 'T' and title is None:
                title = val
            if tag == 'K':
                if not started:
                    key0 = val
                elif val != key0:
                    return None, title, 'key change'
            if tag == 'M':
                if not started:
                    met0 = val
                elif val != met0:
                    return None, title, 'meter changes'
            out.append(line)
            continue
        if line.strip() and not line.lstrip().startswith('%'):
            started = True
            line = fix_body_line(line)
        out.append(line)
    if key0 is None:
        return None, title, 'no key'
    return '\n'.join(out) + '\n', title, None


def tunes_in(path):
    """(id, title, abc text or None, skip reason or None) for every tune of one ABC file."""
    with open(path, encoding='latin-1') as fh:
        text = fh.read()
    for num, lines in split_tunes(text):
        abc, title, reason = clean_tune(lines)
        yield f'nottingham/{os.path.basename(path)}#{num}', title or f'{os.path.basename(path)}#{num}', abc, reason


def build(paths, verbose=False):
    from music21 import converter
    songs, skipped, why = [], [], Counter()
    for p in paths:
        n_file = 0
        for sid, title, abc, reason in tunes_in(p):
            rec = None
            if reason is None:
                try:
                    rec, reason = bf.extract(converter.parseData(abc, format='abc'))
                except Exception as e:                 # noqa: BLE001 - malformed tune
                    rec, reason = None, 'error ' + repr(e)[:80]
            if rec is None:
                skipped.append((sid, reason))
                why[reason.split(' ')[0] if reason.startswith(('mode', 'chord', 'error')) else reason] += 1
                continue
            rec.update({'id': sid, 'title': title})
            songs.append(rec)
            n_file += 1
            if verbose:
                print(f'  kept {sid}  {title!r}  {rec["meter"]}  K={rec["K"]}', flush=True)
        print(f'  {os.path.basename(p)}: kept {n_file}', flush=True)
    return songs, skipped, why


def selftest():
    assert fix_symbol('D/f+') == 'D/F#' and fix_symbol('Gm/bb') == 'Gm/Bb' and fix_symbol('A7/c+') == 'A7/C#'
    assert fix_symbol('Gd') == 'Gdim' and fix_symbol('D#d') == 'D#dim' and fix_symbol('Ea') == 'Eaug'
    assert fix_symbol('D m') == 'Dm' and fix_symbol('(E7)') == 'E7' and fix_symbol(' B7') == 'B7'
    assert fix_symbol('Em/d+') == 'Em/D#' and fix_symbol('G/b') == 'G/B' and fix_symbol('Bm7') == 'Bm7'
    assert fix_symbol(' ') is None and fix_symbol('From here') is None
    from music21 import converter, harmony
    abc, _, why = clean_tune('X:1\nT:t\nM:4/4\nL:1/4\nK:D\n"D/f+"de"Gd"fg|"A7/c+"ab" "cd|]'.split('\n'))
    assert why is None
    figs = [h.figure for h in converter.parseData(abc, format='abc').flatten()
            .getElementsByClass(harmony.ChordSymbol)]
    assert figs == ['D/F#', 'Gdim', 'A7/C#'], figs
    assert bf.label_symbol(harmony.ChordSymbol('D/F#'), 2) == 0 and bf.label_symbol(harmony.ChordSymbol('A7/C#'), 2) == 4
    assert clean_tune('X:1\nT:t\nM:6/8\nK:A\nf|"A"ecc c2f|\nK:D\n"D"f2f Fdd|]'.split('\n'))[2] == 'key change'
    assert clean_tune('X:1\nT:t\nM:4/4\nK:A\nM:6/8\nf|"A"ecc c2f|\nK:A\n"A"f2f Fdd|]'.split('\n'))[2] is None
    assert clean_tune('X:1\nT:t\nM:6/8\nK:A\nf|"A"ecc c2f|\nM:9/8\n"A"f2f Fdd ddd|]'.split('\n'))[2] == 'meter changes'
    bf.selftest()
    print('nmd selftest ok: chord spelling, key / meter change detection')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', nargs='*', default=[])
    ap.add_argument('--out', default=os.path.join(bf.DATA, 'folk_nmd_preview.json'))
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    paths = []
    for p in a.src:
        paths += sorted(glob.glob(os.path.join(p, '*.abc'))) if os.path.isdir(p) else [p]
    songs, skipped, why = build(paths, a.verbose)
    parts = bf.split(songs)
    src = ('Nottingham Music Database, jukedeck/nottingham-dataset ABC_cleaned (GPL-3.0): '
           + ', '.join(os.path.basename(p) for p in paths))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump({**parts, 'source': src, 'skipped_reasons': dict(why), 'n_skipped': len(skipped)}, fh)
    beats = [b for s in songs for b in s['beats']]
    print(f'\nwrote {a.out}  ({os.path.getsize(a.out) / 1e3:.1f} kB)')
    print('tunes', len(songs), {k: len(v) for k, v in parts.items()}, '| skipped', len(skipped), dict(why))
    print('errors', [x for x in skipped if x[1].startswith('error')][:5])
    print('meters', dict(Counter(s['meter'] for s in songs)))
    print('tonics', dict(Counter(bf.PC_NAMES[s['tonic']] for s in songs)))
    print('beats', len(beats), 'unlabelled', sum(b['z'] < 0 for b in beats),
          'labels', {CHORDS[z] if z >= 0 else 'none': c for z, c in sorted(Counter(b['z'] for b in beats).items())})
    print('symbols -> none', Counter(b['sym'] for b in beats if b['z'] < 0 and b['sym']).most_common(12))
    for s in songs[:2]:
        print('\n' + bf.describe(s))


if __name__ == '__main__':
    main()
