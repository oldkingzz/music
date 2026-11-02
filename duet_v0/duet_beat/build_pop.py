"""Hooktheory pop segments on a beat grid, in the same song format as data.load()['split'] songs.

Source: the Sheet Sage release of Hooktheory TheoryTab annotations (Donahue et al. 2022),
  https://raw.githubusercontent.com/chrisdonahue/sheetsage-data/main/hooktheory/Hooktheory.json.gz
LICENSE: CC BY-NC-SA. Private use only: never put this data, or pages built from it, in public.

The gzipped JSON is read as data only (gzip + json.load). Each record is one annotated segment
(verse, chorus, ...) of a song; onsets/offsets are in beats from beat 0 = the first downbeat
(Hookpad has no pickup bars). For every segment with ONE major key and ONE meter we keep:
  * her melody = the annotated melody, midi = 12 * (octave + 5) + pitch_class, then shifted by whole
    octaves so its median is as close as possible to 67 (the octave field is relative); segments
    whose melody spans more than 3 octaves are dropped (instrumental riffs, not a sung line);
  * M = beats_per_bar of the meter, beat_q = 1.0 (one dataset beat = one grid beat; for x/8 meters
    the dataset's beat is the eighth note), pos = k % M;
  * z = the diatonic triad (theory.CT, relative to the tonic) sharing the most tones with the chord
    sounding over most of the beat; chord tones = root_pitch_class + cumulative
    root_position_intervals (so 7ths/9ths are included); ties: the triad on the same root first,
    then theory.TIE_ORDER; fewer than 2 shared tones (or no chord sounding) -> -1;
  * onset_pcs = sorted(CT[z]) (as songs.twinkle), fit = 1.0 for labelled beats, 0.0 otherwise.
The dataset's own split field is used (TRAIN/VALID/TEST -> train/val/test; it is grouped by song:
no song has segments in two splits). Segments keep the file order inside each split.

  ../.venv/bin/python build_pop.py                 # build bench/data/pop.json
  ../.venv/bin/python build_pop.py --selftest      # chord folding / grid checks
"""
import argparse
import gzip
import json
import math
import os
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from theory import CHORDS, CT, PC_NAMES, ROOT, TIE_ORDER, chord_name  # noqa: E402

SRC = os.path.join(os.path.dirname(HERE), 'data', 'external', 'hooktheory', 'Hooktheory.json.gz')
OUT = os.path.join(HERE, 'bench', 'data', 'pop.json')
MAJOR_SDI = (2, 2, 1, 2, 2, 2)
MODES = {MAJOR_SDI: 'major', (2, 1, 2, 2, 1, 2): 'minor', (2, 2, 1, 2, 2, 1): 'mixolydian',
         (2, 1, 2, 2, 2, 1): 'dorian', (2, 2, 2, 1, 2, 2): 'lydian', (1, 2, 2, 2, 1, 2): 'phrygian',
         (1, 2, 2, 1, 2, 2): 'locrian'}
SPLITS = {'TRAIN': 'train', 'VALID': 'val', 'TEST': 'test'}
MIN_NOTES = 8
MIN_COVER = 0.8          # share of beats that must carry a chord label (as build_folk.py)
TARGET_MEDIAN = 67
MAX_RANGE = 36           # semitones; wider "melodies" are instrumental lines, not something she sings


# ---------------------------------------------------------------------- chord -> triad index
def chord_pcs(h):
    """Absolute pitch classes of a Hooktheory chord: root plus the stacked root-position intervals."""
    pcs, p = {h['root_pitch_class'] % 12}, h['root_pitch_class']
    for iv in h['root_position_intervals']:
        p += iv
        pcs.add(p % 12)
    return pcs


def fold(pcs_rel, root_rel):
    """Diatonic triad index for a chord given relative pitch classes, or -1 (see module doc)."""
    share = [len(pcs_rel & CT[z]) for z in range(7)]
    best = max(share)
    if best < 2:
        return -1
    cands = [z for z in range(7) if share[z] == best]
    return min(cands, key=lambda z: (ROOT[z] != root_rel, TIE_ORDER.index(z)))


def label(h, tonic):
    return fold({(p - tonic) % 12 for p in chord_pcs(h)}, (h['root_pitch_class'] - tonic) % 12)


# ---------------------------------------------------------------------- one segment
def convert(seg):
    """Song dict without id/title, or (None, reason)."""
    a = seg.get('annotations') or {}
    keys, meters = a.get('keys') or [], a.get('meters') or []
    mel, harm = a.get('melody') or [], a.get('harmony') or []
    if not mel:
        return None, 'no melody'
    if not harm:
        return None, 'no harmony'
    if not keys:
        return None, 'no key'
    modes = {MODES.get(tuple(k['scale_degree_intervals']), 'other') for k in keys}
    if modes != {'major'}:
        return None, 'mode ' + '+'.join(sorted(modes))
    if len({k['tonic_pitch_class'] for k in keys}) > 1:
        return None, 'key changes'
    if len({(m['beats_per_bar'], m['beat_unit']) for m in meters}) != 1:
        return None, 'meter changes' if meters else 'no meter'
    tonic = keys[0]['tonic_pitch_class'] % 12
    M, unit = meters[0]['beats_per_bar'], meters[0]['beat_unit']

    notes = sorted((float(n['onset']), float(n['offset']), 12 * (n['octave'] + 5) + n['pitch_class'])
                   for n in mel if n['offset'] > n['onset'])
    for i in range(len(notes) - 1):                    # monophonic: trim any overlap
        if notes[i][1] > notes[i + 1][0]:
            notes[i] = (notes[i][0], notes[i + 1][0], notes[i][2])
    notes = [n for n in notes if n[1] > n[0] + 1e-9]
    if len(notes) < MIN_NOTES:
        return None, 'too few notes'
    if max(m for _, _, m in notes) - min(m for _, _, m in notes) > MAX_RANGE:
        return None, 'melody range > 3 octaves'
    med = statistics.median(m for _, _, m in notes)
    shift = 12 * round((TARGET_MEDIAN - med) / 12)
    sop = [[round(on, 6), round(off, 6), int(m + shift)] for on, off, m in notes]

    end = max(a.get('num_beats') or 0, notes[-1][1], max(h['offset'] for h in harm))
    K = int(math.ceil(end - 1e-9))
    spans = sorted((float(h['onset']), float(h['offset']), label(h, tonic)) for h in harm
                   if h['offset'] > h['onset'])
    beats, labelled = [], 0
    for k in range(K):
        best, best_ov = -1, 0.0
        for on, off, z in spans:
            if on >= k + 1:
                break
            ov = min(off, k + 1) - max(on, k)
            if ov > best_ov + 1e-9:
                best, best_ov = z, ov
        labelled += best >= 0
        beats.append({'pos': k % M, 'z': int(best), 'fit': 1.0 if best >= 0 else 0.0,
                      'onset_pcs': sorted(CT[best]) if best >= 0 else []})
    if labelled < MIN_COVER * K:
        return None, 'chord cover < 0.8'
    return {'tonic': int(tonic), 'M': int(M), 'beat_q': 1.0, 'K': K, 'beats': beats, 'sop': sop,
            '_meter': f'{M}/{unit}', '_shift': shift}, None


def build(src):
    with gzip.open(src, 'rt') as fh:
        raw = json.load(fh)
    out, why, meters, shifts = {'train': [], 'val': [], 'test': []}, Counter(), Counter(), Counter()
    for sid, seg in raw.items():
        part = SPLITS.get(seg.get('split'))
        if part is None:
            why['unknown split'] += 1
            continue
        rec, reason = convert(seg)
        if rec is None:
            why[reason] += 1
            continue
        ht = seg.get('hooktheory') or {}
        meters[rec.pop('_meter')] += 1
        shifts[rec.pop('_shift')] += 1
        out[part].append({'id': f'hooktheory:{sid}', 'title': f"{ht.get('artist', '?')} - {ht.get('song', '?')}",
                          **rec})
    return out, why, meters, shifts, len(raw)


# ---------------------------------------------------------------------- reporting
def describe(song, n=8):
    t = song['tonic']
    lines = [f"{song['id']}  {song['title']!r}  key {PC_NAMES[t]} major  M={song['M']}  K={song['K']}"]
    for k in range(min(n, song['K'])):
        notes = [f"{PC_NAMES[m % 12]}{m // 12 - 1}" for on, off, m in song['sop']
                 if min(off, k + 1) - max(on, k) > 1e-9]
        z = song['beats'][k]['z']
        lab = chord_name(z, t) if z >= 0 else '-1 (none)'
        lines.append(f"  beat {k:2d} pos {song['beats'][k]['pos']}  melody {' '.join(notes) or '(rest)':24s} -> {lab}")
    return '\n'.join(lines)


def selftest():
    t = 2                                                              # D major
    ch = lambda r, ivs: {'root_pitch_class': r, 'root_position_intervals': ivs}  # noqa: E731
    assert label(ch(9, [4, 3, 3]), t) == 4                            # A7 -> V
    assert label(ch(4, [3, 4]), t) == 1                               # Em -> ii
    assert label(ch(7, [4, 3]), t) == 3                               # G -> IV
    assert label(ch(11, [3, 4, 3]), t) == 5                           # Bm7 -> vi (same root wins tie with I)
    assert label(ch(2, [4, 3, 4]), t) == 0                            # Dmaj7 -> I (tie with iii)
    assert label(ch(0, [4, 3]), t) == 1                               # C (bVII) -> ii (TIE_ORDER)
    assert label(ch(10, [4, 3]), t) == -1                             # Bb shares 1 tone at most
    assert label(ch(2, []), t) == -1                                  # bare root
    seg = {'annotations': {'num_beats': 8, 'meters': [{'beat': 0, 'beats_per_bar': 4, 'beat_unit': 4}],
                           'keys': [{'beat': 0, 'tonic_pitch_class': 0, 'scale_degree_intervals': list(MAJOR_SDI)}],
                           'melody': [{'onset': i, 'offset': i + 1, 'octave': 0, 'pitch_class': p}
                                      for i, p in enumerate([0, 2, 4, 5, 7, 5, 4, 2])],
                           'harmony': [{'onset': 0, 'offset': 2.6, 'root_pitch_class': 0, 'root_position_intervals': [4, 3]},
                                       {'onset': 2.6, 'offset': 4, 'root_pitch_class': 5, 'root_position_intervals': [4, 3]},
                                       {'onset': 4, 'offset': 8, 'root_pitch_class': 7, 'root_position_intervals': [4, 3, 3]}]}}
    rec, why = convert(seg)
    assert rec is not None, why
    assert [b['z'] for b in rec['beats']] == [0, 0, 0, 3, 4, 4, 4, 4], [b['z'] for b in rec['beats']]
    assert [b['pos'] for b in rec['beats']] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [m for _, _, m in rec['sop']][:3] == [60, 62, 64]             # median 64 -> no shift
    seg['annotations']['keys'][0]['scale_degree_intervals'] = [2, 1, 2, 2, 1, 2]
    assert convert(seg) == (None, 'mode minor')
    print('selftest ok: chord folding, majority-overlap labels, grid, octave shift, mode filter')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    parts, why, meters, shifts, n_raw = build(a.src)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump({**parts, 'source': 'Hooktheory (Sheet Sage release), CC BY-NC-SA, private use only',
                   'skipped_reasons': dict(why), 'n_skipped': sum(why.values())}, fh, separators=(',', ':'))
    songs = [s for p in parts.values() for s in p]
    beats = [b for s in songs for b in s['beats']]
    mids = [m for s in songs for _, _, m in s['sop']]
    print(f'wrote {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB)')
    print('segments in file', n_raw, '| kept', len(songs), {k: len(v) for k, v in parts.items()},
          '| skipped', sum(why.values()))
    for r, c in why.most_common():
        print(f'    {c:6d}  {r}')
    print('meters', dict(meters.most_common()), '| octave shifts', dict(sorted(shifts.items())))
    print('beats', len(beats), 'unlabelled', sum(b['z'] < 0 for b in beats),
          'labels', {CHORDS[z] if z >= 0 else 'none': c for z, c in sorted(Counter(b['z'] for b in beats).items())})
    print('melody midi range', min(mids), max(mids), 'outside 48..88:', sum(not 48 <= m <= 88 for m in mids),
          'of', len(mids))
    for s in parts['val'][:1] + parts['train'][:1]:
        print('\n' + describe(s))


if __name__ == '__main__':
    main()
