"""POP909 (MIT licence, music-x-lab) on a beat grid, in the song format of data.load()['split'] songs.

Source: https://github.com/music-x-lab/POP909-Dataset  POP909.zip, unzipped (as data, never run) to
../data/external/pop909/POP909/NNN/{NNN.mid, beat_midi.txt, chord_midi.txt, key_audio.txt}.

Per song:
  * key: key_audio.txt must name ONE major key for the whole song (all its lines equal, ':maj');
    songs with a minor key or a modulation are skipped;
  * beat grid: beat_midi.txt rows (time_s, downbeat_2/4, downbeat_4/4). Column 3 marks the bar
    lines. M = the most common bar length in beats (4 in almost all songs); pos = beats since the
    last bar line, mod M (irregular bars, e.g. a 2-beat bar, simply restart pos at their bar line).
    beat_q = 1.0 (quarter-note beats), meter 'M/4';
  * her melody = the MELODY track of NNN.mid (mido), note times in seconds mapped to beat units by
    linear interpolation on the beat_midi times (extrapolated with the last beat length), made
    monophonic (same onset: highest note; a note is cut at the next onset). The grid starts at the
    beat holding her first note and ends at the beat holding her last offset (instrumental intro
    and outro dropped; interludes inside the song kept);
  * z per beat = the chord_midi.txt label (Harte syntax, e.g. 'E:min7/b7', 'A:sus4(b7)') sounding
    over most of the beat, reduced to root/3rd/5th (sus2 -> 1 2 5, sus4 -> 1 4 5, 7ths, 6ths and
    extensions dropped, bass ignored) and folded to the 7 diatonic triads (theory.CT) with the same
    rule as build_folk.fold; 'N' or nothing sharing >= 2 tones -> -1. fit 1.0 if labelled else 0.0,
    onset_pcs = sorted(CT[z]), 'sym' = the raw label.
Split by song id 80/10/10 with seed 5160.

  ../.venv/bin/python -I build_pop909.py [--src DIR] [--out bench/data/pop909.json] [--limit N]
  ../.venv/bin/python -I build_pop909.py --selftest
"""
import argparse
import bisect
import glob
import json
import os
import random
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from theory import CHORDS, CT, PC_NAMES, ROOT, TIE_ORDER, chord_name  # noqa: E402

SRC = os.path.join(os.path.dirname(HERE), 'data', 'external', 'pop909', 'POP909')
OUT = os.path.join(HERE, 'bench', 'data', 'pop909.json')
SEED = 5160
MIN_NOTES = 16
MIN_COVER = 0.8                     # share of melody beats that must carry a chord label
NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
QUAL = {                            # Harte quality -> triad intervals kept (root, 3rd/sus, 5th)
    'maj': (0, 4, 7), 'min': (0, 3, 7), 'dim': (0, 3, 6), 'aug': (0, 4, 8),
    'sus2': (0, 2, 7), 'sus4': (0, 5, 7), '7': (0, 4, 7), 'maj7': (0, 4, 7), 'min7': (0, 3, 7),
    'minmaj7': (0, 3, 7), 'hdim7': (0, 3, 6), 'dim7': (0, 3, 6), 'maj6': (0, 4, 7), 'min6': (0, 3, 7),
    '9': (0, 4, 7), 'maj9': (0, 4, 7), 'min9': (0, 3, 7), '11': (0, 4, 7), 'min11': (0, 3, 7),
    '13': (0, 4, 7), 'maj13': (0, 4, 7), 'min13': (0, 3, 7), '5': (0, 7), '1': (0,), '': (0, 4, 7),
}


def pc_of(name):
    pc = NOTE[name[0]]
    for ch in name[1:]:
        pc += {'#': 1, 'b': -1}[ch]
    return pc % 12


# ---------------------------------------------------------------------- chord label -> triad index
def fold(pcs_rel, root_rel):
    """Copy of build_folk.fold: diatonic triad index sharing the most tones, or -1 (< 2 shared)."""
    share = [len(pcs_rel & CT[z]) for z in range(7)]
    best = max(share)
    if best < 2:
        return -1
    cands = [z for z in range(7) if share[z] == best]
    return min(cands, key=lambda z: (ROOT[z] != root_rel, TIE_ORDER.index(z)))


def parse_harte(lab):
    """(root pc, triad pcs) of a Harte label, or None for 'N' / unknown qualities."""
    if lab in ('N', 'X') or ':' not in lab and lab[:1] not in NOTE:
        return None
    root, _, q = lab.partition(':')
    q = q.split('/')[0].split('(')[0]
    if q not in QUAL:
        return None
    r = pc_of(root)
    return r, frozenset((r + i) % 12 for i in QUAL[q])


def label(lab, tonic):
    p = parse_harte(lab)
    if p is None:
        return -1
    r, pcs = p
    return fold(frozenset((x - tonic) % 12 for x in pcs), (r - tonic) % 12)


# ---------------------------------------------------------------------- files
def read_key(path):
    keys = {ln.split()[2] for ln in open(path) if len(ln.split()) >= 3}
    if len(keys) != 1:
        return None, f'{len(keys)} keys'
    k = keys.pop()
    root, _, mode = k.partition(':')
    if mode != 'maj':
        return None, 'minor'
    return pc_of(root), None


def read_beats(path):
    t, down = [], []
    for ln in open(path):
        f = ln.split()
        if len(f) >= 3:
            t.append(float(f[0]))
            down.append(float(f[2]) >= 0.5)
    return t, down


def read_chords(path):
    out = []
    for ln in open(path):
        f = ln.split()
        if len(f) >= 3:
            out.append((float(f[0]), float(f[1]), f[2]))
    return out


def read_melody(path):
    """[(on_s, off_s, midi)] of the MELODY track, via mido with the file's tempo map."""
    import mido
    mf = mido.MidiFile(path)
    tempo = []                                         # (abs tick, us per beat)
    mel_tr = None
    for tr in mf.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.type == 'set_tempo':
                tempo.append((tick, msg.tempo))
        if any(m.type == 'track_name' and m.name.strip().upper() == 'MELODY' for m in tr):
            mel_tr = tr
    if mel_tr is None:
        return None
    tempo = sorted(tempo) or [(0, 500000)]
    if tempo[0][0] > 0:
        tempo.insert(0, (0, 500000))
    seg_s, s = [], 0.0                                 # seconds at each tempo change
    for i, (tk, us) in enumerate(tempo):
        if i:
            s += (tk - tempo[i - 1][0]) * tempo[i - 1][1] / 1e6 / mf.ticks_per_beat
        seg_s.append(s)
    ticks = [tk for tk, _ in tempo]

    def sec(tk):
        i = bisect.bisect_right(ticks, tk) - 1
        return seg_s[i] + (tk - ticks[i]) * tempo[i][1] / 1e6 / mf.ticks_per_beat

    notes, on, tick = [], {}, 0
    for msg in mel_tr:
        tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            on.setdefault(msg.note, []).append(tick)
        elif msg.type in ('note_off', 'note_on') and on.get(msg.note):
            t0 = on[msg.note].pop(0)
            if tick > t0:
                notes.append((sec(t0), sec(tick), msg.note))
    return sorted(notes)


def to_beats(ts, grid):
    """Seconds -> beat units by linear interpolation on the grid (extrapolated at both ends)."""
    i = bisect.bisect_right(grid, ts) - 1
    i = min(max(i, 0), len(grid) - 2)
    return i + (ts - grid[i]) / (grid[i + 1] - grid[i])


# ---------------------------------------------------------------------- one song
def extract(d):
    sid = os.path.basename(d)
    tonic, why = read_key(os.path.join(d, 'key_audio.txt'))
    if tonic is None:
        return None, why
    grid, down = read_beats(os.path.join(d, 'beat_midi.txt'))
    if len(grid) < 16 or any(b <= a for a, b in zip(grid, grid[1:])):
        return None, 'bad beat grid'
    raw = read_melody(os.path.join(d, f'{sid}.mid'))
    if raw is None:
        return None, 'no MELODY track'
    if len(raw) < MIN_NOTES:
        return None, 'too few notes'
    # monophonic: same onset -> highest; cut at the next onset
    by_on = {}
    for on, off, m in raw:
        key = round(on, 4)
        if key not in by_on or m > by_on[key][2]:
            by_on[key] = (on, off, m)
    mono = sorted(by_on.values())
    mel = []
    for i, (on, off, m) in enumerate(mono):
        if i + 1 < len(mono):
            off = min(off, mono[i + 1][0])
        a, b = to_beats(on, grid), to_beats(off, grid)
        if b - a > 1e-3:
            mel.append((a, b, m))
    if len(mel) < MIN_NOTES:
        return None, 'too few notes'
    # bar lines -> M and pos
    bars = [i for i, f in enumerate(down) if f]
    if len(bars) < 2:
        return None, 'no bar lines'
    lens = Counter(b - a for a, b in zip(bars, bars[1:]))
    M = lens.most_common(1)[0][0]
    if M not in (2, 3, 4, 6):
        return None, f'bar length {M}'
    regular = sum(c * n for n, c in lens.items() if n == M) / max(1, bars[-1] - bars[0])
    k0 = int(mel[0][0] // 1)
    k1 = int((mel[-1][1] - 1e-6) // 1)                  # last beat holding melody
    k0 = max(k0, -10_000)
    K = k1 - k0 + 1
    chords = [(to_beats(a, grid), to_beats(b, grid), lab) for a, b, lab in read_chords(os.path.join(d, 'chord_midi.txt'))]
    zl = {}
    beats, labelled = [], 0
    ci = 0
    for kk in range(k0, k1 + 1):
        a, b = kk, kk + 1
        while ci < len(chords) and chords[ci][1] <= a:
            ci += 1
        best, best_ov = None, 0.0
        for on, off, lab in chords[ci:]:
            if on >= b:
                break
            ov = min(off, b) - max(on, a)
            if ov > best_ov + 1e-9:
                best, best_ov = lab, ov
        if best is not None and best not in zl:
            zl[best] = label(best, tonic)
        z = zl[best] if best is not None else -1
        labelled += z >= 0
        j = bisect.bisect_right(bars, kk) - 1
        pos = (kk - bars[j]) % M if j >= 0 else (kk - bars[0]) % M
        beats.append({'pos': int(pos), 'z': int(z), 'fit': 1.0 if z >= 0 else 0.0,
                      'onset_pcs': sorted(CT[z]) if z >= 0 else [], 'sym': best})
    if labelled < MIN_COVER * K:
        return None, 'chord cover'
    sop = [[round(on - k0, 4), round(off - k0, 4), int(m)] for on, off, m in mel]
    return {'id': f'pop909/{sid}', 'title': f'POP909 {sid}', 'tonic': int(tonic), 'M': int(M),
            'beat_q': 1.0, 'K': int(K), 'beats': beats, 'sop': sop, 'meter': f'{M}/4',
            'bar_regular': round(regular, 3), 'tempo_bpm': round(60.0 * (len(grid) - 1) / (grid[-1] - grid[0]), 1)}, None


def split(songs, seed=SEED):
    ids = sorted(s['id'] for s in songs)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    a, b = round(0.8 * n), round(0.8 * n) + round(0.1 * n)
    part = {i: ('train' if r < a else 'val' if r < b else 'test') for r, i in enumerate(ids)}
    out = {'train': [], 'val': [], 'test': []}
    for s in sorted(songs, key=lambda s: s['id']):
        out[part[s['id']]].append(s)
    return out


def describe(song, n=10):
    t = song['tonic']
    lines = [f"{song['id']}  key {PC_NAMES[t]} major  {song['meter']}  M={song['M']}  K={song['K']}  "
             f"~{song['tempo_bpm']} bpm  bars regular {song['bar_regular']}"]
    k0 = next((k for k, b in enumerate(song['beats']) if b['z'] >= 0), 0)
    for k in range(k0, min(k0 + n, song['K'])):
        notes = [f"{PC_NAMES[m % 12]}{m // 12 - 1}" for on, off, m in song['sop'] if min(off, k + 1) - max(on, k) > 1e-9]
        b = song['beats'][k]
        lab = chord_name(b['z'], t) if b['z'] >= 0 else '-1 (none)'
        lines.append(f"  beat {k:3d} pos {b['pos']}  melody {' '.join(notes) or '(rest)':22s} "
                     f"label {str(b['sym']):12s} -> {lab}")
    return '\n'.join(lines)


def selftest():
    t = 7                                                    # G major
    assert label('D:7', t) == 4 and label('E:min7/b7', t) == 5 and label('C:maj/3', t) == 3
    assert label('A:min', t) == 1 and label('F#:hdim7', t) == 6 and label('N', t) == -1
    assert label('D:sus4(b7)', t) == 4                       # D G A -> V (D, A shared; root D)
    assert label('G:sus2', t) == 0                           # G A D -> I
    assert label('Bb:maj', t) == -1                          # Bb D F shares 1 tone with every triad? D,F -> vii/ii
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    dirs = sorted(d for d in glob.glob(os.path.join(a.src, '[0-9][0-9][0-9]')) if os.path.isdir(d))
    if a.limit:
        dirs = dirs[:a.limit]
    t0 = time.time()
    songs, why = [], Counter()
    for d in dirs:
        try:
            rec, reason = extract(d)
        except Exception as e:                               # noqa: BLE001
            rec, reason = None, 'error ' + type(e).__name__
        if rec is None:
            why[reason] += 1
            continue
        songs.append(rec)
    parts = split(songs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump({**parts, 'source': 'POP909 (MIT), music-x-lab/POP909-Dataset POP909.zip: MELODY track, '
                   'beat_midi.txt, chord_midi.txt, key_audio.txt; single major key songs only',
                   'skipped_reasons': dict(why), 'n_skipped': sum(why.values())}, fh, separators=(',', ':'))
    beats = [b for s in songs for b in s['beats']]
    print(f'{len(dirs)} songs read in {time.time() - t0:.1f} s; wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB)')
    print('kept', len(songs), {k: len(v) for k, v in parts.items()}, '| skipped', sum(why.values()), dict(why))
    print('meters', dict(Counter(s['meter'] for s in songs)),
          'bars regular (median)', sorted(s['bar_regular'] for s in songs)[len(songs) // 2] if songs else None)
    print('beats', len(beats), 'unlabelled', sum(b['z'] < 0 for b in beats),
          'labels', {CHORDS[z] if z >= 0 else 'none': c for z, c in sorted(Counter(b['z'] for b in beats).items())})
    for s in parts['train'][:2]:
        print('\n' + describe(s))


if __name__ == '__main__':
    main()
