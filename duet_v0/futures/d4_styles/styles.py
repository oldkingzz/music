"""D4 / A4.1 + A4.2: the same melody harmonized as Bach (SATB), pop (triads) and jazz (7ths, ii-V, tritone subs).

Every style returns NOTE EVENTS {on, off, midi, voice} (what Duet sounds) plus unpitched percussion;
the no-clash shares are measured on exactly these events against her TRUE notes, then rendered to audio.

  .venv/bin/python futures/d4_styles/styles.py          # metrics + demos + blind kit
"""
import json
import math
import os
import random
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'futures', 'common'))
from feats import dur_onset, features, load_all  # noqa: E402
from satb import harmonize as satb_harmonize  # noqa: E402  (loads harmony/model.py as 'model')
sys.modules.pop('model', None)                    # duet_beat/model.py must win for chord_board
sys.path[:] = [p for p in sys.path if not p.rstrip('/').endswith('/harmony')]
import chord_board  # noqa: E402
from chord_board import NpBiGRU, StyleHMM, fb, trans_mats  # noqa: E402

from dsp import SR, colored_noise, midi_to_hz  # noqa: E402
import singer  # noqa: E402

OUT = os.path.join(HERE, 'out')
DEMO = os.path.join(HERE, 'demo')
BLIND = os.path.join(HERE, 'blind')
for d in (OUT, DEMO, BLIND):
    os.makedirs(d, exist_ok=True)
MAJOR = (0, 2, 4, 5, 7, 9, 11)
HARSH = frozenset({1, 2, 6, 10, 11})
FRAME = 0.005
LAME = '/opt/homebrew/bin/lame'
SEVENTH = {0: 'maj7', 3: 'maj7', 4: '7', 1: 'm7', 2: 'm7', 5: 'm7', 6: 'm7b5'}
TRIAD = {0: 'maj', 3: 'maj', 4: 'maj', 1: 'm', 2: 'm', 5: 'm', 6: 'dim'}
IV = {'maj': (0, 4, 7), 'm': (0, 3, 7), 'dim': (0, 3, 6), 'maj7': (0, 4, 7, 11), '7': (0, 4, 7, 10),
      'm7': (0, 3, 7, 10), 'm7b5': (0, 3, 6, 10)}
ROOTLESS = {'maj7': [(4, 7, 11, 14), (4, 7, 11, 9), (4, 7, 11, 12)], '7': [(4, 9, 10, 14), (4, 7, 10, 14), (4, 7, 10, 12)],
            'm7': [(3, 7, 10, 14), (3, 7, 10, 12), (3, 5, 7, 10)], 'm7b5': [(3, 6, 10, 12), (3, 6, 10, 14)]}


# ====================================================================== clash metrics (on events)
def her_frames(notes, T):
    n = int(math.ceil(T / FRAME)) + 1
    her = np.full(n, -1)
    for on, off, m in notes:
        her[int(round(on / FRAME)):int(round(off / FRAME))] = m
    return her


def no_clash(notes, events, rule='harsh'):
    """Time share of her singing where NO Duet event is a clash (rule 'harsh': interval class in
    {1,2,6,10,11}; rule 'avoid': her pc a semitone ABOVE the event pc). Per voice too."""
    T = max([o for _, o, _ in notes] + [e['off'] for e in events] + [0]) + 1
    her = her_frames(notes, T)
    sing = her >= 0
    any_c = np.zeros(len(her), bool)
    per = {}
    for e in events:
        a, b = int(round(e['on'] / FRAME)), int(round(e['off'] / FRAME))
        h = her[a:b]
        if rule == 'harsh':
            c = np.isin(np.abs(h - e['midi']) % 12, list(HARSH))
        else:
            c = ((h - e['midi']) % 12) == 1
        c &= h >= 0
        any_c[a:b] |= c
        per.setdefault(e['voice'], np.zeros(len(her), bool))[a:b] |= c
    tot = sing.sum()
    return {'all': float(1 - any_c[sing].sum() / tot),
            'per_voice': {v: float(1 - m[sing].sum() / tot) for v, m in per.items()},
            'her_s': float(tot * FRAME)}


def _selftest_metrics():
    her = [(0.0, 1.0, 60)]
    assert no_clash(her, [{'on': 0, 'off': 1, 'midi': 52, 'voice': 'A'}])['all'] == 1.0      # E below: 3rd
    r = no_clash(her, [{'on': 0, 'off': 0.5, 'midi': 62, 'voice': 'A'}])
    assert abs(r['all'] - 0.5) < 0.01                                                     # D: 2nd, half the time
    assert no_clash(her, [{'on': 0, 'off': 1, 'midi': 53, 'voice': 'A'}])['all'] == 1.0      # F: 4th allowed
    her_f = [(0.0, 1.0, 65)]
    assert no_clash(her_f, [{'on': 0, 'off': 1, 'midi': 52, 'voice': 'P'}], 'avoid')['all'] == 0.0   # F over E: avoid note
    assert no_clash([(0.0, 1.0, 64)], [{'on': 0, 'off': 1, 'midi': 53, 'voice': 'P'}], 'avoid')['all'] == 1.0  # E under F: allowed
    assert no_clash([(0.0, 1.0, 64)], [{'on': 0, 'off': 1, 'midi': 53, 'voice': 'P'}], 'harsh')['all'] == 0.0
    assert no_clash(her, [{'on': 2, 'off': 3, 'midi': 61, 'voice': 'A'}])['all'] == 1.0      # not overlapping


# ====================================================================== melodies
class Melody:
    """Her notes in seconds plus a beat grid (beat k at t0 + k * beat_s, bar position pos[k])."""

    def __init__(self, name, notes, t0, beat_s, M, pos, tonic, mode='major', audio=None, source='', licence=''):
        self.name, self.notes, self.t0, self.beat_s, self.M = name, notes, t0, beat_s, M
        self.pos, self.tonic, self.mode, self.audio, self.source, self.licence = pos, tonic, mode, audio, source, licence
        self.K = len(pos)

    @property
    def rel_tonic(self):          # the 7 diatonic triads are of the major key (relative major for minor)
        return self.tonic if self.mode == 'major' else (self.tonic + 3) % 12

    def tb(self, k):
        return self.t0 + k * self.beat_s

    def beat_song(self):
        sop = [[(on - self.t0) / self.beat_s, (off - self.t0) / self.beat_s, m] for on, off, m in self.notes]
        return {'tonic': self.rel_tonic, 'M': self.M, 'K': self.K, 'sop': sop,
                'beats': [{'pos': p, 'z': -1} for p in self.pos]}

    def her_in(self, a, b):
        """[(overlap_s, midi)] of her notes within [a, b) seconds."""
        return [(min(off, b) - max(on, a), m) for on, off, m in self.notes if min(off, b) - max(on, a) > 1e-6]


def from_dataset(name, song, b0, nb, bpm, lead=0.6, source='', licence=''):
    beat_s = 60.0 / bpm
    notes = []
    for on, off, m in song['sop']:
        if on >= b0 + nb or off <= b0:
            continue
        on, off = max(on, b0), min(off, b0 + nb)
        notes.append((lead + (on - b0) * beat_s, lead + (off - b0) * beat_s, int(m)))
    notes = [(a, min(b, notes[i + 1][0]) if i + 1 < len(notes) else b, m) for i, (a, b, m) in enumerate(notes)]
    pos = [song['beats'][k]['pos'] for k in range(b0, min(b0 + nb, song['K']))]
    return Melody(name, notes, lead, beat_s, song['M'], pos, song['tonic'], 'major', None, source, licence)


def vocadito_melody():
    """The vocadito excerpt with the steadiest pulse (beat from estimate_beat on the note annotations)."""
    from live.rehearsal import estimate_beat
    import soundfile as sf
    from harmony.model import detect_key_melody
    base = os.path.join(REPO, 'data', 'external', 'vocadito')
    best = None
    for tid in range(1, 41):
        p = os.path.join(base, 'Annotations', 'Notes', f'vocadito_{tid}_notesA1.csv')
        if not os.path.exists(p):
            continue
        rows = np.loadtxt(p, delimiter=',', ndmin=2)
        notes = [{'on': float(a), 'off': float(a + d), 'midi': int(round(69 + 12 * math.log2(f / 440.0)))} for a, f, d in rows]
        if len(notes) < 16:
            continue
        P = estimate_beat(notes, [])
        ons = np.array([n['on'] for n in notes])
        # phase and regularity: share of onsets within 12 % of a beat (or half beat) of the grid
        best_ph = None
        for ph in np.linspace(0, P, 40, endpoint=False):
            q = (ons - ons[0] - ph) / (P / 2)
            fit = float(np.mean(np.abs(q - np.round(q)) < 0.12 * 2))
            if best_ph is None or fit > best_ph[0]:
                best_ph = (fit, ph)
        h = np.zeros(12)
        for n in notes:
            h[n['midi'] % 12] += n['off'] - n['on']
        in_key = max(sum(h[(t + d) % 12] for d in MAJOR) for t in range(12)) / h.sum()
        sc = best_ph[0] * in_key ** 2            # steady pulse AND diatonic (the chord vocabulary is diatonic)
        if 0.45 <= P <= 0.9 and in_key >= 0.85 and (best is None or sc > best[0]):
            best = (sc, tid, notes, P, ons[0] + best_ph[1])
    sc, tid, notes, P, t0 = best
    t, mode = detect_key_melody(notes)
    while t0 - P > -1e-6:                 # start the grid at or before the first note
        t0 -= P
    K = int(math.ceil((notes[-1]['off'] - t0) / P)) + 1
    wav = os.path.join(base, 'Audio', f'vocadito_{tid}.wav')
    x, sr = sf.read(wav, dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    nl = [(n['on'], n['off'], n['midi']) for n in notes]
    m = Melody(f'vocadito_{tid}', nl, t0, P, 4, [k % 4 for k in range(K)], t, mode, x,
               f'vocadito track {tid} (Bittner et al.), notes from its A1 annotation; beat {P:.3f}s by estimate_beat; '
               f'pulse x key score {sc:.2f}', 'CC BY 4.0')
    return m


# ====================================================================== chord posteriors (pop-trained model)
_GRU = None


def chord_logp(mel, style='pop'):
    global _GRU
    song = mel.beat_song()
    path = os.path.join(HERE, 'models', 'bigru.npz')
    if os.path.exists(path):
        if _GRU is None:
            _GRU = NpBiGRU(path)
        return _GRU.logp(features(song, style)), 'bigru'
    data = load_all()
    h = StyleHMM(0.5, 0.0).fit(data['pop']['train'])
    post = fb(np.log(h.init), h.ll(song, 1.0), trans_mats(h, song))
    return np.log(post + 1e-9), 'style_hmm'


def spans(mel):
    """Half bars in 4/4 (2 beats), whole bars otherwise; aligned to the bar grid."""
    L = 2 if mel.M in (2, 4) else mel.M
    out, k = [], 0
    while k < mel.K:
        e = k + 1
        while e < mel.K and (mel.pos[e] % L) != 0 and e - k < L:
            e += 1
        out.append((k, e))
        k = e
    return out


def clash_time(mel, a, b, pcs, rule='avoid'):
    t = 0.0
    for ov, m in mel.her_in(a, b):
        for p in pcs:
            d = (m - p) % 12
            if (rule == 'avoid' and d == 1) or (rule == 'harsh' and d in HARSH):
                t += ov
                break
    return t


def pop_chords(mel, mu=4.0, mu_h=1.5, split_pen=1.5):
    """Diatonic triads per half bar from the pop-trained chord model, split into two one-beat chords
    when that fits her melody clearly better; penalties for avoid-note and harsh time against her."""
    lp, src = chord_logp(mel, 'pop')
    t = mel.rel_tonic

    def score(k, e, z):
        pcs = [(t + MAJOR[z] + i) % 12 for i in IV[TRIAD[z]]]
        a, b = mel.tb(k), mel.tb(e)
        return lp[k:e, z].sum() - (mu * clash_time(mel, a, b, pcs) + mu_h * clash_time(mel, a, b, pcs, 'harsh')) / mel.beat_s

    ch = []
    for k, e in spans(mel):
        one = max((score(k, e, z), z) for z in range(7))
        if e - k == 2:
            two = [max((score(j, j + 1, z), z) for z in range(7)) for j in (k, k + 1)]
            if two[0][1] != two[1][1] and two[0][0] + two[1][0] - split_pen > one[0]:
                ch += [(k, k + 1, two[0][1]), (k + 1, e, two[1][1])]
                continue
        ch.append((k, e, one[1]))
    return ch, src


# ====================================================================== styles -> events
def ev(on, off, midi, voice, vel=0.7):
    return {'on': float(on), 'off': float(off), 'midi': int(midi), 'voice': voice, 'vel': vel}


def near(pc, target):
    """MIDI note with pitch class pc nearest to target."""
    return int(target + ((pc - target + 6) % 12) - 6)


def style_bach(mel):
    notes = [(on, off - on, m) for on, off, m in mel.notes]
    out = satb_harmonize(notes, tonic=mel.tonic, mode=mel.mode)
    evs = []
    for v in ('A', 'T', 'B'):
        cur = None
        for s in out['slices']:
            m, on, off = s[v], s['on'], s['on'] + s['dur']
            if cur is not None and cur['midi'] == m and on - cur['off'] < 0.05:
                cur['off'] = off                           # a held voice
                continue
            cur = ev(on, off, m, v, 0.6)
            evs.append(cur)
    labels = [s['label'] for s in out['slices']]
    return {'events': evs, 'perc': [], 'chords': labels, 'key': out['key_name']}


def style_pop(mel):
    ch, src = pop_chords(mel)
    t = mel.rel_tonic
    evs, perc, labels = [], [], []
    top = min(m for _, _, m in mel.notes)
    hi = min(max(top - 2, 60), 72)
    for k, e, z in ch:
        a, b = mel.tb(k), mel.tb(e)
        pcs = [(t + MAJOR[z] + i) % 12 for i in IV[TRIAD[z]]]
        labels.append(['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°'][z])
        root = near(pcs[0], 40 if pcs[0] >= 4 else 48)
        root = root - 12 if root > 47 else root
        # bass on beats 1 and 3 (each span start, and the middle of a whole-bar span)
        for kk in range(k, e, 2 if mel.M in (2, 4) else e - k):
            evs.append(ev(mel.tb(kk), min(b, mel.tb(kk + 1.5)) - 0.02, root, 'bass', 0.8))
        # piano: a soft block chord on the span start (one beat) + broken-chord eighths (no pedal)
        chord = sorted(near(p, hi - 5) for p in pcs)
        chord = [c - 12 if c > hi else c for c in chord]
        for c in chord:
            evs.append(ev(a, min(b, mel.tb(k + 1)) - 0.03, c, 'piano', 0.45))
        n8 = 2 * (e - k)
        arp = [chord[0], chord[1], chord[2], chord[1]]
        for j in range(2, n8):
            on = a + j * mel.beat_s / 2
            evs.append(ev(on, on + mel.beat_s / 2 - 0.02, arp[j % 4] + (12 if j % 4 == 2 else 0), 'piano', 0.3))
    for k in range(mel.K):
        for j in (0, 0.5):
            perc.append(('shaker', mel.tb(k + j), 0.25 if j else 0.4))
    return {'events': evs, 'perc': perc, 'chords': labels, 'chord_src': src}


def jazz_plan(mel):
    base, src = pop_chords(mel, mu=4.0)
    t = mel.rel_tonic
    plan = []                                         # (on_beat, off_beat, root_pc_abs, quality, kind)
    nsub = 0
    for i, (k, e, z) in enumerate(base):
        root = (t + MAJOR[z]) % 12
        default = (k, e, root, SEVENTH[z], 'diatonic7')
        nxt = base[i + 1] if i + 1 < len(base) else None
        a, b = mel.tb(k), mel.tb(e)

        def ct(r, q, a=a, b=b):
            return clash_time(mel, a, b, [(r + x) % 12 for x in IV[q]])
        d_ct = ct(root, SEVENTH[z])
        if d_ct > ct(root, TRIAD[z]) + 1e-6:
            default = (k, e, root, TRIAD[z], 'triad')
            d_ct = ct(root, TRIAD[z])
        chosen = [default]
        if nxt is not None and nxt[2] != z and nxt[2] != 6 and mel.pos[nxt[0]] == 0:
            tr = (t + MAJOR[nxt[2]]) % 12
            opts = ['tritone', 'iiV', 'secdom']
            r = nsub % 3
            opts = opts[r:] + opts[:r]
            for o in opts:
                if o == 'tritone':
                    cand = [(k, e, (tr + 1) % 12, '7', 'tritone_sub')]
                elif o == 'secdom':
                    cand = [(k, e, (tr + 7) % 12, '7', 'secondary_dominant')]
                else:
                    if e - k < 2:
                        continue
                    h = (k + e) // 2
                    cand = [(k, h, (tr + 2) % 12, 'm7', 'ii_of_ii-V'), (h, e, (tr + 7) % 12, '7', 'V_of_ii-V')]
                c_ct = sum(clash_time(mel, mel.tb(c[0]), mel.tb(c[1]), [(c[2] + x) % 12 for x in IV[c[3]]]) for c in cand)
                melt = sum(ov for ov, _ in mel.her_in(a, b)) or 1.0
                if c_ct <= d_ct + 1e-6 and c_ct / melt <= 0.10:
                    chosen = cand
                    nsub += 1
                    break
        plan += chosen
    return plan, src


def voicing(mel, a, b, root, q, prev, cap):
    """Rootless voicing (3-7 with 9/13/5) below her, closest to the previous one, fewest avoid clashes."""
    if q not in ROOTLESS:                                # a triad fallback: close position
        sets = [IV[q]]
    else:
        sets = ROOTLESS[q]
    best = None
    for s in sets:
        pcs = [(root + x) % 12 for x in s]
        c = clash_time(mel, a, b, pcs) + 0.25 * clash_time(mel, a, b, pcs, 'harsh')
        for inv in range(len(s)):
            notes = []
            base = 50
            for j in range(len(s)):
                p = pcs[(inv + j) % len(s)]
                m = near(p, base if not notes else notes[-1] + 3)
                while notes and m <= notes[-1]:
                    m += 12
                notes.append(m)
            while notes[-1] > cap:
                notes = [x - 12 for x in notes]
            while notes[0] < 48:
                notes = [x + 12 for x in notes]
            if notes[-1] > cap + 2:
                continue
            mv = 0 if prev is None else sum(min(abs(x - y) for y in prev) for x in notes)
            sc = 10 * c + 0.05 * mv
            if best is None or sc < best[0]:
                best = (sc, notes)
    return best[1] if best else [near((root + x) % 12, 55) for x in IV[q][:3]]


COMP = [((0.0, 0.55), (1.667, 1.2)), ((0.667, 0.3), (2.667, 1.0)), ((0.0, 1.4), (2.667, 1.1)), ((1.0, 0.5), (1.667, 0.3), (3.0, 0.8))]


def style_jazz(mel):
    plan, src = jazz_plan(mel)
    evs, perc, labels = [], [], []
    top = min(m for _, _, m in mel.notes)
    cap = min(max(top - 2, 58), 70)
    prev = None
    by_beat = {}
    for c in plan:
        for kk in range(c[0], c[1]):
            by_beat[kk] = c
        nm = {'maj7': 'maj7', '7': '7', 'm7': 'm7', 'm7b5': 'm7b5', 'maj': '', 'm': 'm', 'dim': 'dim'}[c[3]]
        labels.append(f"{['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B'][c[2]]}{nm}:{c[4]}")
    # comping: swung chord stabs, per bar a rhythm from COMP
    bar_starts = [k for k in range(mel.K) if mel.pos[k] == 0]
    if not bar_starts or bar_starts[0] > 0:
        bar_starts = [0] + bar_starts
    for bi, kb in enumerate(bar_starts):
        for off_b, dur_b in COMP[bi % len(COMP)]:
            kk = kb + off_b
            if kk >= mel.K:
                continue
            c = by_beat.get(int(kk))
            if c is None:
                continue
            a = mel.tb(kk)
            b = min(mel.tb(kk + dur_b), mel.tb(c[1]))
            if b - a < 0.06:
                continue
            vo = voicing(mel, a, b, c[2], c[3], prev, cap)
            prev = vo
            for m in vo:
                evs.append(ev(a, b - 0.02, m, 'comp', 0.4))
    # walking bass: one note per beat; first beat of a chord = root, last beat = approach to the next root
    last = None
    for k in range(mel.K):
        c = by_beat[k]
        a, b = mel.tb(k), mel.tb(k + 1)
        nxt = by_beat.get(k + 1)
        root = c[2]
        if k == c[0]:
            cands = [root]
        elif nxt is not None and nxt is not c and k == c[1] - 1:
            nr = nxt[2]
            cands = [(nr + 1) % 12, (nr - 1) % 12, (nr + 7) % 12, (root + IV[c[3]][2]) % 12]
        else:
            cands = [(root + IV[c[3]][1]) % 12, (root + IV[c[3]][2]) % 12, (root + 2) % 12, root]
        best = None
        for i, p in enumerate(cands):
            sc = 3 * clash_time(mel, a, b, [p]) + clash_time(mel, a, b, [p], 'harsh') + 0.02 * i * mel.beat_s
            if best is None or sc < best[0] - 1e-9:
                best = (sc, p)
        m = near(best[1], last if last is not None else 40)
        while m > 52:
            m -= 12
        while m < 33:
            m += 12
        last = m
        evs.append(ev(a, b - 0.03, m, 'bass', 0.75))
    # swung ride: 1, 2, 2a, 3, 4, 4a
    for k in range(mel.K):
        perc.append(('ride', mel.tb(k), 0.5))
        if k % 2 == 1:
            perc.append(('ride', mel.tb(k + 0.667), 0.3))
    return {'events': evs, 'perc': perc, 'chords': labels, 'chord_src': src}


# ====================================================================== synthesis
def tone(midi, dur, kind, vel):
    f = midi_to_hz(midi)
    tail = {'piano': 0.5, 'comp': 0.25, 'bass': 0.15, 'organ': 0.08}[kind]
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    if kind == 'piano':
        for h, amp in enumerate([1, 0.45, 0.25, 0.12, 0.08, 0.04], 1):
            y += amp * np.exp(-t * (1.2 + 0.9 * h) * (f / 260) ** 0.3) * np.sin(2 * np.pi * h * f * t * (1 + 0.0004 * h * h))
    elif kind == 'comp':                                 # electric-piano-like (tine + bell)
        y = np.sin(2 * np.pi * f * t + 1.2 * np.exp(-t * 6) * np.sin(2 * np.pi * f * t)) * np.exp(-t * 1.5)
        y += 0.25 * np.sin(2 * np.pi * 7 * f * t) * np.exp(-t * 18)
    elif kind == 'bass':
        for h, amp in enumerate([1, 0.6, 0.3, 0.15], 1):
            y += amp * np.sin(2 * np.pi * h * f * t) * np.exp(-t * (2.5 + 1.5 * h))
    else:                                                # soft organ (for a Bach stand-in)
        for h, amp in enumerate([1, 0.5, 0.3, 0.1], 1):
            y += amp * np.sin(2 * np.pi * h * f * t)
    att = np.minimum(1, t / (0.004 if kind != 'organ' else 0.04))
    rel = np.clip((dur + tail - t) / tail, 0, 1) if kind != 'piano' else np.where(t > dur, np.exp(-(t - dur) * 12), 1.0)
    return vel * y * att * rel


def perc_hit(kind, rng):
    if kind == 'ride':
        n = int(0.6 * SR)
        x = colored_noise(n, rng, 0.6) * np.exp(-np.arange(n) / SR * 7)
        x += 0.3 * np.sin(2 * np.pi * 5200 * np.arange(n) / SR) * np.exp(-np.arange(n) / SR * 9)
        return 0.05 * x
    n = int(0.06 * SR)
    x = np.diff(colored_noise(n + 1, rng, 0.8)) * np.exp(-np.arange(n) / SR * 60)
    return 0.05 * x / (np.abs(x).max() + 1e-9)


def render_style(mel, style, out, her=None):
    T = max(max(e['off'] for e in out['events']), mel.notes[-1][1]) + 1.5
    acc = np.zeros(int(T * SR) + SR)
    rng = np.random.default_rng(7)
    if style == 'bach':
        for v, lvl in (('A', 0.5), ('T', 0.45), ('B', 0.55)):
            nl = [(e['on'], e['off'], e['midi']) for e in out['events'] if e['voice'] == v]
            if not nl:
                continue
            y, _ = singer.render(nl, seed={'A': 11, 'T': 12, 'B': 13}[v], tail=0.5)
            acc[:len(y)] += lvl * y[:len(acc)]
    else:
        for e in out['events']:
            kind = 'piano' if e['voice'] == 'piano' else e['voice']
            y = tone(e['midi'], e['off'] - e['on'], kind, e['vel'])
            i = int(e['on'] * SR)
            acc[i:i + len(y)] += 0.25 * y[:len(acc) - i]
        for kind, t0, vel in out['perc']:
            y = perc_hit(kind, rng) * vel
            i = int(t0 * SR)
            if i < len(acc):
                acc[i:i + len(y)] += y[:len(acc) - i]
    if her is None:
        her = mel.audio if mel.audio is not None else singer.render(mel.notes, seed=3)[0]
    n = max(len(acc), len(her))
    mix = np.zeros(n)
    acc_r = np.sqrt(np.mean(acc ** 2)) + 1e-9
    her_r = np.sqrt(np.mean(her[her != 0] ** 2)) + 1e-9 if np.any(her) else 1.0
    mix[:len(her)] += her / her_r * 0.1
    mix[:len(acc)] += acc / acc_r * 0.1 * (0.75 if style != 'bach' else 0.8)
    mix *= 0.89 / (np.abs(mix).max() + 1e-9)
    return mix.astype(np.float32), her


def to_mp3(x, path, kbps=96):
    import soundfile as sf
    wav = path[:-4] + '.tmp.wav'
    sf.write(wav, x, SR, subtype='PCM_16')
    subprocess.run([LAME, '--silent', '-m', 'm', '-b', str(kbps), wav, path], check=True)
    os.remove(wav)
    return os.path.getsize(path)


# ====================================================================== style statistics (A4.1 proxy)
def stats(mel, style, out):
    evs = out['events']
    dur = mel.notes[-1][1] - mel.notes[0][0]
    onsets = sorted({round(e['on'], 3) for e in evs})
    off_grid = []
    for o in onsets:
        q = (o - mel.t0) / mel.beat_s
        off_grid.append(q - math.floor(q + 1e-6))
    off_grid = np.array(off_grid)
    # chord vocabulary from simultaneous pitch-class sets at each onset
    sev = 0
    subs = 0
    labels = out['chords']
    if style == 'jazz':
        sev = sum(1 for l in labels if any(q in l for q in ('maj7', '7', 'm7'))) / max(len(labels), 1)
        subs = sum(1 for l in labels if any(q in l for q in ('tritone', 'secondary', 'ii-V'))) / max(len(labels), 1)
    elif style == 'bach':
        sev = sum(1 for l in labels if '7' in str(l)) / max(len(labels), 1)
    return {'onsets_per_s': len(onsets) / dur, 'swung_offbeat_share': float(np.mean(np.abs(off_grid - 0.667) < 0.03)),
            'straight_eighth_share': float(np.mean(np.abs(off_grid - 0.5) < 0.03)),
            'seventh_chord_share': sev, 'substitution_share': subs, 'chords': len(labels),
            'chords_per_beat': len(labels) / mel.K,
            'mean_note_beats': float(np.mean([(e['off'] - e['on']) / mel.beat_s for e in evs])),
            'voices_max': int(max(sum(1 for e in evs if e['on'] <= t < e['off']) for t in onsets)) if onsets else 0}


# ====================================================================== main
def melodies():
    data = load_all()
    out = []
    bach = [s for s in data['bach']['test'] if s['M'] == 4]
    out.append(from_dataset('bach_test_' + bach[0]['id'].split('/')[-1], bach[0], 0, min(32, bach[0]['K']), 76,
                            source=f"Bach chorale {bach[0]['id']} soprano (music21 corpus, test split), synthetic singer",
                            licence='public domain'))
    folk = [s for s in data['folk']['test'] if s['M'] == 4 and s['K'] >= 32] or [s for s in data['folk']['test'] if s['K'] >= 24]
    f = folk[0]
    bpm = 100 if f['beat_q'] <= 1 else 66
    out.append(from_dataset('folk_' + f['id'].split('/')[-1].replace('.abc#', '_'), f, 0, min(32, f['K']), bpm,
                            source=f"Nottingham folk {f['id']} (test split), synthetic singer", licence='GPL (Nottingham dataset)'))
    p = data['pop909']['test'][0]
    first = int(math.floor(p['sop'][0][0]))
    b0 = next(k for k in range(first, p['K']) if p['beats'][k]['pos'] == 0)
    out.append(from_dataset('pop909_' + p['id'].split('/')[-1], p, b0, 32, p.get('tempo_bpm', 90),
                            source=f"POP909 {p['id']} melody, bars from beat {b0} (test split), synthetic singer", licence='MIT'))
    out.append(vocadito_melody())
    return out


def main():
    _selftest_metrics()
    mels = melodies()
    res = {'melodies': {}, 'stats': {}}
    rendered = {}
    for mel in mels:
        row = {'source': mel.source, 'licence': mel.licence, 'tonic': mel.tonic, 'mode': mel.mode,
               'beat_s': mel.beat_s, 'K': mel.K, 'seconds': mel.notes[-1][1] - mel.notes[0][0]}
        her_audio = None
        for style, fn in (('bach', style_bach), ('pop', style_pop), ('jazz', style_jazz)):
            o = fn(mel)
            h = no_clash(mel.notes, o['events'], 'harsh')
            a = no_clash(mel.notes, o['events'], 'avoid')
            row[style] = {'no_clash_harsh': h['all'], 'no_clash_avoid': a['all'], 'per_voice_harsh': h['per_voice'],
                          'per_voice_avoid': a['per_voice'], 'chords': o['chords'][:64], 'chord_src': o.get('chord_src', 'satb'),
                          'key': o.get('key')}
            res['stats'].setdefault(style, []).append(stats(mel, style, o))
            x, her_audio = render_style(mel, style, o, her_audio)
            rendered[(mel.name, style)] = x
            print(mel.name, style, f"harsh {h['all']:.3f} avoid {a['all']:.3f}", flush=True)
        res['melodies'][mel.name] = row
    # Bach's own no-clash on the same chorale (soprano-onset slices of harmony/data/bach_satb.json)
    bs = json.load(open(os.path.join(REPO, 'harmony', 'data', 'bach_satb.json')))['chorales']
    bname = mels[0].name.replace('bach_test_', 'bach/')
    rec = next((c for c in bs if c['id'] == bname), None)
    if rec:
        sl = rec['notes']
        her = [(s[0], s[0] + s[1], s[2]) for s in sl if s[0] < 32]
        evs = [ev(s[0], s[0] + s[1], s[i], v) for s in sl if s[0] < 32 for i, v in ((4, 'A'), (5, 'T'), (6, 'B'))]
        res['bach_own_reference'] = {'chorale': bname, 'no_clash_harsh': no_clash(her, evs, 'harsh'),
                                     'note': 'Bach real A/T/B vs soprano, soprano-onset slices, same 32 beats (beat units)'}
    # robustness: the same three arrangers on 10 more test melodies per symbolic dataset (32 beats each, no audio)
    data = load_all()
    rob = {}
    for ds in ('bach', 'folk', 'pop909'):
        for i, song in enumerate([x for x in data[ds]['test'] if x['K'] >= 24][1:11]):
            first = int(math.floor(song['sop'][0][0]))
            b0 = next((k for k in range(first, song['K']) if song['beats'][k]['pos'] == 0), first)
            mel = from_dataset(f'{ds}_{i}', song, b0, min(32, song['K'] - b0), 90)
            if len(mel.notes) < 6:
                continue
            for style, fn in (('bach', style_bach), ('pop', style_pop), ('jazz', style_jazz)):
                o = fn(mel)
                for rule in ('harsh', 'avoid'):
                    rob.setdefault(style, {}).setdefault(rule, []).append(no_clash(mel.notes, o['events'], rule)['all'])
    res['robustness_30_test_melodies'] = {st_: {r: {'mean': float(np.mean(v)), 'min': float(np.min(v)), 'n': len(v),
                                                   'share_ge_0.90': float(np.mean(np.array(v) >= 0.90))}
                                               for r, v in d.items()} for st_, d in rob.items()}
    print('robustness', json.dumps(res['robustness_30_test_melodies']), flush=True)
    # summary per style (time-weighted over the 4 melodies)
    summ = {}
    for style in ('bach', 'pop', 'jazz'):
        w = [res['melodies'][m.name]['seconds'] for m in mels]
        for rule in ('harsh', 'avoid'):
            v = [res['melodies'][m.name][style][f'no_clash_{rule}'] for m in mels]
            summ.setdefault(style, {})[rule] = {'mean': float(np.average(v, weights=w)), 'min': float(min(v))}
        summ[style]['stats_mean'] = {k: float(np.mean([s[k] for s in res['stats'][style]])) for k in res['stats'][style][0]}
    res['summary'] = summ
    # demos (page-safe): 2 melodies, the three styles back to back
    demos, src_lines = [], []
    gap = np.zeros(int(0.8 * SR), np.float32)
    for mel in (mels[1], mels[3], mels[0]):
        seq = []
        for style in ('bach', 'pop', 'jazz'):
            seq += [rendered[(mel.name, style)], gap]
        fn = os.path.join(DEMO, f'{mel.name}_bach_pop_jazz.mp3')
        sz = to_mp3(np.concatenate(seq), fn, 64 if sum(len(s) for s in seq) / SR > 150 else 96)
        demos.append({'file': os.path.relpath(fn, REPO), 'bytes': sz, 'melody': mel.name, 'source': mel.source, 'licence': mel.licence})
        src_lines.append(f"- {os.path.basename(fn)}: {mel.source}; licence {mel.licence}"
                         + ('; attribution: vocadito (Bittner et al.), CC BY 4.0' if 'vocadito' in mel.name else '')
                         + '; order: Bach, pop, jazz (0.8 s gaps); accompaniment synthesized by Duet D4.')
    open(os.path.join(DEMO, 'SOURCES.md'), 'w').write('# D4 demo sources\n\n' + '\n'.join(src_lines) + '\n')
    res['demos'] = demos
    # blind kit: 4 melodies x 3 styles, random ids
    rnd = random.Random(20261030)
    items = [(m.name, s) for m in mels for s in ('bach', 'pop', 'jazz')]
    ids = rnd.sample(range(100, 1000), len(items))
    order = list(range(len(items)))
    while True:                                   # no melody twice in a row
        rnd.shuffle(order)
        if all(items[order[j]][0] != items[order[j + 1]][0] for j in range(len(order) - 1)):
            break
    key = {}
    for j, i in enumerate(order):
        name, style = items[i]
        cid = f'clip_{ids[j]}'
        to_mp3(rendered[(name, style)], os.path.join(BLIND, f'{cid}.mp3'), 64)
        key[cid] = {'melody': name, 'style': style, 'play_order': j + 1}
    json.dump(key, open(os.path.join(BLIND, 'KEY.json'), 'w'), indent=1)
    listing = '\n'.join(f'| {j + 1} | clip_{ids[j]}.mp3 | |' for j in range(len(items)))
    open(os.path.join(BLIND, 'INSTRUCTIONS.md'), 'w').write(
        '# 换个风格：盲听测试\n\n'
        '1. 请听者不要打开 KEY.json（只给主持人看）。\n'
        '2. 按下表顺序播放 12 段录音：4 段不同的人声旋律，每段旋律各配 3 种伴奏风格，顺序已打乱。\n'
        '3. 每段听完，在“风格”一栏写 巴赫(Bach) / 流行(pop) / 爵士(jazz) 之一。可以重听，不能回改。\n'
        '4. 收齐后，主持人用 KEY.json 对答案：猜对的比例 >= 80% 算通过 A4.1。\n\n'
        '| 顺序 | 文件 | 风格（巴赫 / 流行 / 爵士） |\n|---|---|---|\n' + listing + '\n')
    json.dump(res, open(os.path.join(OUT, 'styles.json'), 'w'), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    return res


if __name__ == '__main__':
    main()
