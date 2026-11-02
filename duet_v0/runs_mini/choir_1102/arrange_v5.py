"""v5 of arrange_v4.py (v4 found no voicing for a v11 window whose melody leaps from C#5 down to E4: the S-A gap
may now widen to 19 semitones when 12 gives nothing). v4 of arrange_v3.py (v11 chords change less often: no chord shorter than 1.25 beats, so a singer's rubato of
a few tenths of a beat does not put a quick chord change in the wrong place; Twinkle unchanged). v3 of arrange_v2.py (v2's automatic v11 chords clashed with the melody 0.39 of the time; v3 uses chords written by
hand for v11, following the melody: I IV I IV I V IV I V I IV I V I). v2 of arrange_v1.py (v1 found no voicing for a 10-beat chord span: the melody there spans a 6th). v2 voices every
2-beat window on its own (a part holds its note when it stays), keeps S-A <= 12 against the window's first melody note
only, and makes keeping the same chord cost 0.3 so the chords move a little more. Four-part (SATB) arrangements for the choir rehearsal mode: Vincent sings the melody (S), Duet sings A, T, B.

Songs:
- twinkle: songs.twinkle() (C major, 48 beats), its hand-written chords per beat (I / IV / V).
- v11: the melody of futures/live_test/phone/test_her_v11_countin.wav (vocadito 11, A major, 122.45 bpm), transcribed
  offline (v11_transcribe_v1.py -> v11_notes_v1.json) and cleaned by hand here: scoops merged into the note they reach
  (G3 -> A3 at beat 0, Eb3 -> E3 at 20.5, Bb3 -> A3 at 22.25), 56.4 read as G#3 (leading tone), onsets on the
  quarter-beat grid. Chords: one per 2 beats from I ii iii IV V vi V7 by dynamic programming (melody fit + common
  progressions, start and end on I).
Voicing: dynamic programming over chord spans; S = the melody (v11 an octave up, so the parts sit in choir ranges),
A 55..72, T 48..67, B 40..60, S > A > T > B, S-A <= 12, A-T <= 12, bass mostly the root, root and third present,
small motion, no parallel fifths / octaves between any two voices. Each part holds its note for the chord span.
Writes futures/v4_selfiter/choir_songs_v5.json (new file) and prints checks.

  cd duet_v0 && .venv/bin/python runs_mini/choir_1102/arrange_v1.py
"""
import itertools
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'duet_beat'))
from songs import twinkle  # noqa: E402

OUT = os.path.join(ABS, 'futures/v4_selfiter/choir_songs_v5.json')
MAJOR = (0, 2, 4, 5, 7, 9, 11)
CH = {'I': (0, 4, 7), 'ii': (2, 5, 9), 'iii': (4, 7, 11), 'IV': (5, 9, 0), 'V': (7, 11, 2), 'vi': (9, 0, 4),
      'V7': (7, 11, 2, 5)}
GOOD = {('I', 'IV'), ('I', 'V'), ('I', 'vi'), ('I', 'ii'), ('IV', 'V'), ('IV', 'V7'), ('IV', 'I'), ('ii', 'V'),
        ('ii', 'V7'), ('V', 'I'), ('V7', 'I'), ('vi', 'ii'), ('vi', 'IV'), ('vi', 'V'), ('iii', 'vi'), ('iii', 'IV'),
        ('V', 'vi'), ('V', 'V7'), ('I', 'iii')}
V11_CHORDS = [[0.0, 5.25, 'I'], [5.25, 7.25, 'IV'], [7.25, 9.25, 'I'], [9.25, 10.5, 'IV'], [10.5, 13.25, 'V'],
              [13.25, 15.25, 'IV'], [15.25, 17.5, 'I'], [17.5, 19.0, 'IV'], [19.0, 22.25, 'V'], [22.25, 26.0, 'I']]
RANGE = {'A': (55, 72), 'T': (48, 67), 'B': (40, 60)}

V11 = [(0.0, 2.0, 57), (2.0, 3.25, 52), (3.25, 5.25, 61), (5.25, 7.0, 57), (7.0, 7.25, 61), (7.25, 9.25, 64),
       (9.25, 10.0, 62), (10.0, 10.5, 61), (10.5, 12.5, 59), (12.5, 13.0, 59), (13.0, 13.25, 61), (13.25, 14.5, 62),
       (14.5, 15.25, 62), (15.25, 16.0, 61), (16.0, 16.5, 59), (16.5, 17.5, 61), (17.5, 18.25, 57),
       (18.25, 19.0, 57), (19.0, 19.5, 61), (19.5, 20.5, 59), (20.5, 21.5, 52), (21.5, 22.25, 56),
       (22.25, 24.5, 57)]


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def choose_chords(mel, tonic, K, span=2.0):
    """DP over windows of `span` beats: chord names."""
    wins = [(w * span, min(K, (w + 1) * span)) for w in range(int((K + span - 1e-9) // span))]
    names = list(CH)

    def fit(w, c):
        pcs = {(tonic + d) % 12 for d in CH[c]}
        cost = 0.0
        for on, off, m in mel:
            ov = overlap(on, off, *w)
            if ov > 0 and m % 12 not in pcs:
                cost += ov * (1.5 if on >= w[0] - 1e-9 and on % 1 == 0 else 1.0)
        return cost + (0.15 if c in ('iii', 'V7') else 0.0)

    best = [{c: (fit(wins[0], c) + (0 if c == 'I' else 1.0), [c]) for c in names}]
    for w in wins[1:]:
        row = {}
        for c in names:
            opts = []
            for p, (cp, path) in best[-1].items():
                t = 0.3 if p == c else (0.0 if (p, c) in GOOD else 0.6)
                opts.append((cp + t + fit(w, c), path + [c]))
            row[c] = min(opts)
        best.append(row)
    end = {c: v[0] + (0 if c == 'I' else 2.0) for c, v in best[-1].items()}
    path = best[-1][min(end, key=end.get)][1]
    spans = []
    for w, c in zip(wins, path):
        if spans and spans[-1][2] == c:
            spans[-1][1] = w[1]
        else:
            spans.append([w[0], w[1], c])
    return spans


def voice(spans, mel_s, tonic):
    """DP voicing of A, T, B over chord spans; mel_s = melody at S pitch."""
    cands = []
    for on, off, c in spans:
        pcs = [(tonic + d) % 12 for d in CH[c]]
        root, third = pcs[0], pcs[1]
        smel = [m for a, b, m in mel_s if overlap(a, b, on, off) > 0] or [72]
        s_lo, s_hi, s_first = min(smel), max(smel), smel[0]
        rng = {v: [p for p in range(lo, hi + 1) if p % 12 in pcs] for v, (lo, hi) in RANGE.items()}
        opts = []
        for spread in (12, 19):                            # a window whose melody leaps wide may need a wider S-A gap
            for a, t, b in itertools.product(rng['A'], rng['T'], rng['B']):
                if not (s_lo > a > t > b) or s_first - a > spread or a - t > 12 or t - b > 19:
                    continue
                have = {a % 12, t % 12, b % 12, s_first % 12}
                if root not in have or third not in have:
                    continue
                cost = 0.0
                cost += 0 if b % 12 == root else (1.5 if b % 12 == third else 4.0)
                if c == 'V7' and pcs[3] not in have:
                    cost += 1.0
                if [a % 12, t % 12, b % 12, s_first % 12].count(third) > 1:
                    cost += 1.0
                cost += 0.05 * abs(a - 64) + 0.05 * abs(t - 57) + 0.03 * abs(b - 48)
                opts.append(((a, t, b), cost, s_first))
            if opts:
                break
        cands.append(opts)

    def trans(p, q, sp, sq):
        cost = 0.12 * (abs(p[0] - q[0]) + abs(p[1] - q[1])) + 0.06 * abs(p[2] - q[2])
        vp, vq = (sp,) + p, (sq,) + q
        for i, j in itertools.combinations(range(4), 2):
            ip, iq = (vp[i] - vp[j]) % 12, (vq[i] - vq[j]) % 12
            if ip == iq and ip in (0, 7) and vp[i] != vq[i]:
                cost += 5.0                                # parallel octave / fifth
        return cost

    dp = [{o[0]: (o[1], [o[0]]) for o in cands[0]}]
    sfirst = [cands[k][0][2] if cands[k] else None for k in range(len(cands))]
    for k in range(1, len(cands)):
        row = {}
        for q, cq, sq in cands[k]:
            best = min(((cp + trans(p, q, sfirst[k - 1], sq) + cq, path + [q]) for p, (cp, path) in dp[-1].items()),
                       key=lambda x: x[0])
            row[q] = best
        dp.append(row)
    total, path = min(dp[-1].values(), key=lambda x: x[0])
    parts = {'A': [], 'T': [], 'B': []}
    for (on, off, c), v in zip(spans, path):
        for name, p in zip('ATB', v):
            seq = parts[name]
            if seq and seq[-1][2] == p and abs(seq[-1][1] - on) < 1e-9:
                seq[-1][1] = off                           # same note into the next chord: hold it
            else:
                seq.append([on, off, p])
    return parts, total


def check(mel, parts, shift):
    """Share of melody time where some part is harsh (2nd / tritone / 7th) against her, at her pitch."""
    harsh, tot = 0.0, 0.0
    for on, off, m in mel:
        tot += off - on
        for seq in parts.values():
            for a, b, p in seq:
                ov = overlap(on, off, a, b)
                if ov > 0 and abs(p - (m + shift)) % 12 in (1, 2, 6, 10, 11):
                    harsh += ov
    return round(harsh / tot, 3)


def build(sid, title, key, tonic, mel, K, bpm, s_shift, spans=None):
    spans = spans or choose_chords(mel, tonic, K)
    mel_s = [(a, b, m + s_shift) for a, b, m in mel]
    wins = [[w, min(e, w + 2.0), c] for a, e, c in spans for w in [a + 2.0 * k for k in range(int((e - a + 1.999) // 2))]]
    parts, cost = voice(wins, mel_s, tonic)
    return {'id': sid, 'title': title, 'key': key, 'tonic': tonic, 'M': 4, 'K': K, 'bpm': bpm,
            'melody': [list(n) for n in mel], 'melody_s_shift': s_shift, 'chords': spans, 'parts': parts,
            'voicing_cost': round(cost, 2), 'harsh_share_vs_melody': check(mel_s, parts, 0)}


def main():
    tw = twinkle()
    z = [b['z'] for b in tw['beats']]
    roman = {0: 'I', 3: 'IV', 4: 'V'}
    spans = []
    for k, zz in enumerate(z):
        c = roman[zz]
        if spans and spans[-1][2] == c:
            spans[-1][1] = k + 1.0
        else:
            spans.append([float(k), k + 1.0, c])
    songs = {'twinkle': build('twinkle', 'Twinkle Twinkle Little Star', 'C', 0,
                              [tuple(n) for n in tw['sop']], 48, 80.0, 0, spans),
             'v11': build('v11', 'vocadito 11 (test_her_v11)', 'A', 9, V11, 26, 122.45, 12, V11_CHORDS)}
    for sid, s in songs.items():
        print(sid, 'chords', [(a, c) for a, b, c in s['chords']])
        for v in 'ATB':
            print('  ', v, [(a, p) for a, b, p in s['parts'][v]])
        print('  harsh share vs melody', s['harsh_share_vs_melody'], 'voicing cost', s['voicing_cost'])
    with open(OUT, 'x') as fh:
        json.dump({'written': 'run choir_1102, arrange_v1.py', 'songs': songs}, fh, indent=1)


if __name__ == '__main__':
    main()
