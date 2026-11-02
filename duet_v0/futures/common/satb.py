"""Four-part (SATB) harmonizer for Duet: her melody is the soprano, Duet writes alto, tenor, bass.

    import sys; sys.path.insert(0, 'futures/common'); from satb import harmonize
    out = harmonize(notes, tonic=None, mode=None)
    # notes: [(onset_s, dur_s, midi), ...] monophonic, sorted
    # out = {'tonic': pc, 'mode': 'major'|'minor', 'key_name': 'G major',
    #        'slices': [{'on','dur','S','A','T','B','chord_pcs','label'}, ...]}  one per melody note

Method (numpy only, a few ms per song):
  1. Key: harmony/model.detect_key_melody (Aarden-Essen + final-tonic bonus, 83 % on test).
  2. Chord Viterbi over the whole phrase (full look-ahead). State = (chord pitch-class set,
     bass pc) relative to the tonic, learned from the train split of harmony/data/bach_satb.json.
     Scores: log P(state | prev state) (bigram, back-off to unigram), log P(soprano pc | state),
     a phrase-end term log P(state | phrase end) / P(state), a phrase-start term, and a penalty for
     chord tones that are harsh against her note (the no-clash line).
  3. Voicing DP over candidate (A, T, B) voicings of the chosen chords with costs for voice-leading
     distance, leaps, parallel 5ths/8ves between any two voices including hers (hard ban),
     crossing/overlap, spacing (S-A, A-T <= 12), ranges, doubling, missing chord tones, and harsh
     intervals against her note (hard-ish ban).
Tables are built from the data on first import (about 1 s) and cached in satb_tables.npz.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'harmony', 'data', 'bach_satb.json')
SPLIT = os.path.join(ROOT, 'harmony', 'data', 'split.json')
CACHE = os.path.join(HERE, 'satb_tables.json')
sys.path.insert(0, os.path.join(ROOT, 'harmony'))
from model import detect_key_melody  # noqa: E402

NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
HARSH = np.zeros(12, bool)
HARSH[[1, 2, 6, 10, 11]] = True
RANGE = {'A': (55, 74), 'T': (48, 69), 'B': (40, 62)}
GAP_PHRASE = 0.10           # s; a gap at least this long before the next note ends a phrase

# tunable weights (chosen on the val split, see futures/d1_record_harmonize/tune.py)
P = dict(
    w_trans=1.3, w_emit=0.0, w_end=1.0, w_start=0.5, w_clash=6.0, w_par8=3.0, w_par5=0.5, min_count=3, alpha=2.0, w_ctx=1.0, k_ctx=5.0,
    # voicing
    v_move=0.08, v_leap=0.8, v_bass_move=0.04, v_bass_leap=0.6, v_missing=2.0, v_missing5=0.5,
    v_double_lt=2.5, v_double3=0.6, v_harsh=40.0, v_parallel=1e4, v_overlap=1.5, v_cross=1e3,
    v_unison=1.0, v_tb_wide=1.0, v_center=0.01, v_hidden=0.6, v_hold=0.0, v_bound=1.5,
)


# ---------------------------------------------------------------- data -> tables
def _phrase_flags(recs):
    """end/start flags per slice from fermatas (training) ."""
    n = len(recs)
    end = [bool(r[3]) for r in recs]
    end[-1] = True
    start = [True] + end[:-1]
    return end, start


def build_tables(ids=None):
    data = json.load(open(DATA))['chorales']
    split = json.load(open(SPLIT))
    keep = set(split['train'] if ids is None else ids)
    T = {}
    for mode in ('major', 'minor'):
        uni, big, emit, endc, startc = Counter(), Counter(), Counter(), Counter(), Counter()
        ctx3, ctx2, ctx1 = Counter(), Counter(), Counter()
        for ch in data:
            if ch['id'] not in keep or ch['mode'] != mode:
                continue
            t = ch['tonic']
            recs = ch['notes']
            end, start = _phrase_flags(recs)
            prev = None
            for k, r in enumerate(recs):
                if None in r[2:7] or not r[7]:
                    prev = None
                    continue
                pcs = tuple(sorted({(p - t) % 12 for p in r[7]}))
                st = (pcs, (r[6] - t) % 12)
                uni[st] += 1
                emit[(st, (r[2] - t) % 12)] += 1
                sp = (r[2] - t) % 12
                pv = (recs[k - 1][2] - t) % 12 if k > 0 and recs[k - 1][2] is not None and not end[k - 1] else 12
                nx = (recs[k + 1][2] - t) % 12 if k + 1 < len(recs) and recs[k + 1][2] is not None and not end[k] else 12
                ctx3[(pv, sp, nx, st)] += 1
                ctx2[(sp, nx, st)] += 1
                ctx1[(sp, st)] += 1
                if end[k]:
                    endc[st] += 1
                if start[k]:
                    startc[st] += 1
                if prev is not None:
                    big[(prev, st)] += 1
                prev = st
        states = [s for s, c in uni.most_common() if c >= P['min_count']]
        T[mode] = dict(states=[[list(s[0]), s[1]] for s in states],
                       uni=[uni[s] for s in states],
                       big=[[states.index(a), states.index(b), c] for (a, b), c in big.items()
                            if a in states and b in states],
                       emit=[[states.index(s), pc, c] for (s, pc), c in emit.items() if s in states],
                       end=[endc[s] for s in states], start=[startc[s] for s in states],
                       ctx3=[[a, b, c, states.index(st), n] for (a, b, c, st), n in ctx3.items() if st in states],
                       ctx2=[[b, c, states.index(st), n] for (b, c, st), n in ctx2.items() if st in states],
                       ctx1=[[b, states.index(st), n] for (b, st), n in ctx1.items() if st in states])
    return T


class Tables:
    def __init__(self, T):
        self.m = {}
        for mode, d in T.items():
            S = [(tuple(p), b) for p, b in d['states']]
            n = len(S)
            uni = np.array(d['uni'], float)
            pu = uni / uni.sum()
            big = np.zeros((n, n))
            for a, b, c in d['big']:
                big[a, b] += c
            a = P['alpha']
            trans = (big + a * pu[None, :]) / (big.sum(1, keepdims=True) + a)
            emit = np.full((n, 12), 0.02)
            for s, pc, c in d['emit']:
                emit[s, pc] += c
            emit /= emit.sum(1, keepdims=True)
            end = (np.array(d['end'], float) + 0.5) / (sum(d['end']) + 0.5 * n)
            start = (np.array(d['start'], float) + 0.5) / (sum(d['start']) + 0.5 * n)
            # context classifier P(state | prev, cur, next soprano pc) with back-off (13 = none)
            k_ = P['k_ctx']
            c1 = np.zeros((12, n))
            for b, st, c in d['ctx1']:
                c1[b, st] += c
            p1 = (c1 + k_ * pu[None, :]) / (c1.sum(1, keepdims=True) + k_)
            c2 = np.zeros((12, 13, n))
            for b, c, st, cnt in d['ctx2']:
                c2[b, c, st] += cnt
            p2 = (c2 + k_ * p1[:, None, :]) / (c2.sum(2, keepdims=True) + k_)
            c3 = np.zeros((13, 12, 13, n))
            for a_, b, c, st, cnt in d['ctx3']:
                c3[a_, b, c, st] += cnt
            p3 = (c3 + k_ * p2[None, :, :, :]) / (c3.sum(3, keepdims=True) + k_)
            lctx = np.log(p3) - np.log(pu)[None, None, None, :]
            pcs_mask = np.zeros((n, 12), bool)
            for i, (p, b) in enumerate(S):
                pcs_mask[i, list(p)] = True
            self.m[mode] = dict(states=S, lpu=np.log(pu), ltrans=np.log(trans), lemit=np.log(emit),
                                lend=np.log(end) - np.log(pu), lstart=np.log(start) - np.log(pu), lctx=lctx,
                                mask=pcs_mask)


_TABLES = None


def tables():
    global _TABLES
    if _TABLES is None:
        if os.path.exists(CACHE):
            T = json.load(open(CACHE))
        else:
            T = build_tables()
            try:
                json.dump(T, open(CACHE, 'w'))
            except OSError:
                pass
        _TABLES = Tables(T)
    return _TABLES


# ---------------------------------------------------------------- chord Viterbi
def _clash_count(mask, spc):
    """number of chord pcs harsh against soprano pc, per state."""
    h = HARSH[(np.arange(12) - spc) % 12]
    return (mask & h[None, :]).sum(1)


def chord_viterbi(spcs, ends, starts, mode, p=None, moved=None, ban=(), return_idx=False):
    p = P if p is None else p
    tb = tables().m[mode]
    n, K = len(spcs), len(tb['states'])
    U = np.zeros((n, K))
    for k in range(n):
        U[k] = p['w_emit'] * tb['lemit'][:, spcs[k]] - p['w_clash'] * _clash_count(tb['mask'], spcs[k])
        U[k][~tb['mask'][:, spcs[k]]] = -1e9          # her note must be a chord tone
        if p['w_ctx']:
            pv = spcs[k - 1] if k > 0 and not ends[k - 1] else 12
            nx = spcs[k + 1] if k + 1 < n and not ends[k] else 12
            U[k] += p['w_ctx'] * tb['lctx'][pv, spcs[k], nx]
        if ends[k]:
            U[k] += p['w_end'] * tb['lend']
        if starts[k]:
            U[k] += p['w_start'] * tb['lstart']
    for k, i in ban:
        U[k, i] = -1e9
    W = p['w_trans'] * tb['ltrans']
    bass = np.array([b for _, b in tb['states']])
    delta = U[0] + tb['lpu']
    bp = np.zeros((n, K), int)
    for k in range(1, n):
        sc = delta[:, None] + W
        if (moved[k] if moved is not None else spcs[k] != spcs[k - 1]):
            # bass doubling her moving note on both slices (octaves) or a fifth below on both:
            # often forces parallels in the voicing stage, so discourage it here
            o0, o1 = bass == spcs[k - 1], bass == spcs[k]
            f0, f1 = (spcs[k - 1] - bass) % 12 == 7, (spcs[k] - bass) % 12 == 7
            sc = sc - p['w_par8'] * (o0[:, None] & o1[None, :]) - p['w_par5'] * (f0[:, None] & f1[None, :])
        bp[k] = sc.argmax(0)
        delta = sc.max(0) + U[k]
    path = [int(delta.argmax())]
    for k in range(n - 1, 0, -1):
        path.append(int(bp[k, path[-1]]))
    path = path[::-1]
    if return_idx:
        return [tb['states'][i] for i in path], path
    return [tb['states'][i] for i in path]


# ---------------------------------------------------------------- voicing DP
def _root_of(pcs, bass):
    """root pc of a pitch-class set (third-stacking), falls back to the bass."""
    s = set(pcs)
    best = None
    for r in s:
        score = (((r + 4) % 12 in s) or ((r + 3) % 12 in s)) * 2 + ((r + 7) % 12 in s) + ((r + 6) % 12 in s) * 0.5
        score += 0.1 * (r == bass)
        if best is None or score > best[0]:
            best = (score, r)
    return best[1] if best else bass


def candidates(s, pcs, bass_pc, root, tonic, p, ext=0):
    """all (A, T, B) for soprano midi s and chord pcs (absolute) with unary costs.
    Ranges follow her note when it leaves Bach's soprano register; `ext` widens them further."""
    pcs = set(pcs)
    up = max(0, s - 79) + ext
    dn = max(0, 62 - s) + ext
    RA = (RANGE['A'][0] - dn, RANGE['A'][1] + up)
    RT = (RANGE['T'][0] - dn, RANGE['T'][1] + up)
    RB = (RANGE['B'][0] - dn, RANGE['B'][1] + up // 2)
    out = []
    lt = (tonic + 11) % 12
    third = None
    for q in ((root + 4) % 12, (root + 3) % 12):
        if q in pcs:
            third = q
            break
    fifth = (root + 7) % 12
    Bs = [b for b in range(RB[0], RB[1] + 1) if b % 12 == bass_pc and b < s]
    Ts = [t for t in range(RT[0], RT[1] + 1) if t % 12 in pcs]
    As = [a for a in range(RA[0], RA[1] + 1) if a % 12 in pcs and a <= s and s - a <= 12]
    for B in Bs:
        for T in Ts:
            if T < B or T - B > 24:
                continue
            for A in As:
                if A < T or A - T > 12:
                    continue
                v = [s, A, T, B]
                c = 0.0
                have = {x % 12 for x in v}
                for q in pcs - have:
                    c += p['v_missing5'] if q == fifth else p['v_missing']
                cnt = Counter(x % 12 for x in v)
                if cnt[lt] > 1 and lt in pcs:
                    c += p['v_double_lt']
                if third is not None and cnt[third] > 1:
                    c += p['v_double3']
                for x in (A, T, B):
                    if HARSH[abs(s - x) % 12]:
                        c += p['v_harsh']
                if A == s:
                    c += p['v_unison']
                if A == T or T == B:
                    c += p['v_unison'] * 0.5
                if T - B > 19:
                    c += p['v_tb_wide']
                c += p['v_center'] * ((A - 64.5) ** 2 + (T - 58.5) ** 2 + (B - 51) ** 2) / 10
                out.append((A, T, B, c))
    return out


def _pair_cost(prev, cur, sp, sc, p):
    """transition cost between voicing prev=(A,T,B) and cur, with soprano sp -> sc."""
    v0 = (sp,) + prev
    v1 = (sc,) + cur
    c = 0.0
    for i in (1, 2):
        d = abs(v1[i] - v0[i])
        c += p['v_move'] * d + (p['v_leap'] * (d - 4) if d > 4 else 0.0)
        if d == 0:
            c += p['v_hold']
    d = abs(v1[3] - v0[3])
    c += p['v_bass_move'] * d + (p['v_bass_leap'] * (d - 7) if d > 7 else 0.0)
    for i in range(4):
        for j in range(i + 1, 4):
            mi, mj = v1[i] - v0[i], v1[j] - v0[j]
            if mi and mj and (mi > 0) == (mj > 0):
                ic0, ic1 = abs(v0[i] - v0[j]) % 12, abs(v1[i] - v1[j]) % 12
                if ic0 == ic1 and ic1 in (0, 7):
                    c += p['v_parallel']
                elif ic1 in (0, 7) and i == 0 and j == 3 and abs(mi) > 2:
                    c += p['v_hidden']
    # overlap: a lower voice moves above the previous note of the voice above (or vice versa)
    for i in range(3):
        if v1[i + 1] > v0[i] or v1[i] < v0[i + 1]:
            c += p['v_overlap']
    return c


def voice(sops, chords, tonic, p=None):
    p = P if p is None else p
    C, U = [], []
    for s, (pcs, b) in zip(sops, chords):
        root = _root_of(pcs, b)
        cand = candidates(s, pcs, b, root, tonic, p)
        for ext in (4, 8, 12):                          # widen the ranges, then try other bass notes
            if len(cand) >= 4:
                break
            cand = cand + [(a, t, bb, c + 0.5 * ext) for a, t, bb, c in candidates(s, pcs, b, root, tonic, p, ext)]
        if not cand:
            for b2 in pcs:
                cand += candidates(s, pcs, b2, root, tonic, p, 8)
            cand = [(a, t, bb, c + 3.0) for a, t, bb, c in cand]
        if not cand:                                    # last resort: a third and sixth below
            cand = [(s - 3, s - 8, s - 15, 50.0)]
        C.append(np.array([x[:3] for x in cand]))
        U.append(np.array([x[3] for x in cand]))
    n = len(sops)
    delta = U[0].copy()
    bps = []
    for k in range(1, n):
        P0, P1 = C[k - 1], C[k]
        M = np.zeros((len(P0), len(P1)))
        # vectorised transition cost
        v0 = np.concatenate([np.full((len(P0), 1), sops[k - 1]), P0], 1)[:, None, :]
        v1 = np.concatenate([np.full((len(P1), 1), sops[k]), P1], 1)[None, :, :]
        mv = v1 - v0                                          # (n0, n1, 4)
        d = np.abs(mv)
        for i in (1, 2):
            M += p['v_move'] * d[..., i] + p['v_leap'] * np.maximum(d[..., i] - 4, 0) + p['v_hold'] * (d[..., i] == 0)
        M += p['v_bass_move'] * d[..., 3] + p['v_bass_leap'] * np.maximum(d[..., 3] - 7, 0)
        for i in range(4):
            for j in range(i + 1, 4):
                both = (mv[..., i] != 0) & (mv[..., j] != 0) & ((mv[..., i] > 0) == (mv[..., j] > 0))
                ic0 = np.abs(v0[..., i] - v0[..., j]) % 12
                ic1 = np.abs(v1[..., i] - v1[..., j]) % 12
                par = both & (ic0 == ic1) & ((ic1 == 0) | (ic1 == 7))
                M += p['v_parallel'] * par
                if i == 0 and j == 3:
                    hid = both & ~par & ((ic1 == 0) | (ic1 == 7)) & (np.abs(mv[..., 0]) > 2)
                    M += p['v_hidden'] * hid
        for i in range(3):
            ov = (v1[..., i + 1] > v0[..., i]) | (v1[..., i] < v0[..., i + 1])
            M += p['v_overlap'] * ov
        if p['v_bound']:
            # robustness at the boundary: her real onset may come a little before or after the
            # detected one, so the old voice note should not be harsh against her new note, and the
            # new voice note not harsh against her old note (only matters for a voice that moves)
            for i in (1, 2, 3):
                h = HARSH[np.abs(v0[..., i] - v1[..., 0]) % 12] | HARSH[np.abs(v1[..., i] - v0[..., 0]) % 12]
                M += p['v_bound'] * (h & (mv[..., i] != 0))
        sc = delta[:, None] + M
        bp = sc.argmin(0)
        delta = sc[bp, np.arange(len(P1))] + U[k]
        bps.append(bp)
    idx = [int(delta.argmin())]
    for bp in bps[::-1]:
        idx.append(int(bp[idx[-1]]))
    idx = idx[::-1]
    return [tuple(int(x) for x in C[k][i]) for k, i in enumerate(idx)]


def _first_parallel(sops, V):
    """index k of the first slice that makes parallel 5ths/8ves with slice k-1 (BRIEF definition)."""
    for k in range(1, len(V)):
        v0 = (sops[k - 1],) + tuple(V[k - 1])
        v1 = (sops[k],) + tuple(V[k])
        for i in range(4):
            for j in range(i + 1, 4):
                mi, mj = v1[i] - v0[i], v1[j] - v0[j]
                if mi and mj and (mi > 0) == (mj > 0):
                    a, b = abs(v0[i] - v0[j]) % 12, abs(v1[i] - v1[j]) % 12
                    if a == b and a in (0, 7):
                        return k
    return None


# ---------------------------------------------------------------- labels
ROMAN = ['I', 'bII', 'II', 'bIII', 'III', 'IV', '#IV', 'V', 'bVI', 'VI', 'bVII', 'VII']


def label(pcs, bass, tonic):
    """rough Roman numeral, e.g. V6, ii, vii o6, V7."""
    root = _root_of(pcs, bass)
    s = {(q - root) % 12 for q in pcs}
    rn = ROMAN[(root - tonic) % 12]
    minor = 3 in s and 4 not in s
    dim = minor and 6 in s and 7 not in s
    if minor:
        rn = rn.lower()
    seventh = 10 in s or 11 in s
    inv = (bass - root) % 12
    if seventh:
        fig = {0: '7', 3: '65', 4: '65', 7: '43', 6: '43', 10: '42', 11: '42'}.get(inv, '7')
    else:
        fig = {0: '', 3: '6', 4: '6', 7: '64', 6: '64'}.get(inv, '')
    return rn + ('o' if dim else '') + fig


# ---------------------------------------------------------------- public API
def phrase_flags(notes):
    n = len(notes)
    ends = []
    for k in range(n):
        if k == n - 1:
            ends.append(True)
        else:
            gap = notes[k + 1][0] - (notes[k][0] + notes[k][1])
            ends.append(gap >= GAP_PHRASE)
    starts = [True] + ends[:-1]
    return ends, starts


def harmonize(notes, tonic=None, mode=None, params=None):
    p = dict(P)
    if params:
        p.update(params)
    notes = [(float(a), float(b), int(m)) for a, b, m in notes if int(m) > 0]
    if not notes:
        return {'tonic': 0, 'mode': 'major', 'key_name': 'C major', 'slices': []}
    if tonic is None or mode is None:
        t, md = detect_key_melody([{'on': a, 'off': a + b, 'midi': m} for a, b, m in notes])
        tonic = t if tonic is None else tonic
        mode = md if mode is None else mode
    # her register -> an octave shift so the soprano sits in Bach's range (median ~ 69-71)
    med = float(np.median([m for _, _, m in notes]))
    shift = int(round((70 - med) / 12.0)) * 12
    sops = [m + shift for _, _, m in notes]
    ends, starts = phrase_flags(notes)
    spcs = [(m - tonic) % 12 for m in sops]
    moved = [False] + [sops[k] != sops[k - 1] for k in range(1, len(sops))]
    ban = []
    for _ in range(8):                      # repair: ban a chord that forced a parallel, re-decode
        chords_rel, idx = chord_viterbi(spcs, ends, starts, mode, p, moved, ban, return_idx=True)
        chords = [(tuple(sorted((q + tonic) % 12 for q in pcs)), (b + tonic) % 12) for pcs, b in chords_rel]
        V = voice(sops, chords, tonic, p)
        bad = _first_parallel(sops, V)
        if bad is None:
            break
        ban += [(bad, idx[bad]), (bad - 1, idx[bad - 1])] if bad > 0 else [(bad, idx[bad])]
    out = []
    for (on, dur, m), s, (A, T, B) in zip(notes, sops, V):
        pcs = sorted({x % 12 for x in (s, A, T, B)})
        out.append({'on': on, 'dur': dur, 'S': m, 'A': A - shift, 'T': T - shift, 'B': B - shift,
                    'chord_pcs': pcs, 'label': label(pcs, B % 12, tonic)})
    return {'tonic': int(tonic), 'mode': mode, 'key_name': f'{NAMES[tonic]} {mode}', 'slices': out}


if __name__ == '__main__':
    import time
    mel = [(0.0, 0.5, 67), (0.5, 0.5, 69), (1.0, 0.5, 71), (1.5, 0.5, 72), (2.0, 1.0, 74), (3.0, 1.0, 71),
           (4.2, 0.5, 72), (4.7, 0.5, 71), (5.2, 0.5, 69), (5.7, 1.5, 67)]
    t0 = time.perf_counter()
    o = harmonize(mel)
    print(o['key_name'], f'{(time.perf_counter() - t0) * 1e3:.1f} ms')
    for s in o['slices']:
        print(s)
