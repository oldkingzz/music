"""Listen, follow, predict, play.  (v2, 2026-10-15; v1 classes kept for comparison)

FeatureTracker  - per 10 ms: pitch (YIN), voicing, loudness, onset novelty. Causal. (unchanged)
offline_notes   - rehearsal-time note segmentation of the reference, vibrato-robust (v2).
align_score     - a symbolic score of the song aligned to her rehearsal by offline DTW (v2).
rehearsal_reference - the rehearsal reference from either source (audio alone, or score + audio).
plan_harmony    - rehearsal-time harmony line: a diatonic third (or sixth) below her. (unchanged)
Follower        - note-level HMM over the reference notes: where is she in the song? (v2)
Accompanist     - anchors her detected onsets to reference notes, predicts her next onsets with
                  phase correction on a slowly adapting tempo, and commits each machine note
                  early enough to land on time. (v2)
FollowerV1, AccompanistV1, offline_notes_v1 - the phase-1/2 versions, verbatim, for comparison.
What changed and what each change measured: eval/V2_CHANGES.md.
"""
import numpy as np

from dsp import FMAX, FMIN, HOP, SR, WIN, hz_to_midi, yin

INF = 1e18
CONTENT_DELAY = WIN / 2 / SR      # a frame describes the audio 20 ms before "now"


# ---------------------------------------------------------------- features
class FeatureTracker:
    def __init__(self):
        self.buf = np.zeros(WIN)
        self.i = -1
        self.peak_db = -80.0
        self.m, self.v, self.db, self.nov = [], [], [], []
        self.raw = []

    def step(self, chunk):
        """Feed HOP new samples. Returns the frame index."""
        self.buf = np.concatenate([self.buf[len(chunk):], chunk])
        self.i += 1
        f0, ap, rms = yin(self.buf)
        db = 20 * np.log10(rms + 1e-9)
        self.peak_db = max(db, self.peak_db - 0.02)          # forgets ~2 dB per second
        voiced = ap < 0.25 and db > self.peak_db - 30 and db > -60 and FMIN <= f0 <= FMAX
        m = float(hz_to_midi(f0)) if voiced else np.nan
        self.raw.append(m)
        if voiced and len(self.v) >= 2 and self.v[-1] and self.v[-2]:
            m = float(np.median(self.raw[-3:]))
        self.m.append(m)
        self.v.append(voiced)
        self.db.append(db)
        self.nov.append(self._novelty())
        return self.i

    def _novelty(self):
        i = self.i
        if not self.v[i]:
            return 0.0
        if i >= 3 and not any(self.v[i - 3:i]):
            return 1.0                                       # voice starts after silence
        p = 0.0
        if i >= 4 and self.v[i - 4]:
            p = np.clip((abs(self.m[i] - self.m[i - 4]) - 0.5) / 1.5, 0.0, 1.0)
        e = np.clip((self.db[i] - min(self.db[max(0, i - 8):i + 1]) - 3.0) / 9.0, 0.0, 1.0)
        return float(max(p, e))

    def t(self, i):
        """Audio time (s) that frame i describes."""
        return (i + 1) * HOP / SR - CONTENT_DELAY

    def arrays(self):
        m = np.array(self.m)
        v = np.array(self.v)
        return {'t': np.array([self.t(i) for i in range(len(m))]), 'midi': np.nan_to_num(m),
                'voiced': v, 'db': np.array(self.db), 'nov': np.array(self.nov)}


def analyse(x):
    ft = FeatureTracker()
    for k in range(len(x) // HOP):
        ft.step(x[k * HOP:(k + 1) * HOP])
    return ft.arrays()


# ---------------------------------------------------------------- rehearsal analysis
def offline_notes_v1(F, min_len=0.06):
    """v1 (phase 1/2) segmentation, kept for comparison. Over-segments vibrato ~4x on real
    voices (eval/RESULTS.md). Returns [{'on','off','midi','kind'}] (times in s)."""
    t, v, m, db = F['t'], F['voiced'], F['midi'], F['db']
    n = len(t)
    ms = m.copy()
    for i in range(n):                                       # centred median over voiced frames
        if v[i]:
            w = [m[k] for k in range(max(0, i - 2), min(n, i + 3)) if v[k]]
            ms[i] = np.median(w)
    q = np.round(ms)
    bounds = []                                              # (frame, kind)
    i = 0
    while i < n:
        if not v[i]:
            i += 1
            continue
        s = i
        while i < n and v[i]:
            i += 1
        e = i
        if e - s < 3:
            continue
        bounds.append((s, 'attack'))
        cur = np.median(q[s:min(e, s + 4)])
        k = s + 1
        while k < e:
            if q[k] != cur and np.all(q[k:min(e, k + 4)] == q[k]) and min(e, k + 4) - k >= 3:
                mid = (cur + q[k]) / 2
                b = k
                while b > bounds[-1][0] + 2 and (ms[b - 1] - mid) * (q[k] - cur) > 0:
                    b -= 1
                bounds.append((b, 'glide'))
                cur = q[k]
            k += 1
        bounds.append((e, 'end'))
    notes = []
    for (b0, kind), (b1, _) in zip(bounds[:-1], bounds[1:]):
        if kind == 'end':
            continue
        notes.append({'s': b0, 'e': b1, 'kind': kind})
    out = []                                                 # split repeated pitches at loudness dips
    for nt in notes:
        s, e = nt['s'], nt['e']
        cuts = []
        for k in range(s + 8, e - 8):
            lo, hi = max(s, k - 12), min(e, k + 13)
            if db[k] == db[lo:hi].min() and db[lo:k].max() - db[k] >= 4 and db[k:hi].max() - db[k] >= 4:
                cuts.append(k)
        edges = [s] + cuts + [e]
        for j in range(len(edges) - 1):
            a, b = edges[j], edges[j + 1]
            if (b - a) * HOP / SR < min_len:
                continue
            kind = nt['kind'] if j == 0 else 'repeat'
            out.append({'on': float(t[a]), 'off': float(t[min(b, n - 1)]),
                        'midi': int(np.median(q[a:b][v[a:b]])) if v[a:b].any() else 0, 'kind': kind})
    return out


# ---------------------------------------------------------------- rehearsal analysis (v2)
def _runs(mask):
    """[(start, end)] of the True runs of a boolean array (end exclusive)."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _nan_filter(x, half, fn):
    """Centred running statistic over the finite values of x within +-half frames."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - half):i + half + 1]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = fn(w)
    return out


def _attack_starts(v):
    """Frames where the voice starts after >= 4 unvoiced frames (the live detector's attack rule)."""
    v = np.asarray(v, bool)
    return np.array([j for j in range(len(v)) if v[j] and (j < 4 or not v[j - 4:j].any())], int)


def note_kinds(notes, F, pre=0.08, post=0.06):
    """Onset kind of each note as the score-informed detector sees it: 'attack' if the voice
    starts after >= 4 unvoiced frames within [on - pre, on + post] (or the first note), 'repeat'
    if the rounded pitch equals the previous note's, else 'glide'."""
    t = F['t']
    starts = t[_attack_starts(F['voiced'])] if len(t) else np.array([])
    out = []
    for k, nt in enumerate(notes):
        if k == 0 or (starts.size and np.any((starts >= nt['on'] - pre) & (starts <= nt['on'] + post))):
            out.append('attack')
        elif int(round(nt['midi'])) == int(round(notes[k - 1]['midi'])):
            out.append('repeat')
        else:
            out.append('glide')
    return out


def offline_notes(F, min_len=0.15, smooth=0.10, hyst=0.8, stable=0.08, bridge=0.03, dip_db=6.0,
                  dip_len=0.15, merge_gap=0.0):
    """Segment the reference into notes, robust to vibrato (v2). Returns [{'on','off','midi','kind'}].

    v1 quantised a 50 ms median of the pitch, so every vibrato excursion across a semitone
    boundary became a new 'glide' note (~4x over-segmentation on trained voices, eval/RESULTS.md).
    Here, inside each voiced region (unvoiced gaps <= `bridge` s are bridged):
      1. pitch = 50 ms median (removes octave spikes), then a `smooth` s running mean, about one
         vibrato cycle, which cancels the vibrato;
      2. a new note starts only when the smoothed pitch has moved more than `hyst` semitones from
         the current note AND stays within 0.45 semitone of the new semitone for `stable` s
         (hysteresis); the boundary is put where the smoothed pitch crosses the midpoint;
      3. segments shorter than `min_len` are merged into the neighbour of closer pitch;
      4. a held note is split as a 'repeat' at a loudness dip of >= `dip_db` dB with >= `dip_len` s
         on both sides; same-pitch notes separated by an unvoiced gap < `merge_gap` s are merged.
    Kinds follow the live detector's rules (note_kinds)."""
    t, v, m, db = F['t'], np.asarray(F['voiced'], bool), np.asarray(F['midi'], float), np.asarray(F['db'], float)
    n = len(t)
    if n == 0:
        return []
    dt = HOP / SR
    fr = lambda s: max(1, int(round(s / dt)))  # noqa: E731
    vb = v.copy()
    for s, e in _runs(~v):
        if s > 0 and e < n and e - s <= fr(bridge):
            vb[s:e] = True
    segs = []                                               # (start, end, pitch)
    for s, e in _runs(vb):
        if e - s < fr(min_len):
            continue
        p = np.where(v[s:e], m[s:e], np.nan)
        med = _nan_filter(p, 2, np.median)
        sm = _nan_filter(med, fr(smooth) // 2, np.mean)
        if not np.isfinite(sm).any():
            continue
        idx = np.arange(e - s)
        ok = np.isfinite(sm)
        sm = np.interp(idx, idx[ok], sm[ok])
        L = fr(stable)
        cur = float(np.round(np.median(sm[:min(len(sm), L)])))
        bnd = [0]
        j = 1
        while j < len(sm):
            q = float(np.round(sm[j]))
            if q != cur and abs(sm[j] - cur) > hyst and j + L <= len(sm) and np.all(np.abs(sm[j:j + L] - q) <= 0.45):
                mid = (cur + q) / 2
                b = j
                while b > bnd[-1] + 1 and (sm[b - 1] - mid) * (q - cur) > 0:
                    b -= 1
                bnd.append(b)
                cur = q
                j += L
                continue
            j += 1
        bnd.append(len(sm))
        parts = []
        for a, b in zip(bnd[:-1], bnd[1:]):
            if b > a:
                pm = med[a:b][np.isfinite(med[a:b])]
                parts.append([s + a, s + b, float(np.round(np.median(pm))) if pm.size else np.nan])
        changed = True                                      # merge too-short parts
        while changed and len(parts) > 1:
            changed = False
            for i, (a, b, pq) in enumerate(parts):
                if b - a < fr(min_len) or not np.isfinite(pq):
                    if i == 0:
                        j2 = 1
                    elif i == len(parts) - 1:
                        j2 = i - 1
                    else:
                        dl = abs(parts[i - 1][2] - pq) if np.isfinite(pq) else 0
                        dr = abs(parts[i + 1][2] - pq) if np.isfinite(pq) else 1
                        j2 = i - 1 if dl <= dr else i + 1
                    lo, hi = min(i, j2), max(i, j2)
                    a2, b2 = parts[lo][0], parts[hi][1]
                    keep = parts[j2][2]
                    parts[lo:hi + 1] = [[a2, b2, keep]]
                    changed = True
                    break
        for a, b, pq in parts:
            if np.isfinite(pq) and b - a >= fr(min_len):
                if segs and segs[-1][2] == pq and segs[-1][1] == a:   # same pitch, contiguous
                    segs[-1][1] = b
                else:
                    segs.append([a, b, pq])
    if merge_gap > 0:
        out = []
        for sg in segs:
            if out and out[-1][2] == sg[2] and (sg[0] - out[-1][1]) * dt < merge_gap:
                out[-1][1] = sg[1]
            else:
                out.append(sg)
        segs = out
    D = fr(dip_len)
    notes = []
    for a, b, pq in segs:                                   # split held notes at loudness dips
        cuts = []
        k = a + D
        while k < b - D:
            lo, hi = max(a, k - 12), min(b, k + 13)
            if (db[k] == db[lo:hi].min() and db[lo:k].max() - db[k] >= dip_db
                    and db[k:hi].max() - db[k] >= dip_db and (not cuts or k - cuts[-1] >= D)):
                cuts.append(k)
                k += D
                continue
            k += 1
        edges = [a] + cuts + [b]
        for x0, x1 in zip(edges[:-1], edges[1:]):
            notes.append({'on': float(t[x0]), 'off': float(t[min(x1, n - 1)]), 'midi': int(pq), 'kind': ''})
    for nt, kd in zip(notes, note_kinds(notes, F)):
        nt['kind'] = kd
    return notes


def align_score(score, F, hop=2, band=0.25, refine=0.06, pitch_shift=None, hold=0.15, attack_refine=0.15):
    """Align a symbolic score of the song to the rehearsal recording by offline DTW (v2).

    score: [{'on','off','midi'}] in nominal time (any tempo; e.g. MIDI seconds or bars). Returns
    the score's notes with HER rehearsal timing, [{'on','off','midi','kind','score_on'}], so the
    reference has the right notes (from the score) and her timing (from the audio).
    Method: both sides on a 20 ms grid; the score is scaled linearly onto her voiced span; cost =
    octave-tolerant pitch distance on voiced frames / voicing mismatch; DTW with steps (1,1), (1,2),
    (2,1) and (1,0) (she may hold any note or rest longer than the score, at `hold` per frame; the
    score may go at most 2x faster than her), free start and end on the audio side, inside a band of
    +-`band` x the piece length around the linear map. Each note onset is then snapped to the
    nearest attack within +-`attack_refine` s (attacks are unambiguous; the DTW's free start tends
    to start late), else to a pitch-midpoint crossing within +-`refine` s. pitch_shift: semitones added to
    the score (None = estimated, e.g. a transposed MIDI)."""
    sc = sorted([dict(nt) for nt in score if nt.get('midi', 0)], key=lambda x: x['on'])
    if not sc:
        return []
    t, v, m = F['t'], np.asarray(F['voiced'], bool), np.asarray(F['midi'], float)
    n = len(t) // hop
    va = v[:n * hop].reshape(n, hop).any(axis=1)
    pm = np.where(v[:n * hop], m[:n * hop], np.nan).reshape(n, hop)
    with np.errstate(all='ignore'):
        pa = np.nanmedian(np.where(np.isfinite(pm), pm, np.nan), axis=1)
    va &= np.isfinite(pa)
    ta = t[:n * hop].reshape(n, hop)[:, 0]
    vi = np.flatnonzero(va)
    if vi.size < 2:
        return []
    a0, a1 = ta[vi[0]], ta[vi[-1]]
    s0, s1 = sc[0]['on'], max(x['off'] for x in sc)
    scale = (a1 - a0) / max(s1 - s0, 1e-6)
    gs = HOP * hop / SR
    ns = int(np.ceil((s1 - s0) * scale / gs)) + 1
    sp = np.zeros(ns)
    for x in sc:
        j0, j1 = int((x['on'] - s0) * scale / gs), int(np.ceil((x['off'] - s0) * scale / gs))
        sp[j0:max(j1, j0 + 1)] = x['midi']
    if pitch_shift is None:                                  # transposition: compare pitch-class histograms
        hs = np.bincount((np.round(sp[sp > 0]).astype(int)) % 12, minlength=12).astype(float)
        ha = np.bincount((np.round(pa[va]).astype(int)) % 12, minlength=12).astype(float)
        sh = int(np.argmax([np.dot(np.roll(hs, k), ha) for k in range(12)]))
        sh = sh - 12 if sh > 6 else sh
        oc = np.round((np.median(pa[va]) - np.median(sp[sp > 0]) - sh) / 12.0) * 12
        pitch_shift = sh + oc
    sp = np.where(sp > 0, sp + pitch_shift, 0.0)
    # DTW, row by row (audio frame i), keeping only the step choices for the backtrack
    isn = sp > 0
    lin = (ta - a0) / gs                                   # linear map audio frame -> score frame
    J = np.arange(ns)
    BIG = np.float32(1e30)

    def cost_row(i):
        if va[i]:
            d = np.abs(pa[i] - sp)
            d = np.minimum(d, np.abs(d - 12) + 0.5)
            c = np.where(isn, np.minimum(d, 3.0) / 3.0, 0.5)
        else:
            c = np.where(isn, 0.7, 0.0)
        c = c.astype(np.float32)
        c[np.abs(J - lin[i]) > band * ns] = BIG
        return c
    step = np.zeros((n, ns), np.int8)
    last = np.full(n, BIG, np.float32)
    c_prev = cost_row(0)
    D_prev = np.full(ns, BIG, np.float32)
    D_prev[0] = c_prev[0]
    D_prev2 = np.full(ns, BIG, np.float32)
    last[0] = D_prev[ns - 1]
    hold = np.float32(hold)
    for i in range(1, n):
        c = cost_row(i)
        best = np.full(ns, BIG, np.float32)
        best[1:] = D_prev[:-1]                             # (1,1)
        st = np.zeros(ns, np.int8)
        if ns > 2:
            cand = D_prev[:-2] + c[1:-1]                    # (1,2): two score frames for one audio frame
            better = cand < best[2:]
            best[2:] = np.where(better, cand, best[2:])
            st[2:][better] = 1
        cand = D_prev2[:-1] + c_prev[1:]                    # (2,1): two audio frames for one score frame
        better = cand < best[1:]
        best[1:] = np.where(better, cand, best[1:])
        st[1:][better] = 2
        cand = D_prev + hold                                # (1,0): she holds longer than the score
        better = cand < best
        best = np.where(better, cand, best)
        st[better] = 3
        row = np.minimum(best + c, BIG)
        row[0] = c[0]                                       # free start on the audio side
        step[i] = st
        D_prev2, D_prev, c_prev = D_prev, row, c
        last[i] = row[ns - 1]
    i = int(np.argmin(last))                                # free end on the audio side
    j = ns - 1
    path = [(i, j)]
    while j > 0 and i > 0:
        s_ = step[i, j]
        if s_ == 0:
            i, j = i - 1, j - 1
        elif s_ == 1:
            path.append((i, j - 1))
            i, j = i - 1, j - 2
        elif s_ == 2:
            path.append((i - 1, j))
            i, j = i - 2, j - 1
        else:
            i, j = i - 1, j
        path.append((i, j))
    path = path[::-1]
    pj = np.array([p[1] for p in path])
    pi = np.array([p[0] for p in path])

    def a_time(x_nominal):
        jj = (x_nominal - s0) * scale / gs
        k = int(np.searchsorted(pj, jj))
        k = min(max(k, 0), len(pj) - 1)
        return float(ta[min(pi[k], n - 1)] + (jj - pj[k]) * gs if pj[k] < jj else ta[min(pi[k], n - 1)])
    starts = t[_attack_starts(v)]
    notes = []
    for k, x in enumerate(sc):
        on = a_time(x['on'])
        mid = x['midi'] + pitch_shift
        best = None
        if starts.size:                                    # snap: an attack nearby (attacks are
            dd = np.abs(starts - on)                       # unambiguous, so a wider window; the free
            if dd.min() <= max(refine, attack_refine):     # start of the DTW tends to start late)
                best = float(starts[int(np.argmin(dd))])
        if best is None and notes:                         # or the pitch crossing the midpoint
            p0 = notes[-1]['midi']
            if p0 != mid:
                lo_i = int(np.searchsorted(t, on - refine))
                hi_i = int(np.searchsorted(t, on + refine))
                half = (p0 + mid) / 2.0
                sg = np.sign(mid - p0)
                for jx in range(max(lo_i, 1), min(hi_i, len(t))):
                    if v[jx] and v[jx - 1] and (m[jx - 1] - half) * sg < 0 <= (m[jx] - half) * sg:
                        best = float(t[jx])
                        break
        on = best if best is not None else on
        if notes and on <= notes[-1]['on'] + 0.03:
            on = notes[-1]['on'] + 0.03
        notes.append({'on': on, 'off': max(a_time(x['off']), on + 0.03), 'midi': int(round(mid)),
                      'score_on': float(x['on'])})
    for k in range(len(notes) - 1):
        notes[k]['off'] = min(notes[k]['off'], notes[k + 1]['on'])
    for nt, kd in zip(notes, note_kinds(notes, F)):
        nt['kind'] = kd
    return notes


def rehearsal_reference(F, score=None, **kw):
    """The rehearsal reference notes, from either source (v2 rehearsal API):
    score=None -> segment her rehearsal recording (offline_notes);
    score=[{'on','off','midi'}] -> the song's symbolic score aligned to her rehearsal (align_score):
    correct notes from the score AND her rehearsal timing."""
    return offline_notes(F, **kw) if score is None else align_score(score, F, **kw)


MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def detect_key(notes):
    hist = np.zeros(12)
    for nt in notes:
        hist[nt['midi'] % 12] += nt['off'] - nt['on']
    best = None
    for tonic in range(12):
        for mode, prof in (('major', MAJOR), ('minor', MINOR)):
            c = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
            if best is None or c > best[0]:
                best = (c, tonic, mode)
    return best[1], best[2], hist


def plan_harmony(notes, min_dur=0.12):
    """A harmony line a diatonic third below her (a sixth if the third falls outside the key)."""
    tonic, mode, hist = detect_key(notes)
    steps = [0, 2, 4, 5, 7, 9, 11] if mode == 'major' else [0, 2, 3, 5, 7, 8, 10]
    scale = {(tonic + s) % 12 for s in steps}
    if mode == 'minor' and hist[(tonic + 11) % 12] > hist[(tonic + 10) % 12]:
        scale = (scale - {(tonic + 10) % 12}) | {(tonic + 11) % 12}
    used = {nt['midi'] % 12 for nt in notes}
    plan = []
    for k, nt in enumerate(notes):
        if nt['off'] - nt['on'] < min_dur or nt['midi'] == 0:
            continue
        m = nt['midi']
        choice = None
        for iv in (3, 4, 8, 9):
            if (m - iv) % 12 in scale:
                choice = m - iv
                break
        if choice is None:
            for iv in (3, 4, 8, 9):
                if (m - iv) % 12 in used:
                    choice = m - iv
                    break
        plan.append({'note': k, 'r': nt['on'], 'dur': nt['off'] - nt['on'], 'midi': m,
                     'h': choice if choice is not None else m - 3})
    return plan, f'{NAMES[tonic]} {mode}'


# ---------------------------------------------------------------- following
class FollowerV1:
    """v1 follower (phase 1/2, kept verbatim for comparison): online DTW over the reference
    features, one live frame at a time.

    Steps allowed: diagonal, 2 live : 1 ref, 1 live : 2 ref, and 'wait' (live moves, ref holds).
    """
    def __init__(self, R, band=(40, 80)):
        self.R = R
        self.n = len(R['t'])
        self.D1 = np.full(self.n, INF)
        self.D1[0] = 0.0
        self.D2 = np.full(self.n, INF)
        self.L1 = np.zeros(self.n)
        self.L2 = np.zeros(self.n)
        self.c1 = np.zeros(self.n)
        self.band = band
        self.pos = 0
        self.ema = 0.5
        self.lost = 0
        self.global_left = 0
        self.resets = 0
        self.anchor_pos = 0
        self.ar = np.arange(self.n)

    def cost(self, m, v, nov):
        R = self.R
        if v:
            c = np.where(R['voiced'], np.minimum(np.abs(m - R['midi']), 3.0) / 3.0, 1.0)
        else:
            c = np.where(R['voiced'], 0.6, 0.0)
        return c + 0.3 * np.abs(nov - R['nov'])

    def step(self, m, v, nov):
        c = self.cost(m, v, nov)
        cand = np.full((4, self.n), INF)
        lens = np.zeros((4, self.n))
        cand[0, 1:] = self.D1[:-1] + c[1:]
        lens[0, 1:] = self.L1[:-1] + 1
        cand[1, 1:] = self.D2[:-1] + self.c1[1:] + c[1:]
        lens[1, 1:] = self.L2[:-1] + 2
        cand[2, 2:] = self.D1[:-2] + c[1:-1] + c[2:]
        lens[2, 2:] = self.L1[:-2] + 2
        cand[3] = self.D1 + c + (0.25 if v else 0.03)
        lens[3] = self.L1 + 1
        k = np.argmin(cand, axis=0)
        D = cand[k, self.ar]
        L = lens[k, self.ar]
        self.D2, self.L2, self.c1 = self.D1, self.L1, c
        self.D1, self.L1 = D, L
        norm = D / np.maximum(L, 1.0)
        if self.global_left > 0:
            self.global_left -= 1
            score = norm + 0.15 * (self.ar < self.anchor_pos - 100)
            self.pos = int(np.argmin(score))
        else:
            lo = max(0, self.pos - self.band[0])
            hi = min(self.n, self.pos + self.band[1])
            self.pos = lo + int(np.argmin(norm[lo:hi]))
        self.ema = 0.9 * self.ema + 0.1 * float(c[self.pos])
        if v and self.ema > 0.45:
            self.lost += 1
        else:
            self.lost = max(0, self.lost - 2)
        if self.lost > 80:
            self.reset()
        return self.pos

    def reset(self):
        """Lost for 0.8 s: restart with an open beginning so it can re-find her anywhere."""
        self.anchor_pos = self.pos
        self.D1 = np.zeros(self.n)
        self.D2 = np.zeros(self.n)
        self.L1 = np.zeros(self.n)
        self.L2 = np.zeros(self.n)
        self.global_left = 60
        self.lost = 0
        self.resets += 1

    @property
    def conf(self):
        return 1.0 - self.ema

    def ref_time(self):
        return float(self.R['t'][self.pos])


# ---------------------------------------------------------------- following (v2)
FOLLOW_PARAMS = dict(
    rest_min=0.12,       # s: a reference gap at least this long is a rest state
    frac_note=0.35,      # sub-states per reference frame inside a note (duration model)
    frac_rest=0.20,      # ... inside a rest (broader: breaths vary more than notes)
    kappa=0.30,          # share of 2-sub-state moves (a note may be ~3x shorter than rehearsed)
    sigma=1.0,           # pitch tolerance, semitones
    eps=0.10,            # pitch-outlier floor (vibrato excursions, octave errors, bleed)
    u_note=0.12,         # likelihood of an unvoiced frame inside a note (consonants); in a rest: 1
    v_rest=0.25,         # likelihood of a voiced frame inside a rest (other singers' bleed)
    q_opt=0.03,          # a breath where the reference has none
    early_exit=0.002,    # per frame: from the 2nd half of a rest straight into the next note
    p_skip=0.002,        # she leaves one note out
    wait_exit=0.05,      # per frame, leaving the last sub-state of a rest (the waiting state); None =
                         # v2 as frozen (3bd5be966104), where it left at the duration-chain rate
    lost_q=0.30,         # voiced-frame match quality below which she does not match where we are
    lost_frames=200,     # voiced frames of poor match (net) before a re-search
    inject=0.10,         # share of the belief re-spread over the piece on a re-search
    inject_gap=300,      # frames between re-searches
    fwd_s=30.0,          # re-search prefers the next fwd_s seconds of the piece (x4 weight)
    window=1.5)          # s: confidence = belief within +-window of the position


class Follower:
    """Where is she in the song? v2: a note-level hidden Markov model over the reference notes.

    The reference becomes a left-to-right chain of segments: pre-roll rest, note 1, [rest],
    note 2, ... Each segment is a chain of sub-states whose length encodes its rehearsal duration
    (scaled by her current tempo, set_tempo), so the belief moves through the song at her pace.
    Each live frame is scored against every sub-state: a note expects voice at its pitch
    (octave-tolerant, with an outlier floor for vibrato and bleed), a rest expects silence.
    Consequences (eval/RESULTS.md v1 failure modes):
      * in her rests the belief advances at her tempo through the rest and then WAITS at its end:
        it cannot enter the next note without voice near that note's pitch (no racing ahead);
      * moves are local (next sub-state, skip one note, an unplanned breath); the whole-piece
        re-search of v1 is replaced by re-spreading a small share of the belief over the piece,
        preferring forward continuation, only after ~2 s of voiced frames that match nothing near
        the current position, at most every 3 s;
      * conf = belief within +-1.5 s of the position, reduced while the recent voice does not
        match: the Accompanist holds instead of singing when it is unsure.
    API as v1: Follower(R[, notes]).step(midi, voiced, novelty) -> frame index of R, ref_time(),
    conf, resets, global_left (always 0). Without `notes` the reference is segmented with
    offline_notes(R); an Accompanist re-binds the follower to its own notes on its first step."""

    def __init__(self, R, notes=None, params=None, band=None):
        self.R = R
        self.p = dict(FOLLOW_PARAMS, **(params or {}))
        self.auto_notes = notes is None
        self.global_left = 0
        self.resets = 0
        self.hist = []
        self._build(offline_notes(R) if notes is None else notes)

    # -------------------------------------------------------------- model
    def _build(self, notes):
        P = self.p
        notes = sorted([nt for nt in notes if nt.get('midi', 0)], key=lambda x: x['on'])
        self.notes = notes
        segs = []                                         # (kind, r0, r1, pitch, note index)
        first = notes[0]['on'] if notes else 1.0
        segs.append(('rest', min(0.0, first - 0.5), first, np.nan, -1))
        for k, nt in enumerate(notes):
            nxt = notes[k + 1]['on'] if k + 1 < len(notes) else None
            off = nt['off'] if nxt is None else min(nt['off'], nxt)
            off = max(off, nt['on'] + 0.03)
            if nxt is not None and nxt - off < P['rest_min']:
                segs.append(('note', nt['on'], max(nxt, nt['on'] + 0.03), float(nt['midi']), k))
                segs.append(('opt', nxt, nxt, np.nan, k))
            else:
                segs.append(('note', nt['on'], off, float(nt['midi']), k))
                segs.append(('rest', off, nxt if nxt is not None else off + 5.0, np.nan, k))
        self.segs = segs
        n_sub, first_sub = [], []
        S = 0
        for kind, r0, r1, _, _ in segs:
            D = max(r1 - r0, 0.0) / (HOP / SR)
            if kind == 'opt':
                ns = 1
            else:
                ns = max(1, int(round((P['frac_note'] if kind == 'note' else P['frac_rest']) * D)))
            first_sub.append(S)
            n_sub.append(ns)
            S += ns
        self.S = S
        self.n_sub = np.array(n_sub)
        self.first_sub = np.array(first_sub)
        self.seg_of = np.repeat(np.arange(len(segs)), n_sub)
        j = np.arange(S) - self.first_sub[self.seg_of]
        kinds = [s[0] for s in segs]
        self.is_note = np.array([kinds[s] == 'note' for s in self.seg_of])
        self.is_opt = np.array([kinds[s] == 'opt' for s in self.seg_of])
        self.pitch = np.array([segs[s][3] for s in self.seg_of])
        r0 = np.array([s[1] for s in segs])
        r1 = np.array([s[2] for s in segs])
        frac = (j + 0.5) / self.n_sub[self.seg_of]
        self.r_state = r0[self.seg_of] + frac * (r1 - r0)[self.seg_of]
        self.r_state = np.maximum.accumulate(self.r_state)
        self.sub_j = j
        self.note_start = np.array([self.first_sub[i] for i, s in enumerate(segs) if s[0] == 'note'], int)
        self.note_r = np.array([s[1] for s in segs if s[0] == 'note'])
        self.sub_of_note = {s[4]: self.first_sub[i] for i, s in enumerate(segs) if s[0] == 'note'}
        self.L_unv = np.where(self.is_note, P['u_note'], 1.0)
        self.b = 1.0
        self._rates(1.0)
        self.alpha = np.zeros(S)
        self.alpha[0] = 1.0
        self.r = float(self.r_state[0])
        self.conf = 1.0
        self.ema_q = 0.8
        self.lost = 0
        self.since_inject = 10 ** 9
        self.i = -1
        self.pos = 0

    def _rates(self, b):
        """Per sub-state move probabilities for tempo b (live s per reference s)."""
        P = self.p
        segs = self.segs
        S = self.S
        p0, p1, p2 = np.zeros(S), np.zeros(S), np.zeros(S)
        src, dst, pj = [], [], []
        kap = P['kappa']
        for s, (kind, r0, r1, _, k) in enumerate(segs):
            f, ns = self.first_sub[s], self.n_sub[s]
            if kind == 'opt':
                p1[f] = 0.15
                p0[f] = 0.85
                continue
            D = max(r1 - r0, 0.01) * b / (HOP / SR)
            rho = min(ns / D, 1.0)
            a1, a2 = rho * (1 - kap), rho * kap / 2
            sl = slice(f, f + ns)
            p1[sl], p2[sl] = a1, a2
            last = f + ns - 1
            nxt_kind = segs[s + 1][0] if s + 1 < len(segs) else None
            if nxt_kind == 'opt':                           # most of the time: no breath here
                adv = a1 + a2
                p1[last], p2[last] = adv * P['q_opt'], adv * (1 - P['q_opt'])
            if kind == 'rest' and P.get('wait_exit') is not None:
                # waiting state (amendment A2): at the end of a rest she may stay silent for any
                # length; without this, silence (equally likely in every rest) drained the belief
                # through the following notes into the next rest after ~2 s of extra silence.
                e = min(P['wait_exit'], a1 + a2)
                p1[last], p2[last] = e * (1 - kap), e * kap
            if kind == 'rest' and s + 1 < len(segs) and ns >= 2:   # early entry
                for q in range(f + ns // 2, last + 1):
                    src.append(q)
                    dst.append(self.first_sub[s + 1])
                    pj.append(P['early_exit'])
            if kind == 'note' and P['p_skip'] > 0:          # she leaves out the next note
                tgt = [i for i in range(s + 1, len(segs)) if segs[i][0] == 'note'][1:2]
                if tgt:
                    src.append(last)
                    dst.append(self.first_sub[tgt[0]])
                    pj.append(P['p_skip'])
        self.J_src = np.array(src, int)
        self.J_dst = np.array(dst, int)
        self.J_p = np.array(pj, float)
        out = p1 + p2
        if self.J_src.size:
            out = out + np.bincount(self.J_src, weights=self.J_p, minlength=S)
        p0 = np.clip(1.0 - out, 0.0, 1.0)
        if P.get('wait_exit') is not None:          # the end of the piece is not absorbing either
            p0[-1] = 1.0 - min(P['wait_exit'], p1[-1] + p2[-1])
        p1[-1] = p2[-1] = 0.0
        p2[-2:] = 0.0
        self.P0, self.P1, self.P2 = p0, p1, p2
        self.b = b

    def set_tempo(self, b):
        """Her current tempo (live s per reference s) from the Accompanist's prediction line."""
        b = float(np.clip(b, 0.6, 1.8))
        if abs(b - self.b) > 0.03 * self.b:
            self._rates(b)

    def bind_notes(self, notes):
        """Re-build on the given reference notes and replay the frames seen so far."""
        hist = self.hist
        self.auto_notes = False
        self.hist = []
        self._build(notes)
        for m, v in hist:
            self.step(m, v)

    # -------------------------------------------------------------- filtering
    def step(self, m, v, nov=0.0):
        P = self.p
        self.i += 1
        self.hist.append((m, v))
        a = self.alpha
        pred = a * self.P0
        pred[1:] += a[:-1] * self.P1[:-1]
        pred[2:] += a[:-2] * self.P2[:-2]
        if self.J_src.size:
            pred += np.bincount(self.J_dst, weights=a[self.J_src] * self.J_p, minlength=self.S)
        voiced = bool(v) and m == m
        if voiced:
            d = np.abs(m - self.pitch)
            d = np.minimum(d, np.abs(d - 12.0) + 0.5)
            g = (1 - P['eps']) * np.exp(-0.5 * (d / P['sigma']) ** 2) + P['eps']
            L = np.where(self.is_note, g, P['v_rest'])
        else:
            L = self.L_unv
        post = pred * L
        z = post.sum()
        tot = pred.sum()
        if not (z > 0 and np.isfinite(z)):
            post, z = pred, tot
        self.alpha = post / z
        if voiced:
            q = z / max(tot, 1e-300)
            self.ema_q = 0.97 * self.ema_q + 0.03 * q
            self.lost = self.lost + 1 if self.ema_q < P['lost_q'] else max(0, self.lost - 2)
        self.since_inject += 1
        if self.lost > P['lost_frames'] and self.since_inject > P['inject_gap']:
            self._research()
        c = np.cumsum(self.alpha)
        im = min(int(np.searchsorted(c, 0.5 * c[-1])), self.S - 1)
        self.r = float(self.r_state[im])
        lo = int(np.searchsorted(self.r_state, self.r - P['window'])) - 1
        hi = int(np.searchsorted(self.r_state, self.r + P['window'], side='right')) - 1
        mass = c[hi] - (c[lo] if lo >= 0 else 0.0)
        self.conf = float(mass * min(1.0, self.ema_q / P['lost_q']))
        self.pos = int(np.clip(np.searchsorted(self.R['t'], self.r), 0, len(self.R['t']) - 1))
        return self.pos

    def _research(self):
        """She matches nothing near the current position: re-spread a share of the belief over the
        note starts of the whole piece, preferring forward continuation."""
        P = self.p
        if self.note_start.size == 0:
            return
        w = np.where((self.note_r >= self.r - 1.0) & (self.note_r <= self.r + P['fwd_s']), 4.0, 1.0)
        g = P['inject']
        self.alpha *= (1 - g)
        np.add.at(self.alpha, self.note_start, g * w / w.sum())
        self.lost = 0
        self.since_inject = 0
        self.resets += 1

    def reset(self):
        """Compatibility with v1 callers: a re-search request is honoured as a (rate-limited)
        re-spread of the belief, not a restart."""
        if self.since_inject > self.p['inject_gap']:
            self._research()

    def ref_time(self):
        return self.r

    def note_belief(self, k):
        """Belief that she is at or past the onset of reference note k."""
        f = self.sub_of_note.get(k)
        return 0.0 if f is None else float(self.alpha[f:].sum())


# ---------------------------------------------------------------- playing
def p_prev_r(notes, k):
    return notes[k - 1]['on']


class AccompanistV1:
    """v1 accompanist (phase 1/2, kept verbatim for comparison). Turns following into machine
    notes, two ways.

    pred:  fit her tempo from recent anchored onsets; send each note early enough
           (compute + output latency) to sound at the predicted time.
    react: sound each note only after her onset for it has been heard.
    """
    def __init__(self, notes, plan, join_r, out_latency=0.005, conf_thr=0.55,
                 bias_live=None, profile=None, fit_win=2.0, fit_n=4, fit_span=0.5,
                 breaths=()):
        self.notes = notes
        self.plan = [p for p in plan if p['r'] >= join_r]
        self.by_note = {p['note']: j for j, p in enumerate(self.plan)}
        self.O = out_latency
        self.thr = conf_thr
        self.bias_live = bias_live or {}
        self.u = profile or (lambda r: r)    # expected-time coordinate learned from a rehearsal
        self.fit_win, self.fit_n, self.fit_span = fit_win, fit_n, fit_span
        self.breaths = sorted(breaths)       # reference times of breaths: phrases end there
        self.b_long = 1.0
        self.shrink = 0.0
        self.waiting_phrase = None           # phrase whose first note we cancelled: wait for her
        self.entry = None                    # (plan index, predicted time) of a phrase entry to verify
        self.false_entries = []
        self.unconfirmed = 0                 # expected onsets since the last confirmed one
        self.fails = 0                       # wrong-pitch onsets where the next note should be
        self.asked_reset = False
        self.k = 0                           # next note we expect to hear
        self.tent = None                     # its detected onset, waiting for pitch confirmation
        self.scan_from = 0
        self.last_anchor_r = -1.0
        self.max_r = -1.0
        self.anchors = []                    # (live time, ref time) of confirmed onsets
        self.a, self.b = 0.0, 1.0
        self.next_plan = 0
        self.pred = {}                       # plan index -> dict
        self.react = {}
        self.log = []

    def _fit(self, now):
        if not self.anchors:
            return
        last_t, last_r = self.anchors[-1]
        phrase = self._phrase(last_r)
        # Within a phrase: her local tempo from the last few onsets.
        # Just after a breath: singers return to their basic tempo, so do not carry the
        # slowing-down from the end of the last phrase; use the tempo of the last 10 s.
        pts = [p for p in self.anchors if last_t - p[0] <= self.fit_win and self._phrase(p[1]) == phrase][-self.fit_n:]
        T = np.array([p[0] for p in pts])
        Rr = np.array([self.u(p[1]) for p in pts])
        w = np.linspace(0.5, 1.0, len(pts))
        # her basic tempo: consecutive onsets inside one phrase over the last 10 s
        # (never across a breath, a pause or a jump)
        dt = du = 0.0
        recent = [p for p in self.anchors if last_t - p[0] <= 10.0]
        for (t0, r0), (t1, r1) in zip(recent[:-1], recent[1:]):
            if self._phrase(r0) == self._phrase(r1) and 0.05 < r1 - r0 < 3.0:
                dt += t1 - t0
                du += self.u(r1) - self.u(r0)
        if du >= 1.5:
            self.b_long = float(np.clip(dt / du, 0.6, 1.8))
        if len(pts) >= 2 and Rr.max() - Rr.min() >= self.fit_span:
            span = Rr.max() - Rr.min()               # trust the local tempo more when it spans more music
            self.b = (span * self._slope(T, Rr, w) + self.shrink * self.b_long) / (span + self.shrink)
        else:
            self.b = self.b_long
        self.a = float(np.average(T - self.b * Rr, weights=w))

    def _phrase(self, r):
        return sum(1 for g in self.breaths if g < r)

    @staticmethod
    def _slope(T, Rr, w):
        rm, tm = np.average(Rr, weights=w), np.average(T, weights=w)
        b = np.sum(w * (Rr - rm) * (T - tm)) / np.sum(w * (Rr - rm) ** 2)
        return float(np.clip(b, 0.6, 1.8))

    def step(self, now, ft, fol, compute):
        cnow = now - CONTENT_DELAY
        r_pos = fol.ref_time()
        if fol.global_left == 59:                       # follower just restarted: re-find the next note
            self.k = next((k for k, nt in enumerate(self.notes) if nt['on'] > r_pos - 0.3), len(self.notes))
            self.tent = None
        self.max_r = max(self.max_r, r_pos) if fol.global_left == 0 else r_pos
        self._listen(now, ft, fol, compute, r_pos)
        if self.entry is not None:          # did she really start the phrase we entered on?
            j, at, ph = self.entry
            if self.anchors and self._phrase(self.anchors[-1][1]) >= ph:
                self.entry = None
            elif now >= at + 0.12 and (not any(ft.v[-10:]) or
                                       self._pitch_check(ft, at, self.plan[j]['midi']) is False):
                self.false_entries.append((at, j))   # a short faded blip was heard
                self.next_plan = j          # take the note back; play it when she does start
                for k in [k for k in self.pred if k >= j]:
                    del self.pred[k]
                self.waiting_phrase = ph
                self.entry = None
        self._fit(cnow)
        if not self.anchors:
            self.a, self.b = cnow - self.u(r_pos), 1.0
        # predictive commits
        while self.next_plan < len(self.plan):
            p = self.plan[self.next_plan]
            if p['r'] < self.max_r - 0.3:
                self.pred[self.next_plan] = {'status': 'missed'}
                self.next_plan += 1
                continue
            ph = self._phrase(p['r'])
            if self.waiting_phrase is not None:
                if not self.anchors or self._phrase(self.anchors[-1][1]) < self.waiting_phrase:
                    break                   # she has not started this phrase yet
                self.waiting_phrase = None
            t_hat = self.a + self.b * self.u(p['r'])
            if t_hat - (compute + self.O) > now + HOP / SR:
                break
            if fol.conf >= self.thr and fol.global_left == 0 and self.unconfirmed < 2:
                self.pred[self.next_plan] = {'status': 'played', 'at': max(t_hat, now + compute + self.O),
                                             'b': self.b, 'sent': now}
                first_in_phrase = not self.anchors or ph > self._phrase(self.anchors[-1][1])
                if first_in_phrase:
                    self.entry = (self.next_plan, self.pred[self.next_plan]['at'], ph)
            else:
                self.pred[self.next_plan] = {'status': 'held'}
            self.next_plan += 1
        self.log.append((now, r_pos, fol.conf, self.a, self.b, fol.global_left > 0))

    def _listen(self, now, ft, fol, compute, r_pos):
        """Score-informed onset detection: we know which note should come next, so look for
        exactly that change in her voice (pitch crossing the midpoint, a loudness dip for a
        repeated pitch, or the voice starting after a breath), then confirm the pitch."""
        while self.k < len(self.notes) and self.notes[self.k]['on'] < r_pos - 0.35 and any(ft.v[-5:]):
            if self.notes[self.k]['on'] > self.last_anchor_r:
                self.unconfirmed += 1                  # expected a note here, never heard it
                self._react(self.k, now, compute, fol)
            self.k += 1
            self.tent = None
            self.scan_from = max(self.scan_from, ft.i - 30)
        # overdue: she keeps singing but the note we expect has not come long after its predicted time
        while (self.k < len(self.notes) and self.tent is None and self.anchors and any(ft.v[-5:])
               and self._phrase(self.notes[self.k]['on']) == self._phrase(self.anchors[-1][1])):
            k = self.k
            nxt = self.notes[k + 1]['on'] if k + 1 < len(self.notes) else self.notes[k]['off']
            t_exp = self.a + self.b * self.u(self.notes[k]['on'])
            if now - CONTENT_DELAY <= t_exp + max(0.35, 0.6 * self.b * (self.u(nxt) - self.u(self.notes[k]['on']))):
                break
            self.unconfirmed += 1
            self._react(k, now, compute, fol)
            self.k += 1
            self.scan_from = max(self.scan_from, ft.i - 30)
        if self.unconfirmed >= 2 and fol.global_left == 0 and not self.asked_reset and any(ft.v[-5:]):
            fol.reset()
            self.asked_reset = True
        if self.k >= len(self.notes) or self.notes[self.k]['on'] > r_pos + 0.6:
            return
        nt = self.notes[self.k]
        if self.tent is None:
            self.tent = self._detect(ft, self.k)
            if self.tent is not None:
                self._react(self.k, now, compute, fol)  # the reactive baseline sounds on detection
                j = self.by_note.get(self.k)
                if (self.waiting_phrase is not None and self._phrase(nt['on']) == self.waiting_phrase
                        and j is not None and j == self.next_plan and j not in self.pred):
                    # she has started the phrase we were waiting for: come in right away
                    self.pred[j] = {'status': 'played', 'at': now + compute + self.O, 'b': self.b, 'sent': now}
                    self.next_plan = j + 1
                    self.waiting_phrase = None
        if self.tent is None:
            return
        ok = self._pitch_check(ft, self.tent, nt['midi'])
        if ok is None:
            return                                      # not enough voice after the onset yet
        t_anchor = self.tent - self.bias_live.get(nt['kind'], 0.0)
        if ok and self.anchors and self.anchors[-1][1] == self.notes[self.k - 1]['on']:
            t_prev, r_prev = self.anchors[-1]          # implied tempo must be plausible
            ratio = (t_anchor - t_prev) / max(1e-3, self.u(nt['on']) - self.u(r_prev)) / self.b_long
            ok = 0.6 <= ratio <= 1.6
        if ok:
            self.anchors.append((t_anchor, nt['on']))
            self.last_anchor_r = nt['on']
            self.unconfirmed = 0
            self.fails = 0
            self.asked_reset = False
            self.k += 1
        else:
            self.fails += 1                            # she sang something else here
            if self.fails >= 2:
                self.unconfirmed += 1
                self.fails = 0
        self.tent = None
        if self.unconfirmed >= 2 and fol.global_left == 0 and not self.asked_reset and any(ft.v[-5:]):
            fol.reset()                                # two expected notes not there: find her again
            self.asked_reset = True

    def _detect(self, ft, k):
        """Scan the frames since the last look for the onset of note k. Returns a time or None."""
        i = ft.i
        j0 = max(self.scan_from, i - 40, 12)
        nt = self.notes[k]
        p1 = nt['midi']
        p0 = self.notes[k - 1]['midi'] if k else None
        v, m, db = ft.v, ft.m, ft.db
        # too early to be this note: before half the expected gap after the previous one
        t_min = -1.0
        if k and self.anchors and self.anchors[-1][1] == p_prev_r(self.notes, k):
            t_min = self.anchors[-1][0] + 0.5 * self.b * (self.u(nt['on']) - self.u(self.anchors[-1][1]))
        hit = None
        for j in range(j0, i + 1):
            if ft.t(j) < t_min:
                continue
            if v[j] and not any(v[j - 4:j]):            # voice starts after silence
                hit = ft.t(j)
            elif nt['kind'] == 'glide' and p0 is not None and p0 != p1 and v[j] and v[j - 1]:
                mid, sg = (p0 + p1) / 2.0, np.sign(p1 - p0)
                if (m[j - 1] - mid) * sg < 0 <= (m[j] - mid) * sg:
                    f = (mid - m[j - 1]) / (m[j] - m[j - 1]) if m[j] != m[j - 1] else 1.0
                    hit = ft.t(j - 1) + f * HOP / SR
            elif nt['kind'] == 'repeat' and j >= 15 and j - 3 >= 0:
                w = db[j - 12:j + 1]
                jm = int(np.argmin(w))
                if jm == len(w) - 4 and max(w[:jm + 1]) - w[jm] >= 3 and w[-1] - w[jm] >= 2:
                    hit = ft.t(j - 12 + jm)
            if hit is not None:
                self.scan_from = j + 1
                return hit
        self.scan_from = i + 1
        return None

    @staticmethod
    def _pitch_check(ft, t_on, midi, tol=1.5):
        """Does the voice after onset t_on sit on this note? None = not enough frames yet."""
        i0 = int(round((t_on + 0.03 + CONTENT_DELAY) * SR / HOP)) - 1
        ms = [m for m, v in zip(ft.m[i0:i0 + 8], ft.v[i0:i0 + 8]) if v]
        if len(ms) < 3:
            return None if ft.i < i0 + 8 else False
        d = abs(np.median(ms) - midi)
        return bool(min(d, abs(d - 12)) <= tol)

    def _react(self, k, now, compute, fol):
        j = self.by_note.get(k)
        if j is None or j in self.react:
            return
        if fol.conf >= self.thr and fol.global_left == 0 and self.unconfirmed < 2:
            self.react[j] = {'status': 'played', 'at': now + compute + self.O, 'b': self.b}
        else:
            self.react[j] = {'status': 'held'}

    def machine_notes(self, which):
        """[(start, dur, midi, plan index)] for rendering."""
        src = self.pred if which == 'pred' else self.react
        played = sorted((v['at'], j, v['b']) for j, v in src.items() if v['status'] == 'played')
        out = []
        for i, (at, j, b) in enumerate(played):
            dur = self.plan[j]['dur'] * b
            if i + 1 < len(played):
                dur = min(dur, played[i + 1][0] - at - 0.02)
            out.append((at, max(dur, 0.08), self.plan[j]['h'], j))
        return out


# ---------------------------------------------------------------- playing (v2)
# Phase correction gain per onset kind. Weighting by kind (attack > glide > repeat, phase-2
# recommendation) was tested on the tune split and did not help (L3 tune_fit 70.4 % vs 74.7 % on time
# with equal weights; L2 a tie), so all kinds count equally by default (eval/V2_CHANGES.md).
KIND_WEIGHT = {'attack': 1.0, 'glide': 1.0, 'repeat': 1.0}


class Accompanist:
    """Turns following into machine notes (v2), two ways.

    pred:  predict her next onsets from her anchored onsets and send each note early enough
           (compute + output latency) to sound at the predicted time;
    react: sound each note only after her onset for it has been heard (baseline).

    Listening is v1's score-informed onset detection (kind-specific detectors, pitch check,
    plausibility gate, phrase-entry verify/take-back). Changes from v1 (eval/V2_CHANGES.md):
      * tempo model 'phase_slow': phase correction (alpha 0.75) on a slowly adapting tempo T: the
        median tempo implied by pairs of her onsets >= 1 s apart inside a phrase over the last 20 s,
        shrunk towards the rehearsal tempo (prior worth 30 pairs), instead of v1's constant tempo;
      * the follower decides itself when it is lost: the Accompanist no longer forces a whole-piece
        re-search after two unheard notes (research=False) and plays whenever the follower is
        confident (gate_unconfirmed=False); unheard notes are normal on real voices (30-45 %);
      * 'missed' is reversible: if the follower's position moves back, the plan resumes there;
      * the pitch check is v1's (8 frames from +30 ms, 3 voiced); its window is configurable
        (pc_delay, pc_frames, pc_need, pc_kinds), but every shorter setting tested was worse;
      * the tempo is handed to the follower's duration model (tempo_hint).
    Every change has a switch, so v1 behaviour (with the v1 follower) is AccompanistV1."""

    def __init__(self, notes, plan, join_r, out_latency=0.005, conf_thr=0.55, bias_live=None,
                 profile=None, fit_win=2.0, fit_n=4, fit_span=0.5, breaths=(), tempo='phase_slow',
                 alpha=0.75, kind_weight=None, tempo_win=20.0, tempo_min_span=1.5, research=False,
                 gate_unconfirmed=False, unmiss=True, pc_delay=0.03, pc_frames=8, pc_need=3,
                 tempo_hint=True, score_informed=True, pitch_check=True, plausibility=True,
                 phrase_entry=True, tempo_mode='robust', tempo_prior=30.0, tempo_base=1.0,
                 pc_kinds=('attack', 'glide', 'repeat')):
        self.notes = notes
        self.plan = [p for p in plan if p['r'] >= join_r]
        self.by_note = {p['note']: j for j, p in enumerate(self.plan)}
        self.O = out_latency
        self.thr = conf_thr
        self.bias_live = bias_live or {}
        self.u = profile or (lambda r: r)
        self.fit_win, self.fit_n, self.fit_span = fit_win, fit_n, fit_span
        self.breaths = sorted(breaths)
        self.tempo = tempo
        self.alpha = alpha
        self.kw = dict(KIND_WEIGHT, **(kind_weight or {}))
        self.tempo_win, self.tempo_min_span = tempo_win, tempo_min_span
        self.tempo_mode, self.tempo_prior, self.tempo_base = tempo_mode, tempo_prior, tempo_base
        self.research = research
        self.gate_unconfirmed = gate_unconfirmed
        self.unmiss = unmiss
        self.pc_delay, self.pc_frames, self.pc_need = pc_delay, pc_frames, pc_need
        self.pc_kinds = set(pc_kinds)
        self.tempo_hint = tempo_hint
        self.score_informed = score_informed
        self.use_pitch_check = pitch_check
        self.plausibility = plausibility
        self.phrase_entry = phrase_entry
        self.on_index = {nt['on']: k for k, nt in enumerate(notes)}
        self.b_long = 1.0
        self.shrink = 0.0
        self.waiting_phrase = None
        self.entry = None
        self.false_entries = []
        self.unconfirmed = 0
        self.fails = 0
        self.asked_reset = False
        self.research_calls = 0
        self.gated = 0
        self.k = 0
        self.tent = None
        self.scan_from = 0
        self.last_anchor_r = -1.0
        self.max_r = -1.0
        self.anchors = []                    # (live time, ref time) of confirmed onsets
        self.anchor_k = []                   # note index of each anchor
        self.anchor_log = []                 # (t_anchor, r, now, k): when each anchor became known
        self.a, self.b = 0.0, 1.0
        self.T = 1.0                         # slow tempo: live s per reference s
        self._st = None
        self._n_seen = 0
        self.next_plan = 0
        self.pred = {}
        self.react = {}
        self.log = []

    # -------------------------------------------------------------- tempo
    def _phrase(self, r):
        return sum(1 for g in self.breaths if g < r)

    @staticmethod
    def _slope(T, Rr, w):
        rm, tm = np.average(Rr, weights=w), np.average(T, weights=w)
        b = np.sum(w * (Rr - rm) * (T - tm)) / np.sum(w * (Rr - rm) ** 2)
        return float(np.clip(b, 0.6, 1.8))

    def _slow_tempo(self):
        """Her slowly varying tempo T (live s per reference s).
        'fixed' : T = tempo_base (v1 system (d): the rehearsal tempo throughout);
        'robust': median of the tempi implied by all pairs of anchors inside one phrase that are
                  >= 1 s of music apart, over the last tempo_win s, shrunk towards tempo_base with
                  the weight of `tempo_prior` pairs (few pairs -> stay near the rehearsal tempo);
        'window': kind-weighted sum over consecutive same-phrase pairs (v1's long-term tempo)."""
        if self.tempo_mode == 'fixed':
            self.T = self.tempo_base
            return
        if self.tempo_mode == 'robust':
            last_t = self.anchors[-1][0]
            idx = [i for i, p in enumerate(self.anchors) if last_t - p[0] <= self.tempo_win]
            sl = []
            for x, i0 in enumerate(idx):
                t0, r0 = self.anchors[i0]
                for i1 in idx[x + 1:]:
                    t1, r1 = self.anchors[i1]
                    du = self.u(r1) - self.u(r0)
                    if du >= 1.0 and self._phrase(r0) == self._phrase(r1):
                        q = (t1 - t0) / du
                        if 0.6 <= q <= 1.8:
                            sl.append(q)
            if sl:
                n = len(sl)
                self.T = float((n * np.median(sl) + self.tempo_prior * self.tempo_base) / (n + self.tempo_prior))
            self.b_long = self.T
            return
        last_t = self.anchors[-1][0]
        dt = du = 0.0
        idx = [i for i, p in enumerate(self.anchors) if last_t - p[0] <= self.tempo_win]
        for i0, i1 in zip(idx[:-1], idx[1:]):
            (t0, r0), (t1, r1) = self.anchors[i0], self.anchors[i1]
            if self._phrase(r0) != self._phrase(r1) or not 0.05 < r1 - r0 < 3.0:
                continue
            u0, u1 = self.u(r0), self.u(r1)
            ratio = (t1 - t0) / max(u1 - u0, 1e-3)
            if not 0.67 * self.T <= ratio <= 1.5 * self.T:
                continue
            w = min(self.kw.get(self.notes[self.anchor_k[i0]]['kind'], 1.0),
                    self.kw.get(self.notes[self.anchor_k[i1]]['kind'], 1.0))
            dt += w * (t1 - t0)
            du += w * (u1 - u0)
        if du >= self.tempo_min_span:
            self.T = float(np.clip(dt / du, 0.6, 1.8))
        self.b_long = self.T

    def _fit(self, now):
        if not self.anchors:
            return
        if self.tempo == 'local':                  # v1 local fit (kept for comparison)
            return AccompanistV1._fit(self, now)
        while self._n_seen < len(self.anchors):
            t, r = self.anchors[self._n_seen]
            kind = self.notes[self.anchor_k[self._n_seen]]['kind']
            self._n_seen += 1
            self._slow_tempo()
            u, ph = self.u(r), self._phrase(r)
            st = self._st
            if st is None or u - st['u'] <= 0.02 or ph > st['ph']:
                self._st = {'tau': t, 'u': u, 'ph': ph}  # first onset / of a phrase: phase reset
                continue
            m = st['tau'] + self.T * (u - st['u'])
            e = t - m
            self._st = {'tau': m + self.alpha * self.kw.get(kind, 1.0) * e, 'u': u, 'ph': ph}
        self.b = self.T
        self.a = float(self._st['tau'] - self.b * self._st['u'])

    # -------------------------------------------------------------- main step
    def step(self, now, ft, fol, compute):
        if getattr(fol, 'auto_notes', False) and hasattr(fol, 'bind_notes'):
            fol.bind_notes(self.notes)
        cnow = now - CONTENT_DELAY
        r_pos = fol.ref_time()
        if fol.global_left == 59:                  # (v1 follower) restarted: re-find the next note
            self._resync(r_pos)
        elif r_pos < self.max_r - 1.0:              # the position moved back: resume there
            self._resync(r_pos)
        self.max_r = max(self.max_r, r_pos) if fol.global_left == 0 else r_pos
        self._listen(now, ft, fol, compute, r_pos)
        if self.entry is not None:                  # did she really start the phrase we entered on?
            j, at, ph = self.entry
            if self.anchors and self._phrase(self.anchors[-1][1]) >= ph:
                self.entry = None
            elif now >= at + 0.12 and (not any(ft.v[-10:]) or
                                       self._pitch_check_v1(ft, at, self.plan[j]['midi']) is False):
                self.false_entries.append((at, j))
                self.next_plan = j
                for k in [k for k in self.pred if k >= j]:
                    del self.pred[k]
                self.waiting_phrase = ph
                self.entry = None
        self._fit(cnow)
        if not self.anchors:
            self.a, self.b = cnow - self.T * self.u(r_pos), self.T
        if self.tempo_hint and hasattr(fol, 'set_tempo'):
            fol.set_tempo(self.b)
        sure = fol.conf >= self.thr and fol.global_left == 0 and (not self.gate_unconfirmed or self.unconfirmed < 2)
        while self.next_plan < len(self.plan):
            p = self.plan[self.next_plan]
            if p['r'] < self.max_r - 0.3:
                self.pred[self.next_plan] = {'status': 'missed'}
                self.next_plan += 1
                continue
            ph = self._phrase(p['r'])
            if self.waiting_phrase is not None:
                if not self.anchors or self._phrase(self.anchors[-1][1]) < self.waiting_phrase:
                    break
                self.waiting_phrase = None
            t_hat = self.a + self.b * self.u(p['r'])
            if t_hat - (compute + self.O) > now + HOP / SR:
                break
            if sure:
                self.pred[self.next_plan] = {'status': 'played', 'at': max(t_hat, now + compute + self.O),
                                             'b': self.b, 'sent': now}
                first_in_phrase = not self.anchors or ph > self._phrase(self.anchors[-1][1])
                if first_in_phrase and self.phrase_entry:
                    self.entry = (self.next_plan, self.pred[self.next_plan]['at'], ph)
            else:
                self.pred[self.next_plan] = {'status': 'held'}
            self.next_plan += 1
        self.log.append((now, r_pos, fol.conf, self.a, self.b, fol.global_left > 0))

    def _resync(self, r_pos):
        self.k = next((k for k, nt in enumerate(self.notes) if nt['on'] > r_pos - 0.3), len(self.notes))
        self.tent = None
        self.max_r = r_pos
        if self.unmiss:
            j0 = next((j for j, p in enumerate(self.plan) if p['r'] > r_pos - 0.3), len(self.plan))
            for j in [j for j, v in self.pred.items() if j >= j0 and v['status'] == 'missed']:
                del self.pred[j]
            self.next_plan = min(self.next_plan, j0)
            while self.next_plan < len(self.plan) and self.next_plan in self.pred:
                self.next_plan += 1

    # -------------------------------------------------------------- listening
    def _listen(self, now, ft, fol, compute, r_pos):
        """Score-informed onset detection (v1), with the v2 switches."""
        while self.k < len(self.notes) and self.notes[self.k]['on'] < r_pos - 0.35 and any(ft.v[-5:]):
            if self.notes[self.k]['on'] > self.last_anchor_r:
                self.unconfirmed += 1
                self._react(self.k, now, compute, fol)
            self.k += 1
            self.tent = None
            self.scan_from = max(self.scan_from, ft.i - 30)
        while (self.k < len(self.notes) and self.tent is None and self.anchors and any(ft.v[-5:])
               and self._phrase(self.notes[self.k]['on']) == self._phrase(self.anchors[-1][1])):
            k = self.k
            nxt = self.notes[k + 1]['on'] if k + 1 < len(self.notes) else self.notes[k]['off']
            t_exp = self.a + self.b * self.u(self.notes[k]['on'])
            if now - CONTENT_DELAY <= t_exp + max(0.35, 0.6 * self.b * (self.u(nxt) - self.u(self.notes[k]['on']))):
                break
            self.unconfirmed += 1
            self._react(k, now, compute, fol)
            self.k += 1
            self.scan_from = max(self.scan_from, ft.i - 30)
        self._maybe_research(ft, fol)
        if self.k >= len(self.notes) or self.notes[self.k]['on'] > r_pos + 0.6:
            return
        nt = self.notes[self.k]
        if self.tent is None:
            self.tent = self._detect(ft, self.k) if self.score_informed else self._detect_novelty(ft, self.k)
            if self.tent is not None:
                self._react(self.k, now, compute, fol)
                j = self.by_note.get(self.k)
                if (self.waiting_phrase is not None and self._phrase(nt['on']) == self.waiting_phrase
                        and j is not None and j == self.next_plan and j not in self.pred):
                    self.pred[j] = {'status': 'played', 'at': now + compute + self.O, 'b': self.b, 'sent': now}
                    self.next_plan = j + 1
                    self.waiting_phrase = None
        if self.tent is None:
            return
        check = self.use_pitch_check and nt['kind'] in self.pc_kinds
        ok = self._pitch_check(ft, self.tent, nt['midi']) if check else True
        if ok is None:
            return
        t_anchor = self.tent - self.bias_live.get(nt['kind'], 0.0)
        if ok and self.plausibility and self.anchors and self.anchors[-1][1] == self.notes[self.k - 1]['on']:
            t_prev, r_prev = self.anchors[-1]
            ratio = (t_anchor - t_prev) / max(1e-3, self.u(nt['on']) - self.u(r_prev)) / self.b_long
            ok = 0.6 <= ratio <= 1.6
        if ok:
            self.anchors.append((t_anchor, nt['on']))
            self.anchor_k.append(self.k)
            self.anchor_log.append((t_anchor, nt['on'], now, self.k))
            self.last_anchor_r = nt['on']
            self.unconfirmed = 0
            self.fails = 0
            self.asked_reset = False
            self.k += 1
        else:
            self.fails += 1
            if self.fails >= 2:
                self.unconfirmed += 1
                self.fails = 0
        self.tent = None
        self._maybe_research(ft, fol)

    def _maybe_research(self, ft, fol):
        if self.unconfirmed >= 2 and fol.global_left == 0 and not self.asked_reset and any(ft.v[-5:]):
            if self.research:
                fol.reset()
                self.research_calls += 1
            self.asked_reset = True

    _detect = AccompanistV1._detect

    def _detect_novelty(self, ft, k, thr=0.5):
        """Generic onset detector (ablation): first rising crossing of novelty >= thr."""
        i = ft.i
        j0 = max(self.scan_from, i - 40, 12)
        nt = self.notes[k]
        t_min = -1.0
        if k and self.anchors and self.anchors[-1][1] == p_prev_r(self.notes, k):
            t_min = self.anchors[-1][0] + 0.5 * self.b * (self.u(nt['on']) - self.u(self.anchors[-1][1]))
        for j in range(j0, i + 1):
            if ft.t(j) < t_min:
                continue
            if ft.nov[j] >= thr and ft.nov[j - 1] < thr:
                self.scan_from = j + 1
                return ft.t(j)
        self.scan_from = i + 1
        return None

    def _pitch_check(self, ft, t_on, midi, tol=1.5):
        """Does the voice after onset t_on sit on this note? None = not enough frames yet."""
        i0 = int(round((t_on + self.pc_delay + CONTENT_DELAY) * SR / HOP)) - 1
        ms = [m for m, v in zip(ft.m[i0:i0 + self.pc_frames], ft.v[i0:i0 + self.pc_frames]) if v]
        if len(ms) < self.pc_need:
            return None if ft.i < i0 + self.pc_frames else False
        d = abs(np.median(ms) - midi)
        return bool(min(d, abs(d - 12)) <= tol)

    def _react(self, k, now, compute, fol):
        """The reactive baseline: sound note k on detection, under the same confidence rule."""
        j = self.by_note.get(k)
        if j is None or j in self.react:
            return
        if fol.conf >= self.thr and fol.global_left == 0 and (not self.gate_unconfirmed or self.unconfirmed < 2):
            self.react[j] = {'status': 'played', 'at': now + compute + self.O, 'b': self.b}
        else:
            self.react[j] = {'status': 'held'}

    _pitch_check_v1 = staticmethod(AccompanistV1._pitch_check)
    machine_notes = AccompanistV1.machine_notes


def profile_from_anchors(anchors, r_end, smooth=0.6):
    """Learn her habitual timing from a rehearsal: u(r) = expected seconds from the start
    of the piece to reference time r, smoothed so only phrase-level shaping remains."""
    pts = sorted((r, t) for t, r in anchors)
    rr = np.array([p[0] for p in pts])
    tt = np.array([p[1] for p in pts])
    keep = np.concatenate(([True], np.diff(rr) > 0.05))
    rr, tt = rr[keep], tt[keep]
    grid = np.arange(0.0, r_end + 0.01, 0.01)
    tg = np.interp(grid, rr, tt)
    rho = np.gradient(tg, grid)
    rho[(grid < rr[0]) | (grid > rr[-1])] = 1.0
    k = max(1, int(smooth / 0.01))
    rho = np.convolve(np.pad(rho, k, mode='edge'), np.ones(k) / k, 'same')[k:-k]
    rho = np.clip(rho, 0.5, 2.0)
    u = np.cumsum(rho) * 0.01
    return lambda r: float(np.interp(r, grid, u))
