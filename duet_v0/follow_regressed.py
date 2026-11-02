"""Listen, follow, predict, play.

FeatureTracker  - per 10 ms: pitch (YIN), voicing, loudness, onset novelty. Causal.
offline_notes   - rehearsal-time note segmentation of the reference (may look ahead).
plan_harmony    - rehearsal-time harmony line: a diatonic third (or sixth) below her.
Follower        - online DTW against the reference features: where is she in the song?
Accompanist     - anchors her detected onsets to reference notes, fits her tempo,
                  and commits each machine note early enough to land on time.
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
def offline_notes(F, min_len=0.06):
    """Segment the reference into notes. Returns [{'on','off','midi','kind'}] (times in s)."""
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
class Follower:
    """Online DTW over the reference features, one live frame at a time.

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


# ---------------------------------------------------------------- playing
def p_prev_r(notes, k):
    return notes[k - 1]['on']


class Accompanist:
    """Turns following into machine notes, two ways.

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
        self.need_anchor = False
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
            self.unconfirmed = self.fails = 0
            self.asked_reset = False
            self.need_anchor = True                     # stay quiet until a note is confirmed again
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
            if fol.conf >= self.thr and fol.global_left == 0 and self.unconfirmed < 2 and not self.need_anchor:
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
               and (self._phrase(self.notes[self.k]['on']) == self._phrase(self.anchors[-1][1])
                    or sum(ft.v[-25:]) >= 20)):
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
            self.need_anchor = False
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
