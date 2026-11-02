"""Choir rehearsal engine v5: a score follower for a cappella singing (run choir_1102, Vincent 2026-11-02).

What his a cappella run (09:54, AirPods Max microphone) showed, offline pyin against the live tracker:
- he sang Twinkle from A2 (key A, an octave and a third below the written C4), at about 88 bpm against the page's 80;
- his pitch sits 20-50 cents flat and his intervals are narrow (sol 6.2 semitones above do, la 7.9, mi 3.0), so the
  live tracker's rounded notes flip between neighbours (Bb2 / B2) and split one sung note into many;
- v7 waited for a C, latched onto a C3 in the middle of the song and played the C arrangement against him in A.

v5 (ChoirFollow):
- Its own note segmentation on the tracker's 10 ms frames (float pitch): a note starts after 5 voiced frames, ends after
  3 unvoiced frames, and splits only when the pitch moves >= 0.75 semitone from the note's running median for 7 frames
  with a steady new pitch (the last 7 deviating frames: a glide first, then the new note); a note's pitch is the median
  of its frames (first 2 skipped), refined while she holds it.
- Song: her distinct pitches (repeats merged) are aligned by dynamic programming to the start of each library song, for
  every key (12) and tuning offset (-0.5 .. +0.25 semitone): matches cost min(1, (distance / 1.2 st)^2) on the
  pitch-class circle, an extra note of hers 0.8, a missed song note 0.9, notes before the song's first note 0.5 each
  (clicks, breaths). A song is taken when its best alignment has >= 3 matches, ends on her latest note, costs <= 0.5 per
  match, and beats every other song / key by >= 0.6. Duet is silent until then.
- Key and tuning from her TONIC notes (her steadiest degree), not from the alignment: the parts are transposed to that
  key and retuned so their tonic sits on hers (+-0.9 semitone), re-measured on every tonic note she sings.
- Her own scale: the follower expects written pitch w at (her tonic) + s * (w - written tonic), s = median ratio of her
  intervals to the written ones (0.75-1.05), so a narrow-interval singer is still followed; the parts stay
  equal-tempered from her tonic.
- Following: each new note of hers is matched to the next 1-4 melody notes: if she changed pitch (>= 0.6 st from her
  previous note) only a different written pitch can match, else only a repeat of the written pitch (a re-sung note);
  a repeated note she slurs is skipped for free, a missed one costs 0.7; pitch cost min(1, (distance / 1.6 st)^2) to
  where she should sing it, plus a timing penalty beyond 0.75 beat from where the beat map expects it; accepted at
  total <= 0.9. Accepted notes update the beat map (weighted line fit of her last 4 onsets against their melody beats;
  tempo moved 60 % of the way to the fit, within 0.75-1.33 x the page tempo; re-anchored 80 % of the way to her latest
  onset). After 3 unmatched pitch changes in a row it searches the melody for her last 3 distinct notes, preferring
  places near where she should be (songs repeat phrases), and jumps there.
- Attribute names avoid the base engine's own (notes, cur, pos are PolicyDuet's segmenter state): her notes are
  self.hnotes, the open note self.fcur, the melody position self.mpos.
- Parts: A, T, B of the arrangement, transposed to her key (nearest, |shift| <= 6) and retuned to her tonic
  (float pitch). If she sings 7+ semitones below the arrangement's soprano (a man singing the melody), the tenor part
  goes up an octave so it sits above her instead of on top of her. Each part note sounds LEAD = 35 ms before the beat
  map's time for it (the page tells the engine the calibrated round trip without the rt systems' 20 ms margin).
- She stops for 3 beats -> the parts stop and the run ends; no song recognised within 24 page beats -> the run ends.

  ChoirFollow(songs, bpm, click_all=False)   songs = choir_songs_v3.json['songs'] (or a subset)
"""
import numpy as np

import choir_v1 as C1
from live_bench3 import ReedVoice
from dsp import HOP, SR

load_songs = C1.load_songs


def circ(x):
    """Signed distance on the pitch-class circle, in [-6, 6)."""
    return (x + 6.0) % 12.0 - 6.0


class ChoirFollow(C1.LR.RTDuet):
    LEVEL, GAP, AHEAD, LEAD = 0.2, 0.03, 0.3, 0.035
    MIN_ON, GAP_END, MOVE, MOVE_FRAMES = 5, 3, 0.75, 7
    TOL, INS, DEL, PRE_SKIP = 1.2, 0.8, 0.9, 0.5
    ID_MARGIN, ID_PER_MATCH = 0.6, 0.5
    TOL_F, ACCEPT, TEMPO_GAIN, PHASE_GAIN = 1.6, 0.9, 0.6, 0.8
    QUIT, WAIT = 3.0, 24.0
    TUNINGS = (-0.5, -0.25, 0.0, 0.25)

    def __init__(self, songs, bpm, click_all=False, log=lambda s: None):
        super().__init__(C1._Listen(), 'choir:follow', 0, 4, 400, bpm, count_in=4, click_all=click_all, lead=0.5,
                         log=log, listen_only=True)
        self.songs = songs
        self.dist = {sid: self._distinct([(a, m) for a, b, m in s['melody']]) for sid, s in songs.items()}
        self.choir = ReedVoice(level=self.LEVEL)
        self.click_t0, self.T0_b = self.t0, self.T_b
        # note segmentation
        self.seg_k = 0                                   # next tracker frame to read
        self.fcur, self.unv = None, 0
        self.hnotes = []                                  # her confirmed notes: dict(t, frames, pitch)
        # recognition / following
        self.song_id = self.arr = None
        self.T = self.S = 0
        self.u, self.low = 0.0, False
        self.mel = self.parts = None
        self.mpos, self.unmatched, self.pairs = -1, 0, []
        self.anchor_t = self.anchor_b = None
        self.Tb = self.T0_b
        self.next_part, self.stopped = 0, None
        self.diag = {'pred_err_ms': [], 'relocations': 0, 'id_note': None, 'id_cost': None, 'tune_track': []}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _distinct(seq):
        """[(beat, midi)] -> runs of equal pitch merged: [(beat of the first, midi)]."""
        out = []
        for b, m in seq:
            if not out or out[-1][1] != m:
                out.append((float(b), m))
        return out

    def mc(self, p, q):
        return min(1.0, (abs(circ(p - q)) / self.TOL) ** 2)

    def mcf(self, p, q):
        """Following: she is already placed in the song, so a wider pitch tolerance (narrow intervals, flat notes)."""
        return min(1.0, (abs(circ(p - q)) / self.TOL_F) ** 2)

    def t_of(self, b):
        return self.anchor_t + (b - self.anchor_b) * self.Tb

    def b_of(self, t):
        return self.anchor_b + (t - self.anchor_t) / self.Tb

    # ---------------------------------------------------------------- her notes from the tracker's frames
    def _frames(self):
        while self.seg_k < len(self.ft_t):
            k = self.seg_k
            self.seg_k += 1
            t, m, v = self.ft_t[k], self.ft_m[k], self.ft_v[k]
            if v and m > 0:
                self.unv = 0
                c = self.fcur
                if c is None:
                    self.fcur = {'t': t, 'fr': [m], 'dev': [], 'ok': False}
                else:
                    med = float(np.median(c['fr'][-25:]))
                    if abs(m - med) > self.MOVE:
                        c['dev'].append((t, m))
                        dp = [x[1] for x in c['dev'][-self.MOVE_FRAMES:]]   # a glide first, then a steady pitch
                        if len(dp) >= self.MOVE_FRAMES and abs(np.median(dp) - med) > self.MOVE and np.ptp(dp) < 1.0:
                            self._end()
                            self.fcur = {'t': c['dev'][0][0], 'fr': dp, 'dev': [], 'ok': False}
                    else:
                        c['dev'] = []
                        c['fr'].append(m)
                c = self.fcur
                if not c['ok'] and len(c['fr']) >= self.MIN_ON:
                    c['ok'] = True
                    note = {'t': c['t'], 'fr': c['fr'], 'pitch': self._pitch(c['fr'])}
                    c['note'] = note
                    self.hnotes.append(note)
                    self._on_note(note)
                elif c['ok']:
                    c['note']['pitch'] = self._pitch(c['fr'])
            else:
                self.unv += 1
                if self.unv >= self.GAP_END and self.fcur is not None:
                    self._end()

    @staticmethod
    def _pitch(fr):
        return float(np.median(fr[2:] if len(fr) > 4 else fr))

    def _end(self):
        self.fcur = None

    # ---------------------------------------------------------------- recognition
    def _her_distinct(self):
        out = []
        for n in self.hnotes[-14:]:
            if out and abs(n['pitch'] - out[-1]['pitch']) < 0.6:
                continue
            out.append(n)
        return out

    def _align(self, h, d, off):
        """DP alignment of her pitches h to the start of song pitches d (expected d + off). -> (cost, pairs)."""
        n, m = len(h), len(d)
        INF = 1e9
        D = np.full((n + 1, m + 1), INF)
        back = {}
        for i0 in range(n):
            c = self.PRE_SKIP * i0 + self.mc(h[i0], d[0] + off)
            if c < D[i0 + 1, 1]:
                D[i0 + 1, 1] = c
                back[(i0 + 1, 1)] = ('m', i0, 0, None)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cur = D[i, j]
                if cur >= INF:
                    continue
                if i < n and j < m:
                    c = cur + self.mc(h[i], d[j] + off)
                    if c < D[i + 1, j + 1]:
                        D[i + 1, j + 1] = c
                        back[(i + 1, j + 1)] = ('m', i, j, (i, j))
                if i < n and cur + self.INS < D[i + 1, j]:
                    D[i + 1, j] = cur + self.INS
                    back[(i + 1, j)] = ('i', i, j, (i, j))
                if j < m and cur + self.DEL < D[i, j + 1]:
                    D[i, j + 1] = cur + self.DEL
                    back[(i, j + 1)] = ('d', i, j, (i, j))
        j = int(np.argmin(D[n, 1:])) + 1
        cost = float(D[n, j])
        pairs, key = [], (n, j)
        while key in back:
            kind, i, jj, prev = back[key]
            if kind == 'm':
                pairs.append((i, jj))
            if prev is None:
                break
            key = prev
        return cost, pairs[::-1]

    def _recognise(self):
        hd = self._her_distinct()
        if len(hd) < 3:
            return
        h = [n['pitch'] for n in hd]
        hyps = []
        for sid, dist in self.dist.items():
            d = [m for b, m in dist]
            for T in range(12):
                for u in self.TUNINGS:
                    cost, pairs = self._align(h, d, T + u)
                    hyps.append((cost, sid, T, u, pairs))
        ok = [x for x in hyps if len(x[4]) >= 3 and x[4][-1][0] == len(h) - 1]
        if not ok:
            return
        best = min(ok, key=lambda x: x[0])
        cost, sid, T, u, pairs = best
        if cost > self.ID_PER_MATCH * len(pairs):
            return
        o = T + u
        rivals = [x[0] for x in hyps if x[1] != sid or abs(circ(x[2] + x[3] - o)) >= 0.75]
        if rivals and min(rivals) - cost < self.ID_MARGIN:
            return
        self._start_song(sid, T, u, [(hd[i], self.dist[sid][j]) for i, j in pairs], cost)

    def _start_song(self, sid, T, u, matched, cost):
        arr = self.songs[sid]
        # key and tuning from her TONIC notes (the steadiest degree of an untrained voice; Vincent's sol and la were
        # 0.9-1.1 semitone flat while his do was in tune with itself), else from her first matched note
        ton = [n['pitch'] - m for n, (b, m) in matched if m % 12 == arr['tonic'] % 12] or \
              [matched[0][0]['pitch'] - matched[0][1][1]]
        o0 = ton[0] % 12.0
        o = float(np.median([o0 + circ(x % 12.0 - o0) for x in ton]))
        T = int(round(o)) % 12
        self.u = float(np.clip(circ(o - T), -0.6, 0.6))
        self.song_id, self.arr, self.T = sid, arr, T
        self.S = T if T <= 6 else T - 12
        # her own scale: her tonic as the reference, her intervals measured as a ratio of the written ones (Vincent's
        # were about 0.85 of the written size); the follower expects her notes there, the parts stay equal-tempered
        tpc = arr['tonic'] % 12
        self.ton = [(n, m) for n, (b, m) in matched if m % 12 == tpc] or [(matched[0][0], matched[0][1][1])]
        self.w_ref = self.ton[0][1]
        self.span = [(n, m) for n, (b, m) in matched if m % 12 != tpc and abs(m - self.w_ref) >= 3]
        self.u = self._tuning()
        sop = [n['pitch'] - (m + self.S + arr.get('melody_s_shift', 0)) for n, (b, m) in matched]
        self.low = float(np.median(sop)) <= -7.0
        self.mel = [(float(a), int(m)) for a, b, m in arr['melody']]
        self.parts = []
        for p, seq in arr['parts'].items():
            for on, off, m in seq:
                self.parts.append((float(on), float(off), int(m) + self.S + (12 if (self.low and p == 'T') else 0), p))
        self.parts.sort()
        pts = [(n['t'], b) for n, (b, m) in matched]
        self.pairs = pts[:]
        self.anchor_t, self.anchor_b = pts[-1]
        if len(pts) >= 3 and pts[-1][1] - pts[0][1] >= 1.5:
            slope = float(np.polyfit([b for t, b in pts], [t for t, b in pts], 1)[0])
            self.Tb = float(np.clip(slope, 0.75 * self.T0_b, 1.33 * self.T0_b))
        last_b = pts[-1][1]
        self.mpos = max(i for i, (b, m) in enumerate(self.mel) if b <= last_b + 1e-9)
        self.diag.update({'id_note': len(self.hnotes), 'id_cost': round(cost, 3), 'id_time': round(self.hnotes[-1]['t'], 3)})
        self.log(f'song {sid}: key shift {self.S:+d}, tuning {self.u:+.2f} st, low {self.low}, '
                 f'tempo {60 / self.Tb:.1f} bpm, cost {cost:.2f}')

    # ---------------------------------------------------------------- her scale
    def ref_her(self):
        """Her tonic, in the octave of the written reference note (median of her last 3 tonic notes, refined pitches)."""
        return float(np.median([n['pitch'] - (m - self.w_ref) for n, m in self.ton[-3:]]))

    def scale(self):
        r = self.ref_her()
        v = [(n['pitch'] - r) / (m - self.w_ref) for n, m in self.span[-8:]]
        return float(np.clip(np.median(v), 0.75, 1.05)) if v else 1.0

    def expect(self, w):
        """Where she will sing written pitch w (her tonic + her interval scale)."""
        return self.ref_her() + self.scale() * (w - self.w_ref)

    def _tuning(self):
        return float(np.clip(circ(self.ref_her() - (self.w_ref + self.T)), -0.9, 0.9))

    # ---------------------------------------------------------------- following
    def _follow(self, note):
        p, t = note['pitch'], note['t']
        cur_pitch = self.expect(self.mel[self.mpos][1])
        prev = self.hnotes[-2]['pitch'] if len(self.hnotes) >= 2 else None
        moved = prev is None or abs(p - prev) >= 0.6          # she changed pitch (else: she re-sang or split a note)
        best = None
        for k in range(1, 5):
            c = self.mpos + k
            if c >= len(self.mel):
                break
            if (self.mel[c][1] == self.mel[self.mpos][1]) == moved:
                continue                                       # a moved note is never the same written pitch, and v.v.
            skip, prev = 0.0, self.mel[self.mpos][1]
            for s in range(self.mpos + 1, c):
                skip += 0.0 if self.mel[s][1] == prev else 0.7
                prev = self.mel[s][1]
            x = abs(t - self.t_of(self.mel[c][0])) / self.Tb
            cost = self.mcf(p, self.expect(self.mel[c][1])) + skip + max(0.0, x - 0.75) * 0.8
            if best is None or cost < best[0]:
                best = (cost, c)
        if best is not None and best[0] <= self.ACCEPT:
            self._accept(note, best[1])
            return
        if not moved or abs(circ(p - cur_pitch)) < 0.6:   # a re-attack or split of the note she is on
            return
        self.unmatched += 1
        if self.unmatched >= 3:
            self._relocate()

    def _accept(self, note, c):
        t, b = note['t'], self.mel[c][0]
        pred = self.t_of(b)
        self.diag['pred_err_ms'].append(round((t - pred) * 1000, 1))
        self.pairs.append((t, b))
        pts = self.pairs[-4:]
        if len(pts) >= 3 and pts[-1][1] - pts[0][1] >= 1.5:
            w = np.array([1.0, 1.5, 2.0, 3.0][-len(pts):])
            slope = float(np.polyfit([q for _, q in pts], [s for s, _ in pts], 1, w=w)[0])
            slope = float(np.clip(slope, 0.75 * self.T0_b, 1.33 * self.T0_b))
            self.Tb += self.TEMPO_GAIN * (slope - self.Tb)
        self.anchor_t, self.anchor_b = pred + self.PHASE_GAIN * (t - pred), b
        w = self.mel[c][1]
        if w % 12 == self.arr['tonic'] % 12:                        # her tonic: the tuning reference
            self.ton.append((note, w))
        elif abs(w - self.w_ref) >= 3:
            self.span.append((note, w))
        self.u = self._tuning()
        self.diag['tune_track'].append((round(self.u, 3), round(self.scale(), 3)))
        self.mpos, self.unmatched = c, 0

    def _relocate(self):
        hd = self._her_distinct()[-3:]
        if len(hd) < 3:
            return
        dist = self.dist[self.song_id]
        b_exp = self.b_of(hd[-1]['t'])                    # songs repeat phrases: prefer the place she should be
        best = None
        for q in range(len(dist) - 2):
            cost = sum(self.mcf(hd[i]['pitch'], self.expect(dist[q + i][1])) for i in range(3)) \
                + 0.04 * abs(dist[q + 2][0] - b_exp)
            if best is None or cost < best[0]:
                best = (cost, q)
        if best is not None and best[0] <= 0.9:
            b = dist[best[1] + 2][0]
            self.mpos = max(i for i, (bb, m) in enumerate(self.mel) if bb <= b + 1e-9)
            self.anchor_t, self.anchor_b = hd[-1]['t'], b
            self.pairs = [(hd[-1]['t'], b)]
            self.next_part = 0
            self._silence()
            self.unmatched = 0
            self.diag['relocations'] += 1

    def _on_note(self, note):
        if self.arr is None:
            self._recognise()
        else:
            self._follow(note)

    # ---------------------------------------------------------------- parts
    def part_at(self, beat, part='A'):
        for on, off, m, p in self.parts or []:
            if p == part and on <= beat < off:
                return int(round(m))
        return -1

    def _schedule(self):
        if self.arr is None or self.stopped:
            return
        while self.next_part < len(self.parts):
            on, off, m, part = self.parts[self.next_part]
            start = int(round((self.t_of(on) - self.Lo - self.LEAD) * SR))
            if start > self.n + HOP + int(self.AHEAD * SR):
                break
            end = int(round((self.t_of(off) - self.Lo - self.LEAD) * SR)) - int(self.GAP * SR)
            start = max(start, self.choir.pos)
            if end - start > int(0.05 * SR):
                self.choir.schedule(start, (end - start) / SR, m + self.u, tag=('choir', self.next_part))
            self.next_part += 1

    def _silence(self):
        pos = self.choir.pos
        for nt in list(self.choir.notes):
            if nt.i0 >= pos:
                self.choir.notes.remove(nt)
                if nt.tag is not None:
                    self.choir.by_tag.pop(nt.tag, None)
            elif nt.i0 + int(nt.dur * SR) > pos:
                nt.dur = (pos - nt.i0) / SR

    # ---------------------------------------------------------------- one block
    def process(self, x):
        y = super().process(x)
        self._frames()
        self._schedule()
        y = y + self.choir.render(HOP)
        if self.arr is not None and not self.stopped:
            last = self.last_voiced if self.last_voiced is not None else self.anchor_t
            if self.heard() - last > self.QUIT * self.Tb:
                self._silence()
                self.stopped = 'she_stopped'
        return y

    def _react(self, t):
        n0 = len(self.events)
        super()._react(t)
        for e in self.events[n0:]:
            e['action'] = 'keep'
            if self.arr is None:
                e['duet'] = -1
            else:
                e['song_beat'] = round(self.b_of(self.click_t0 + e['onset_raw'] * self.T_b), 3)
                e['duet'] = self.part_at(e['song_beat'])

    def _stop(self, t):
        pass

    def done(self):
        t = self.n / SR
        if self.arr is None:
            return t > self.click_t0 + self.WAIT * self.T0_b
        if self.stopped:
            return True
        return t > self.t_of(self.arr['K'] + 1)

    def summary(self):
        s = super().summary()
        pe = self.diag['pred_err_ms']
        s.update({'mode': 'choir', 'choir_engine': 'choir_v5', 'choir_song': self.song_id, 'choir_shift': self.S,
                  'tuning_st': self.u, 'low_singer': self.low, 'tempo_bpm': 60 / self.Tb, 'page_bpm': 60 / self.T0_b,
                  'her_notes': [(round(n['t'], 3), round(n['pitch'], 2)) for n in self.hnotes[:60]],
                  'matched': len(self.pairs), 'stopped': self.stopped, 'parts_written': self.next_part,
                  'pred_err_ms_median_abs': float(np.median(np.abs(pe))) if pe else None, 'diag': self.diag})
        return s
