"""Choir rehearsal engine v6 (run choir_1102). Vincent, 2026-11-02 11:05, after trying page v9: "both the a cappella and
the phone playback got worse". His v9 runs (110255 a cappella in G, 110335 phone Twinkle in C) showed why:
- v5 retuned the parts to her "tonic" notes: noise matched as tonic pushed the choir +0.74 and +0.9 semitone sharp
  (the phone is in tune at 0 cents);
- v5 waited for a DP song decision: subharmonic tracker errors (G4 heard as C3 = 1/3, A4 as A2) and split notes kept
  the cost high, so the choir came in 6-7.5 s after she started (v7 came in on her first note);
- the page told a round trip of 204 ms (median of history) while the two latest calibrations said 113 and 130 ms, so
  every part came about 100 ms early.

v6 (ChoirStart, built on choir_v5.ChoirFollow's segmentation, follower and part scheduling):
- A note counts after 8 voiced frames (80 ms; v11 has 0.25-beat notes of 120 ms at 122 bpm).
- Provisional start: when her voice has held one pitch for the last 0.3 s (>= 80 % voiced, p10-p90 within 1.2
  semitone; a scooped or split first note counts as one; clicks, 0.25 s then silence, never qualify), that pitch is the
  song's first note: the key comes from it and the opening chord (both songs open on the tonic chord) comes in right
  away, at the page tempo. Her next different note confirms the song when the interval is that song's opening interval
  (same direction, 0.6-1.2 x its size: Twinkle +7, v11 -5; a narrow-interval singer still fits): tempo from the two
  onsets, nothing re-sung; or switches to the other song (parts re-written); or, if no song opens like that, ends a
  false start (silence) and starts over. Notes > 9 semitones from the first one are tracker errors, but two of them end
  the provisional start (it was humming or noise).
- Fallback start: two of her distinct notes (up to 2 stray notes between them) with a song's opening interval, the first
  held for at least half the time until the second (a song's first note lasts 2 beats; clicks do not).
- If the first 2 pitch changes after a start match nothing, it was a false start: Duet stops and waits for a new pair.
- No retuning: the parts are equal-tempered at A = 440 Hz in her key (the follower still expects her notes on her own
  scale: her tonic + the ratio of her intervals to the written ones, 0.6-1.15).
- Following (see _follow): 6 notes look-ahead; a note far (> 5 semitones) from her current and all next 6 expected
  notes is a tracker error; written notes she skipped cost 0.3 each, with the timing deciding; tempo bounded by the
  song's own tempo (0.5-1.6 x), not the page's.

  ChoirStart(songs, bpm, click_all=False)   songs = choir_songs_v5.json['songs'] (or a subset)
"""
import bisect

import numpy as np

import choir_v5 as C5
from dsp import HOP, SR, hz_to_midi, yin
from follow import FeatureTracker

load_songs = C5.load_songs
circ = C5.circ


class TrackerAP(FeatureTracker):
    """The live tracker (follow.FeatureTracker, its voicing untouched) that also keeps every frame's YIN pitch,
    aperiodicity and level below the running peak, voiced or not, so a note's onset can be traced back through a soft
    or octave-slipped attack (ChoirStart._trace_onset). YIN runs a second time on the same window (~0.2 ms a frame)."""

    def __init__(self):
        super().__init__()
        self.f0m, self.ap, self.rel = [], [], []

    def step(self, chunk):
        i = super().step(chunk)
        f0, ap, _ = yin(self.buf)
        self.f0m.append(float(hz_to_midi(f0)) if f0 > 0 else 0.0)
        self.ap.append(float(ap))
        self.rel.append(self.db[-1] - self.peak_db)
        return i


class ChoirStart(C5.ChoirFollow):
    MIN_ON = 8                                          # 80 ms: v11 has 0.25-beat notes (120 ms at 122 bpm)
    START_RATIO = (0.6, 1.2)
    AHEAD_NOTES = 6                                     # the follower looks this many melody notes ahead
    SKIP_COST = 0.3                                     # per written note she skipped (if her timing fits it)
    DUR_WEIGHT = False                                  # short written notes move the beat map less (tried: no gain
    FULL_WEIGHT_BEATS = 1.0                             # on Vincent's runs, so off) ... full weight from this length
    # tempo from her last 8 matched onsets spanning >= 4 beats, equal weights, moved 30 % per note; phase 60 % per note
    # (choir_timing_eval_v5: v11 runs 138 -> 88 and 90 -> 44 ms median from target, Twinkle runs unchanged)
    TEMPO_WINDOW, TEMPO_MIN_SPAN, RECENCY = 8, 4.0, False
    TEMPO_GAIN, PHASE_GAIN = 0.3, 0.6
    TEMPO_ROBUST = False                                # Theil-Sen tempo instead of the weighted line fit
    TEMPO_ADAPTIVE = True                               # 3 same-side misses in a row: follow fast (4 notes, 0.6/0.8)
    PHASE_JUMP = 0                                      # > 0: a miss beyond this many beats realigns at once (tried
    #                                                     0.25 / 0.35: worse on v11, whose rubato exceeds 1/4 beat)
    # chosen by runs_mini/choir_1102/choir_timing_eval_v7-v9 on Vincent's phone runs (true onsets offline): v11 138 -> 64
    # and 90 -> 36 ms median from target, Twinkle 29 -> 20 and 47 -> 45 ms; his own a cappella runs have no reliable
    # truth, and by the follower's own prediction errors they get a little worse (median +7 to +22 ms)
    JUNK_ST = 5.0                                       # further than this from her note and all next ones: junk
    # tried, off (choir_timing_eval_v10: v11 runs 64/36/58 -> 141/84/85 ms with the warm-up, 175/108/134 with the gap
    # onset; most of her gaps are real breaths, so the gap start was far too early)
    GAP_ONSET = 0                                       # a new pitch after a gap this short starts at the gap (0: off)
    WARMUP_NOTES = 0                                    # the first notes after a start follow fast (tempo unsure)
    PROV_FRAMES = 30                                    # 0.3 s of a steady first note: the opening chord comes in
    LEGATO = True                                       # each part is one voice (see _schedule)
    ONSET_BACK, AP_LOOSE = 0.3, 0.5                     # trace a note's onset back this far, through frames this
    #                                                     periodic (0: off; see _trace_onset)
    TRACE_ALL = False                                   # trace every note, not only the ones that start the song

    def __init__(self, songs, bpm, click_all=False, log=lambda s: None):
        super().__init__(songs, bpm, click_all=click_all, log=log)
        self.ft = TrackerAP()                          # same voicing, plus raw pitch / aperiodicity per frame
        self.name = 'choir:start'
        self.open = {}
        for sid in songs:
            d = self.dist[sid]
            self.open[sid] = ((d[0][0], d[0][1]), (d[1][0], d[1][1]))
        self.after_start, self.false_starts = 0, 0
        self.block_before = 0                          # her distinct notes before this index cannot start a song
        self.provisional, self.prov_note, self.prov_tried = False, None, set()
        self.junk_t = None
        self.pair_w = []                               # weight of each beat-map point (see _accept)
        self.res_hist = []                             # her onset - the beat map's prediction, per matched note
        self._legato_for, self.part_on, self.part_tag = None, set(), {}

    # ---------------------------------------------------------------- parts, one voice each
    def _schedule(self):
        """choir_v5's, but each part is ONE voice. A part note's end used to be fixed when the note was written; when
        the beat map then moved earlier before the part's next note was written, the reed sang both notes at once
        (Vincent's 12:41 v11 run: up to 0.59 s, 1.3 s in all), and when it moved later the part fell silent before its
        next note (up to 0.26 s). Now a note the part goes straight on from is written to hold, and the part's next
        note ends it (GAP before its own start, from the beat map of that moment) when that note is written."""
        if self.arr is None or self.stopped or not self.LEGATO:
            return super()._schedule()
        if self._legato_for is not self.parts:
            self._legato_for, self.part_tag = self.parts, {}
            self.part_on = {(p, on) for on, off, m, p in self.parts}
        gap = int(self.GAP * SR)
        while self.next_part < len(self.parts):
            on, off, m, part = self.parts[self.next_part]
            start = int(round((self.t_of(on) - self.Lo - self.LEAD) * SR))
            if start > self.n + HOP + int(self.AHEAD * SR):
                break
            start = max(start, self.choir.pos)
            held = (part, off) in self.part_on            # the part goes straight on to its next note: hold until then
            end = int(round((self.t_of(off + (1.0 if held else 0.0)) - self.Lo - self.LEAD) * SR)) - gap
            prev = self.part_tag.get(part)
            nt = self.choir.by_tag.get(prev) if prev is not None else None
            if nt is not None and nt.i0 + nt.dur * SR > start - gap:
                if nt.done == 0 and nt.i0 >= start - gap:  # not begun, and replaced before it would: drop it
                    nt.dur = 0.0
                    self.choir.notes.remove(nt)
                    self.choir.by_tag.pop(prev, None)
                else:
                    self.choir.set_dur(prev, (start - gap - nt.i0) / SR)   # never into the past: releases from now
            if end - start > int(0.05 * SR):
                tag = ('choir', self.next_part)
                self.choir.schedule(start, (end - start) / SR, m + self.u, tag=tag)
                self.part_tag[part] = tag
            self.next_part += 1

    # ---------------------------------------------------------------- start
    def _on_note(self, note):
        if self.TRACE_ALL or self.arr is None or self.provisional:
            self._trace_onset(note)                    # the opening interval alone sets the starting tempo
        self._gap_onset(note)
        if self.arr is None:
            self._try_start()
        elif self.provisional:
            self._confirm(note)
        else:
            self._follow(note)

    def _trace_onset(self, note):
        """Her onset is where her attack starts, not where the tracker first calls the note voiced. Walking back from
        the note's first frame, a frame still belongs to its attack while its YIN pitch is within 2.5 semitones of the
        note (a scoop) or 1 semitone of an octave or two away (an octave slip), with aperiodicity < AP_LOOSE and level
        within 30 dB of the running peak; up to 2 other frames may sit between; never into her previous note (a frame
        within 1 semitone of its pitch: so a re-sung note keeps its onset). Still in the attack after ONSET_BACK s: a
        held note, not an attack, so nothing changes. On Vincent's four v11 phone runs the tracker heard the soft low
        E3 up to 0.21 s late and the A3 attack an octave high for 0.12 s; the opening interval then started the beat
        map at 113 or 150 bpm for a 122-143 bpm singer, and the next chords came up to 0.37 s early or 0.22 s late."""
        if not self.ONSET_BACK:
            return
        ft, p, t = self.ft, note['pitch'], note['t']
        k = bisect.bisect_left(self.ft_t, t - 1e-6)
        if k <= 0 or k > len(ft.f0m):
            return
        slip = lambda m, q: min(abs(m - q - o) for o in (12.0, -12.0, 24.0, -24.0)) <= 1.0  # noqa: E731
        prev = None
        for h in reversed(self.hnotes):
            if h is note or h.get('junk'):
                continue
            if t - h['t'] > 2.0:
                break
            if t - h['t'] <= self.ONSET_BACK and slip(h['pitch'], p):
                continue                                   # its own attack, heard an octave off
            prev = h['pitch']
            break
        j, best, miss = k - 1, k, 0
        while j >= 0 and self.ft_t[j] >= t - self.ONSET_BACK:
            m = ft.f0m[j]
            if prev is not None and abs(m - prev) <= 1.0:
                break                                      # her previous note
            if (abs(m - p) <= 2.5 or slip(m, p)) and ft.ap[j] < self.AP_LOOSE and ft.rel[j] > -30.0:
                best, miss = j, 0
            else:
                miss += 1
                if miss >= 3:
                    break
            j -= 1
        else:
            if j >= 0 and miss == 0:
                return                                     # still "attack" at the limit: a held note
        if best < k:
            note['t_heard'], note['t'] = t, self.ft_t[best]
            self.diag['onset_traced'] = self.diag.get('onset_traced', 0) + 1

    def _gap_onset(self, note):
        """A new pitch right after a short unvoiced gap (<= GAP_ONSET s) that followed a held note started when the
        gap did: the live tracker calls a soft, low attack unvoiced for a while (v11's E3 after A3 was heard 0.21 s
        after pyin heard it voiced), and the gap belongs to the new note's attack. Repeated pitches keep their onset."""
        if not self.GAP_ONSET or len(self.hnotes) < 2 or abs(note['pitch'] - self.hnotes[-2]['pitch']) < 0.6:
            return
        k = bisect.bisect_left(self.ft_t, note['t']) - 1
        while k >= 0 and not (self.ft_v[k] and self.ft_m[k] > 0):
            k -= 1
        if k < 0 or k + 1 >= len(self.ft_t):
            return
        gap_start = self.ft_t[k + 1]
        if 0.0 < note['t'] - gap_start <= self.GAP_ONSET and gap_start - self.hnotes[-2]['t'] >= 0.3:
            note['t_heard'], note['t'] = note['t'], gap_start

    def process(self, x):
        y = super().process(x)
        if self.arr is None:
            h0 = self._held_start()
            if h0 is not None:
                self._provisional(h0)
        return y

    def _held_start(self):
        """Her voice has held one pitch for the last 0.3 s (>= 80 % voiced, p10-p90 within 1.2 semitone): the earliest
        of her latest run of notes at that pitch (a scooped or split note counts as one) is the candidate first note,
        with the held pitch as its pitch. Clicks (0.25 s, then silence) never qualify."""
        n = self.PROV_FRAMES
        if len(self.ft_v) < n:
            return None
        ms = [mm for vv, mm in zip(self.ft_v[-n:], self.ft_m[-n:]) if vv and mm > 0]
        if len(ms) < 0.8 * n or np.percentile(ms, 90) - np.percentile(ms, 10) > 1.2:
            return None
        med = float(np.median(ms))
        first = None
        for h in reversed(self.hnotes):
            if h.get('junk'):
                continue
            if abs(h['pitch'] - med) > 1.0:
                break
            first = h
        if first is None or id(first) in self.prov_tried or self.ft_t[-1] - first['t'] > 3.0:
            return None
        self.prov_tried.add(id(first))
        return {'t': first['t'], 'pitch': med, 'fr': [med]}

    def _held_fraction(self, t0, t1, pitch):
        """Share of the tracker frames in [t0, t1) that are voiced within 1 semitone of `pitch`."""
        k0 = bisect.bisect_left(self.ft_t, t0)
        k1 = bisect.bisect_left(self.ft_t, t1)
        if k1 <= k0:
            return 0.0
        ok = sum(1 for k in range(k0, k1) if self.ft_v[k] and self.ft_m[k] > 0 and abs(self.ft_m[k] - pitch) <= 1.0)
        return ok / (k1 - k0)

    def _provisional(self, h0):
        """Her first held note (>= 0.3 s, steady) is taken as the song's first note at once: the opening chord of the
        first library song (both start on the tonic chord) comes in now, in her key, at the page tempo; her next
        different note confirms the song (or switches to the other one, or calls it a false start)."""
        sid = next(iter(self.open))
        (b0, w0), (b1, w1) = self.open[sid]
        arr = self.songs[sid]
        self.song_id, self.arr = sid, arr
        self.T = int(round((h0['pitch'] - w0) % 12.0)) % 12
        self.S = self.T if self.T <= 6 else self.T - 12
        self.u = 0.0
        self.ton, self.w_ref, self.span = [(h0, w0)], w0, []
        self.low = h0['pitch'] - (w0 + self.S + arr.get('melody_s_shift', 0)) <= -7.0
        self.mel = [(float(a), int(m)) for a, b, m in arr['melody']]
        self.parts = sorted((float(on), float(off), int(m) + self.S + (12 if (self.low and p == 'T') else 0), p)
                            for p, seq in arr['parts'].items() for on, off, m in seq)
        self.Tb, self.anchor_t, self.anchor_b = self.T0_b, h0['t'], b0
        self.pairs, self.pair_w, self.mpos = [(h0['t'], b0)], [1.0], 0
        self.next_part, self.unmatched, self.after_start = 0, 0, 0
        self.provisional, self.prov_note, self.prov_far = True, h0, 0
        self.diag.update({'prov_time': round(h0['t'] + self.PROV_FRAMES * 0.01, 3), 'start_t': round(h0['t'], 3)})
        self.log(f'provisional start: {sid}, key shift {self.S:+d}')

    def _confirm(self, note):
        h0 = self.prov_note
        if abs(note['pitch'] - h0['pitch']) < 0.6:        # she re-sang her first note: wait for the next one
            return
        dh = note['pitch'] - h0['pitch']
        if abs(dh) > 9.0:                                  # no opening interval is that wide: a tracker error ...
            note['junk'] = True
            self.prov_far += 1
            if self.prov_far >= 2:                         # ... unless it keeps happening: then the start was wrong
                self.provisional = False
                self._unbegin()
            return
        best = None
        for sid, ((b0, w0), (b1, w1)) in self.open.items():
            r = dh / (w1 - w0)
            if self.START_RATIO[0] <= r <= self.START_RATIO[1]:
                cost = abs(np.log(r))
                if best is None or cost < best[0]:
                    best = (cost, sid)
        self.provisional = False
        if best is None:                                   # no song opens like this: a false start
            self._unbegin()
            self._try_start()
            return
        sid = best[1]
        (b0, w0), (b1, w1) = self.open[sid]
        if sid == self.song_id:                            # confirmed: keep what sounds, fix tempo and place
            self.Tb = self._clamp_tb((note['t'] - h0['t']) / (b1 - b0))
            self.anchor_t, self.anchor_b = note['t'], b1
            self.pairs.append((note['t'], b1))
            self.mpos = max(i for i, (b, m) in enumerate(self.mel) if b <= b1 + 1e-9)
            if abs(w1 - w0) >= 3:
                self.span.append((note, w1))
        else:                                              # the other song: re-write the parts from it
            self._silence()
            self._begin(sid, h0, note, (b0, w0), (b1, w1))
        self.after_start = 1
        self.diag['id_time'] = round(note['t'], 3)

    def _try_start(self):
        """Her latest distinct note against each of the 3 before it (a stray tracker note may sit between the two)."""
        hd = self._her_distinct_all()
        h1 = hd[-1]
        best = None
        for k in range(2, 5):
            if len(hd) - k < self.block_before:
                break
            h0 = hd[-k]
            dh = h1['pitch'] - h0['pitch']
            if self._held_fraction(h0['t'], h1['t'], h0['pitch']) < 0.5:
                continue                                   # a song's first note is held (2 beats), clicks are not
            for sid, ((b0, w0), (b1, w1)) in self.open.items():
                r = dh / (w1 - w0)
                if self.START_RATIO[0] <= r <= self.START_RATIO[1]:
                    cost = abs(np.log(r)) + 0.1 * (k - 2)
                    if best is None or cost < best[0]:
                        best = (cost, sid, h0)
        if best is None:
            return
        sid, h0 = best[1], best[2]
        (b0, w0), (b1, w1) = self.open[sid]
        self._begin(sid, h0, h1, (b0, w0), (b1, w1))

    def _her_distinct(self):
        """choir_v5's (for relocation), without the notes the follower marked as tracker errors."""
        out = []
        for n in self.hnotes[-14:]:
            if n.get('junk') or (out and abs(n['pitch'] - out[-1]['pitch']) < 0.6):
                continue
            out.append(n)
        return out

    def _her_distinct_all(self):
        out = []
        for n in self.hnotes:
            if out and abs(n['pitch'] - out[-1]['pitch']) < 0.6:
                continue
            out.append(n)
        return out

    def _begin(self, sid, h0, h1, d0, d1):
        arr = self.songs[sid]
        (b0, w0), (b1, w1) = d0, d1
        T = int(round((h0['pitch'] - w0) % 12.0)) % 12
        self.song_id, self.arr, self.T = sid, arr, T
        self.S = T if T <= 6 else T - 12
        self.u = 0.0
        self.ton, self.w_ref = [(h0, w0)], w0
        self.span = [(h1, w1)] if abs(w1 - w0) >= 3 else []
        sop = [h0['pitch'] - (w0 + self.S + arr.get('melody_s_shift', 0)),
               h1['pitch'] - (w1 + self.S + arr.get('melody_s_shift', 0))]
        self.low = float(np.median(sop)) <= -7.0
        self.mel = [(float(a), int(m)) for a, b, m in arr['melody']]
        self.parts = sorted((float(on), float(off), int(m) + self.S + (12 if (self.low and p == 'T') else 0), p)
                            for p, seq in arr['parts'].items() for on, off, m in seq)
        self.Tb = self._clamp_tb((h1['t'] - h0['t']) / (b1 - b0))
        self.anchor_t, self.anchor_b = h1['t'], b1
        self.pairs, self.pair_w = [(h0['t'], b0), (h1['t'], b1)], [1.0, 1.0]
        self.mpos = max(i for i, (b, m) in enumerate(self.mel) if b <= b1 + 1e-9)
        self.next_part, self.unmatched, self.after_start = 0, 0, 0
        self.diag.update({'id_note': len(self.hnotes), 'id_time': round(h1['t'], 3), 'start_t': round(h0['t'], 3)})
        self.log(f'start {sid}: key shift {self.S:+d}, low {self.low}, tempo {60 / self.Tb:.1f} bpm')

    def _unbegin(self):
        """A false start: silence, forget the song, wait for a new opening pair after these notes."""
        self._silence()
        self.arr = self.song_id = self.parts = self.mel = None
        self.block_before = len(self._her_distinct_all())
        self.false_starts += 1
        self.diag['false_starts'] = self.false_starts

    # ---------------------------------------------------------------- her scale, no retuning
    def ref_her(self):
        return float(np.median([n['pitch'] - (m - self.w_ref) for n, m in self.ton]))

    def scale(self):
        r = self.ref_her()
        v = [(n['pitch'] - r) / (m - self.w_ref) for n, m in self.span[-8:]]
        return float(np.clip(np.median(v), 0.6, 1.15)) if v else 1.0

    def _tuning(self):
        return 0.0

    # ---------------------------------------------------------------- following
    def _follow(self, note):
        """choir_v5's follower (looking 6 notes ahead), plus: a note more than 5 semitones from where she should be on
        her current note AND on each of the next 6 is a tracker error (subharmonic / octave slip; v11 leaps a sixth, so
        distance to the current note alone is not enough) and is ignored, also as the 'previous note' for the moved /
        repeated test; when such
        a stray note was the attack of the next real note (< 0.3 s before it), its onset is that note's onset; a re-sung
        note (same pitch) only counts within 0.35 beat of where the beat map expects the repeat, else it is a split."""
        p, t = note['pitch'], note['t']
        cur_pitch = self.expect(self.mel[self.mpos][1])
        near = [self.expect(self.mel[self.mpos + k][1]) for k in range(0, self.AHEAD_NOTES + 1)
                if self.mpos + k < len(self.mel)]
        if min(abs(p - e) for e in near) > self.JUNK_ST:      # far from her note and every next one: a tracker error
            note['junk'] = True
            self.junk_t = t
            return
        if self.junk_t is not None and 0.0 < t - self.junk_t < 0.3:
            t = note['t'] = self.junk_t                    # the stray note was this note's attack: its onset counts
        self.junk_t = None
        good = [n for n in self.hnotes[:-1] if not n.get('junk')]
        prev = good[-1]['pitch'] if good else None
        # the search starts where the beat map says she is (minus 0.75 beat), never behind the last match: a stretch of
        # unclear notes (v11's singer sang a flat D and a sharp C# almost alike) no longer leaves the follower stuck
        b_now = self.b_of(t) - 0.75
        tpos = max((i for i, (b, m) in enumerate(self.mel) if b <= b_now), default=self.mpos)
        if tpos > self.mpos:
            self.mpos = tpos
            cur_pitch = self.expect(self.mel[self.mpos][1])
        nxt = next((i for i in range(self.mpos + 1, len(self.mel)) if self.mel[i][1] != self.mel[self.mpos][1]), None)
        # she moved on when her pitch changed, or when it is clearly nearer the next written note than the current one
        # (a flat D followed by a sharp C#: v11's singer moved by 0.2 semitone for a written semitone step)
        moved = prev is None or abs(p - prev) >= 0.6 or \
            (nxt is not None and abs(p - self.expect(self.mel[nxt][1])) < abs(p - cur_pitch) - 0.2)
        best = None
        for k in range(1, self.AHEAD_NOTES + 1):
            c = self.mpos + k
            if c >= len(self.mel):
                break
            if (self.mel[c][1] == self.mel[self.mpos][1]) == moved:
                continue
            skip, pw = 0.0, self.mel[self.mpos][1]
            for s in range(self.mpos + 1, c):              # written notes she skipped (missed or merged by the
                skip += 0.0 if self.mel[s][1] == pw else self.SKIP_COST   # tracker); timing decides if that fits
                pw = self.mel[s][1]
            x = abs(t - self.t_of(self.mel[c][0])) / self.Tb
            if not moved and x > 0.35:
                continue                                   # a re-sung note only counts on time (else: a split note)
            cost = self.mcf(p, self.expect(self.mel[c][1])) + skip + max(0.0, x - 0.5) * 1.0
            if best is None or cost < best[0]:
                best = (cost, c)
        if best is not None and best[0] <= self.ACCEPT:
            self._accept(note, best[1])
            self.after_start += 1
            return
        if not moved or abs(circ(p - cur_pitch)) < 0.6:
            return
        self.unmatched += 1
        if self.after_start == 0 and self.unmatched >= 2:
            self._unbegin()
        elif self.unmatched >= 3:
            self._relocate()

    def _accept(self, note, c):
        """choir_v5's update (beat map, her scale), with the tempo kept within 0.5-1.6 x the SONG's own tempo (not the
        page's: v11 is a 122 bpm song started from an 80 bpm page) and no retuning."""
        t, b = note['t'], self.mel[c][0]
        pred = self.t_of(b)
        self.diag['pred_err_ms'].append(round((t - pred) * 1000, 1))
        # a short written note (a pickup or passing note) is sung less exactly than a long one: it moves the beat map
        # less (weight = its written length in beats, 0.25-1); v11's 0.25-beat pickups had pulled the next chord early
        dur = (self.mel[c + 1][0] - b) if c + 1 < len(self.mel) else 1.0
        wn = float(np.clip(dur / self.FULL_WEIGHT_BEATS, 0.25, 1.0)) if self.DUR_WEIGHT else 1.0
        self.pairs.append((t, b))
        while len(self.pair_w) < len(self.pairs) - 1:
            self.pair_w.append(1.0)
        del self.pair_w[len(self.pairs) - 1:]
        self.pair_w.append(wn)
        # tempo from a long stretch (rubato is not a tempo change: v11's singer stretched one note and the 4-note fit
        # jumped from 122 to 131 bpm, then back to 117, so the chords swung from 0.18 s early to 0.13 s late)
        n, span, gain, phase = self.TEMPO_WINDOW, self.TEMPO_MIN_SPAN, self.TEMPO_GAIN, self.PHASE_GAIN
        self.res_hist.append(t - pred)
        r = self.res_hist[-3:]
        trend = self.TEMPO_ADAPTIVE and len(r) == 3 and (all(e > 0 for e in r) or all(e < 0 for e in r)) \
            and float(np.median(np.abs(r))) > 0.06 * self.Tb
        warm = len(self.pairs) <= 2 + self.WARMUP_NOTES  # just started: the two-note tempo is a rough guess
        if trend or warm:                              # 3 notes in a row on the same side: a real tempo change,
            n, span, gain, phase = 4, 1.5, 0.6, 0.8    # follow it quickly (else: rubato around a steady tempo)
            self.diag['trend_updates' if trend else 'warm_updates'] = \
                self.diag.get('trend_updates' if trend else 'warm_updates', 0) + 1
        pts, pw = self.pairs[-n:], np.array(self.pair_w[-n:])
        if len(pts) >= 3 and pts[-1][1] - pts[0][1] >= span:
            if self.TEMPO_ROBUST:                      # Theil-Sen: median of the pairwise beat lengths >= 1 beat apart
                sl = [(pts[j][0] - pts[i][0]) / (pts[j][1] - pts[i][1]) for i in range(len(pts))
                      for j in range(i + 1, len(pts)) if pts[j][1] - pts[i][1] >= 1.0]
                slope = float(np.median(sl)) if sl else self.Tb
            else:
                rec = np.linspace(1.0, 2.0, len(pts)) if self.RECENCY else np.ones(len(pts))
                slope = float(np.polyfit([q for _, q in pts], [s for s, _ in pts], 1, w=np.sqrt(rec * pw))[0])
            self.Tb += gain * (self._clamp_tb(slope) - self.Tb)
        if self.PHASE_JUMP and abs(t - pred) > self.PHASE_JUMP * self.Tb:
            phase, wn = 1.0, 1.0                       # a big miss is a re-entry (after a breath), not noise: realign
            self.diag['phase_jumps'] = self.diag.get('phase_jumps', 0) + 1
        self.anchor_t, self.anchor_b = pred + phase * wn * (t - pred), b
        w = self.mel[c][1]
        if w % 12 == self.arr['tonic'] % 12:
            self.ton.append((note, w))
        elif abs(w - self.w_ref) >= 3:
            self.span.append((note, w))
        self.u = 0.0
        self.diag['tune_track'].append((0.0, round(self.scale(), 3)))
        self.mpos, self.unmatched = c, 0

    def _clamp_tb(self, tb):
        nominal = 60.0 / float(self.arr.get('bpm', 60.0 / self.T0_b))
        return float(np.clip(tb, nominal / 1.6, nominal / 0.5))

    def summary(self):
        s = super().summary()
        s.update({'choir_engine': 'choir_v6', 'false_starts': self.false_starts})
        return s
