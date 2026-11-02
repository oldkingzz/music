"""r0_predict_live_v1 (run v5rt_1101, phase 1 baseline; not a new idea): the hand-built live look-ahead of
futures/v4_selfiter/live_predict.py (class PredictDuet) ported to bench_rt with its Twinkle-chosen settings
CHOICE 'mimic', KEEP 'consonant', W_CUR 0. It wraps r2_mix_v2 (m1_filter_v5lo's HMM chord filter x the 3-GRU ensemble,
log-linear 0.3/0.7, then m1's chooser), built from the v3 systems exactly as bench3 builds it.

plan(q, heard): only at integer beats k = q, only after her first note has been answered (live: t0 is set), and only
  while she is singing: her last heard note is still sounding (its offset not heard yet) or ended less than 1 beat
  before q. Forecast of the chord of beat k (live _forecast with W_CUR 0): r2_mix_v2's HMM filter run on over the beats
  not finished yet with the notes answered so far (a note whose end is unknown counts to the end of each beat), then one
  beat on by its transition table with no evidence for beat k; the GRU ensemble's belief at her last answered note,
  pushed one beat on by the same table when that note started before beat k; the two mixed log-linearly 0.3/0.7.
  Choice 'mimic' (live _mimic): m1's chooser on that belief as if she repeats her last heard note: notes consonant with
  it and below it, P(chord tone) minus m1's w_oct (her octave), w_step * distance from the note Duet is sounding,
  w_big for leaps over 7 and w_mid for distance from the middle of the alto range (no parallel-motion term). Duet sounds
  it from q (-1 = silence if no note qualifies).
hear(i, onset, midi, prev_dur): r2_mix_v2's own step() (the reactive choice h_r; called exactly once per note, as the
  GRU state requires). KEEP 'consonant': unless this is her first note, if the sounding note is >= 0, below her heard
  note, consonant with it and not her unison/octave, it stays, and the filter's prev is set to it so m1's voice leading
  follows what Duet actually sings; otherwise Duet switches to h_r.
Not ported, because bench_rt already models it or it is not part of the decision: the pitch tracker and note
segmentation, onset snapping, the click, PRE / output-latency timing (a bench_rt plan point is already the moment the
note must sound), and the live engine's release of Duet's note 0.25 s after she stops (bench_rt's reactive adapter
holds the note too, and rests are not scored).
"""
import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID, CONSONANT

SINGING = 1.0          # beats: plan only if her last heard note ended less than this long before q (or still sounds)


class PredictLive:
    def __init__(self, mix):
        self.f = mix                                     # r2_mix_v2.Mix, an m1_filter_v5lo.Filter

    def start(self, meta):
        self.f.start(meta)
        self.h = -1                                      # the note Duet is sounding
        self.answered = 0                                # her notes answered so far

    def plan(self, q, heard):
        if q != int(q) or not heard or not self.answered:
            return None
        on, off, midi = heard[-1]
        if off is not None and q - off >= SINGING:
            return None                                  # she is not singing: no plan, Duet keeps its note
        self.h = self._mimic(self._forecast(int(q)), int(midi))
        return self.h

    def hear(self, i, onset, midi, prev_dur):
        h_r = int(self.f.step(i, onset, midi, prev_dur))
        first = self.answered == 0
        self.answered += 1
        iv = abs(midi - self.h) % 12 if self.h >= 0 else None
        if not first and self.h >= 0 and self.h < midi and iv in CONSONANT and iv != 0:
            self.f.prev = self.h                         # kept: m1's voice leading follows what Duet sings
            return None
        self.h = h_r
        return h_r

    def _forecast(self, k):
        """live_predict.PredictDuet._forecast with W_CUR = 0."""
        f = self.f
        a = f.alpha
        for j in range(f.done, k):
            a = M1.Filter._advance(f, a, j, f._beat_ll(j))
        bh = M1.Filter._advance(f, a, k, np.zeros(M1.S))
        if f._p is None:
            return bh
        bg = np.asarray(f._p, dtype=np.float64)
        if f.notes and int(np.floor(f.notes[-1][0] + 1e-9)) < k:
            bg = bg @ f.m.A[M1.tkey(f._pos(k), f.M)]
        lg = (1 - f.a) * np.log(np.maximum(bh, 1e-300)) + f.a * np.log(np.maximum(bg, 1e-300))
        w = np.exp(lg - lg.max())
        return w / w.sum()

    def _mimic(self, b, midi):
        """live_predict.PredictDuet._mimic: m1's chooser on belief b with her note `midi` standing in for her next one."""
        f = self.f
        pfit = b @ M1.CTM
        ref = self.h if self.h >= 0 else (f.prev if f.prev is not None else ALTO_MID)
        best, bs = -1, -1e9
        for h in range(ALTO[0], min(ALTO[1], midi - 1) + 1):
            iv = abs(midi - h) % 12
            if iv not in CONSONANT:
                continue
            s = pfit[(h - f.t) % 12]
            if iv == 0:
                s -= M1.P['w_oct']
            d = abs(h - ref)
            s -= M1.P['w_step'] * d + M1.P['w_big'] * (d > 7) + M1.P['w_mid'] * abs(h - ALTO_MID)
            if s > bs:
                best, bs = h, s
        return best


def build(train):
    return PredictLive(r2_mix_v2.build(train))
