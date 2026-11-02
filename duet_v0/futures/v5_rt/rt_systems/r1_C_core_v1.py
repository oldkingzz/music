"""r1_C_core_v1 (run v5rt_1101, round 1, idea C): HELPER, not a scored system (no build()). The decision logic of
the r1_C_* systems; each scored file r1_C_<name>_v<k>.py holds its own settings as constants and passes them here.

Duet = r2_mix_v2 (m1_filter_v5lo's HMM chord filter x 3-GRU ensemble, log-linear 0.3/0.7, m1's chooser), built as
bench3 builds it, plus the learned joint forecaster of r1_C_fnet_v1 (trained on train only).

hear(i, onset, midi, prev_dur): the forecaster's GRU reads the heard note; r2_mix_v2's step() gives the reactive
  choice h_r (called exactly once per note, as its GRU state requires). If the sounding note was attacked by plan()
  since the last hear, it stays when it is below her heard note, consonant with it and not her unison/octave, and
  P(chord tone) of it under r2's belief after the hear is at least P(chord tone) of h_r minus KEEP_MARGIN (then m1's
  voice leading follows it: f.prev = it); otherwise Duet switches to h_r.
plan(q, heard): only from beat Q_MIN on, after her first heard note. The forecaster gives p_on = P(a note of hers
  starts at q) and the joint P(chord, pc) of that note; the chord marginal may be mixed log-linearly with r2's own
  forecast for beat int(q) (W_R2; r2's HMM run over the unfinished beats and one beat on, its GRU belief pushed on by
  the HMM table when her last note started before that beat; evidence of beat int(q) only when q is mid-beat).
  Her next note's pitch for each pc = the one nearest her last heard note. For every alto note h:
     hit(h)   = p_on * E[h consonant (not unison/octave) with her next note AND a tone of the chord]
                + (1 - p_on) * [she still holds her heard note] * E[h consonant with that note AND a chord tone]
     harsh(h) = the same weights on P(2nd / tritone / 7th)
     score(h) = hit(h) - LAM * harsh(h) - W_ABOVE * p_on * P(h at or above her next note) - W_STEP * |h - sounding|
  The sounding note's own value uses LAM_KEEP (an old note ringing on is far less audible than a new wrong attack).
  Duet attacks the best new note only if p_on >= P_ON_MIN, its harsh(h) <= HARSH_MAX and its score beats keeping the
  sounding note by GAIN_MIN; otherwise plan() keeps the sounding note (None).
"""
import os

import numpy as np
import torch

from rt_systems.r1_C_fnet_v1 import NF, NpNet, note_feat, query_feat
from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, CONSONANT, CT

ABS = '/Users/san/claude_sandbox/music/duet_v0'
MODELS = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'models')
GOOD = np.array([(i in CONSONANT and i != 0) for i in range(12)], dtype=np.float64)
HARSH = np.array([i in (1, 2, 6, 10, 11) for i in range(12)], dtype=np.float64)
HS = np.arange(ALTO[0], ALTO[1] + 1)
CTM = np.zeros((7, 12))
for _z in range(7):
    for _p in CT[_z]:
        CTM[_z, _p] = 1.0


class JointRT:
    def __init__(self, mix, net, cfg):
        self.f = mix                                     # r2_mix_v2.Mix (an m1_filter_v5lo.Filter)
        self.F = net                                     # r1_C_fnet_v1.NpNet
        self.c = dict(cfg)
        self.trace = None                                # a list to record plan-time forecasts (diagnostics only)

    def start(self, meta):
        self.f.start(meta)
        self.t, self.M, self.pos = meta['tonic'], meta['M'], meta['pos']
        self.s = self.F.zero()
        self.x = np.zeros(NF)
        self.fed = 0
        self.h = -1                                      # the note Duet is sounding
        self.planned = False                             # it was attacked by plan() since the last hear
        self.ctm_h = CTM[:, (HS - self.t) % 12].T        # [alto note, chord]: is h a tone of chord z

    def _posk(self, k):
        return self.pos[k] if k < len(self.pos) else 0

    # ---------------------------------------------------------------- hear
    def hear(self, i, onset, midi, prev_dur):
        k = int(onset)
        x = note_feat((midi - self.t) % 12, midi, prev_dur, self._posk(k), onset - k, self.M).astype(np.float64)
        self.s = self.F.step(self.s, x)
        self.x = x
        self.fed += 1
        h_r = int(self.f.step(i, onset, midi, prev_dur))
        if self.planned and self.h >= 0 and self.h != h_r and self._keep(k, midi, h_r):
            self.f.prev = self.h                         # kept: m1's voice leading follows what Duet sings
            self.planned = False
            return None
        self.planned = False
        self.h = h_r
        return h_r

    def _keep(self, k, midi, h_r):
        h = self.h
        if h >= midi or not GOOD[abs(midi - h) % 12]:
            return False
        if h_r < 0:
            return True
        pfit = self.f.belief(k) @ CTM
        return pfit[(h - self.t) % 12] >= pfit[(h_r - self.t) % 12] - self.c['KEEP_MARGIN']

    # ---------------------------------------------------------------- plan
    def _r2_forecast(self, q):
        """r2_mix_v2's chord belief for beat int(q) from the notes answered so far (no new evidence at q)."""
        f, k = self.f, int(q)
        a = f.alpha
        for j in range(f.done, k):
            a = M1.Filter._advance(f, a, j, f._beat_ll(j))
        ll = f._beat_ll(k) if (q > k and f.done <= k) else np.zeros(M1.S)
        bh = M1.Filter._advance(f, a, k, ll)
        if f._p is None:
            return bh
        bg = np.asarray(f._p, dtype=np.float64)
        if f.notes and int(np.floor(f.notes[-1][0] + 1e-9)) < k:
            bg = bg @ f.m.A[M1.tkey(f._pos(k), f.M)]
        lg = (1 - f.a) * np.log(np.maximum(bh, 1e-300)) + f.a * np.log(np.maximum(bg, 1e-300))
        w = np.exp(lg - lg.max())
        return w / w.sum()

    def plan(self, q, heard):
        c = self.c
        if q < c['Q_MIN'] or not heard or len(heard) != self.fed:
            return None
        on, off, m_last = heard[-1]
        kq = int(q)
        qf = query_feat(q, (on, off, m_last), self.M, self._posk(kq)).astype(np.float64)
        p_on, _, p_z, P = self.F.query(self.s, self.x, qf)
        if c['W_R2'] > 0 and (p_on >= c['P_ON_MIN'] or self.trace is not None):
            br2 = self._r2_forecast(q)
            pz = np.maximum(P.sum(1), 1e-12)
            lg = (1 - c['W_R2']) * np.log(pz) + c['W_R2'] * np.log(np.maximum(br2, 1e-300))
            w = np.exp(lg - lg.max())
            P = P * (w / w.sum() / pz)[:, None]
            lz = (1 - c['W_R2']) * np.log(np.maximum(p_z, 1e-12)) + c['W_R2'] * np.log(np.maximum(br2, 1e-300))
            p_z = np.exp(lz - lz.max())
            p_z /= p_z.sum()
        if self.trace is not None:
            self.trace.append((q, p_on, P.sum(0), P.sum(1), p_z))
        if p_on < c['P_ON_MIN']:
            return None
        P = P / P.sum()
        p_pc = P.sum(0)
        mp = m_last + ((np.arange(12) + self.t - m_last + 6) % 12) - 6       # her next pitch per relative pc
        iv = np.abs(mp[None, :] - HS[:, None]) % 12                          # [alto note, pc]
        good, harsh = GOOD[iv], HARSH[iv]
        hit = p_on * (((self.ctm_h @ P) * good).sum(1))
        bad = p_on * (harsh @ p_pc)
        above = p_on * ((HS[:, None] >= mp[None, :]) @ p_pc)
        if off is None and c['HOLD']:                                        # she may still hold her heard note
            ivh = np.abs(m_last - HS) % 12
            hit = hit + (1 - p_on) * GOOD[ivh] * (self.ctm_h @ p_z)
            bad = bad + (1 - p_on) * HARSH[ivh]
        base = hit - c['W_ABOVE'] * above
        score = base - c['LAM'] * bad
        h0 = self.h
        if h0 >= 0:
            j0 = h0 - ALTO[0]
            if 0 <= j0 < len(HS):
                keep = base[j0] - c['LAM_KEEP'] * bad[j0]
            else:
                keep = -c['LAM_KEEP']
            score = score - c['W_STEP'] * np.abs(HS - h0)
            score[(HS == h0)] = -np.inf
        else:
            keep = 0.0
        score[bad > c['HARSH_MAX']] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]) or score[j] < keep + c['GAIN_MIN']:
            return None
        self.h = int(HS[j])
        self.planned = True
        return self.h


def build_with(train, cfg):
    torch.set_num_threads(1)
    mix = r2_mix_v2.build(train)
    sd = torch.load(os.path.join(MODELS, cfg['CKPT']))
    return JointRT(mix, NpNet(sd), cfg)
