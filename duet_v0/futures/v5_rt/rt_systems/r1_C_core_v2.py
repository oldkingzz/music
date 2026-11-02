"""r1_C_core_v2 (run v5rt_1101, round 1, idea C): HELPER, not a scored system (no build()). r1_C_core_v1's decision
logic plus three things, each switched by the scored file's constants:

1. Calibration of the forecaster's pitch: P(pc | chord) is tempered (power 1 / T_PC) and floored (EPS_PC of uniform)
   before any expected hit / harsh is computed (the trained net is over-confident, mostly on bach).
2. Onset attacks (ATT_ON = 1) exactly as r1_C_core_v1.plan, on the calibrated joint.
3. Re-voicing of a HELD note at a beat boundary (REV = 1): at an integer plan point q = k, when her last heard note is
   still sounding (offset not heard) and started before beat k, and the forecaster says a new note at q is unlikely
   (p_on <= REV_PON_MAX), Duet may move to another alto note h below her held note and consonant with it (not
   unison/octave), when P(h is a chord tone of beat k) - P(sounding note is one) >= REV_GAIN under r2_mix_v2's belief
   for beat k (its HMM run over the unfinished beats WITH her held note as evidence for beat k, its GRU belief pushed
   one beat on by the HMM table, mixed 0.3/0.7). Risk check: p_on * P(h harsh against a new note) + (1 - p_on) *
   P(h harsh against her TRUE held note) <= REV_HARSH_MAX, where her true held pitch class is the posterior from the
   forecaster's prior for that note (the last plan point at or before its onset) and the 10 % mishearing model.
   m1's voice leading then follows the new note (f.prev).
hear() is r1_C_core_v1's: r2_mix_v2's step() once per note; a note attacked by an onset plan stays if below her heard
note, consonant (not unison/octave) and its P(chord tone) >= that of the reactive choice minus KEEP_MARGIN.
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
NOISE = np.full((12, 12), 0.1 / 11)                      # N[true, heard]: bench's 10 % mishearing
np.fill_diagonal(NOISE, 0.9)


class JointRT2:
    def __init__(self, mix, net, cfg):
        self.f = mix                                     # r2_mix_v2.Mix (an m1_filter_v5lo.Filter)
        self.F = net                                     # r1_C_fnet_v1.NpNet
        self.c = dict(cfg)
        self.trace = None                                # diagnostics only: plan-time forecasts
        self.atr = None                                  # diagnostics only: every attack made by plan()

    def start(self, meta):
        self.f.start(meta)
        self.t, self.M, self.pos = meta['tonic'], meta['M'], meta['pos']
        self.s = self.F.zero()
        self.x = np.zeros(NF)
        self.fed = 0
        self.h = -1                                      # the note Duet is sounding
        self.planned = False                             # it was attacked by an onset plan since the last hear
        self.ctm_h = CTM[:, (HS - self.t) % 12].T        # [alto note, chord]: is h a tone of chord z
        self.prior = np.full(12, 1 / 12)                 # forecaster's P(pc) at the last plan point
        self.post = np.full(12, 1 / 12)                  # posterior of her current note's TRUE pc (relative)

    def _posk(self, k):
        return self.pos[k] if k < len(self.pos) else 0

    # ---------------------------------------------------------------- hear
    def hear(self, i, onset, midi, prev_dur):
        k = int(onset)
        hp = (midi - self.t) % 12
        x = note_feat(hp, midi, prev_dur, self._posk(k), onset - k, self.M).astype(np.float64)
        self.s = self.F.step(self.s, x)
        self.x = x
        self.fed += 1
        w = self.prior * NOISE[:, hp]
        self.post = w / w.sum()
        h_r = int(self.f.step(i, onset, midi, prev_dur))
        if self.planned and self.h >= 0 and self.h != h_r and self._keep(k, midi, h_r):
            self.f.prev = self.h
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
    def _r2_belief(self, q, evidence):
        """r2_mix_v2's chord belief for beat int(q) from the notes answered so far; evidence = count her notes
        heard so far (the sounding one up to the end of the beat) as evidence for that beat."""
        f, k = self.f, int(q)
        a = f.alpha
        for j in range(f.done, k):
            a = M1.Filter._advance(f, a, j, f._beat_ll(j))
        ll = f._beat_ll(k) if (evidence and f.done <= k) else np.zeros(M1.S)
        bh = M1.Filter._advance(f, a, k, ll)
        if f._p is None:
            return bh
        bg = np.asarray(f._p, dtype=np.float64)
        if f.notes and int(np.floor(f.notes[-1][0] + 1e-9)) < k:
            bg = bg @ f.m.A[M1.tkey(f._pos(k), f.M)]
        lg = (1 - f.a) * np.log(np.maximum(bh, 1e-300)) + f.a * np.log(np.maximum(bg, 1e-300))
        w = np.exp(lg - lg.max())
        return w / w.sum()

    def _calibrate(self, P):
        """Temper and floor P(pc | chord) row by row; the chord marginal is kept."""
        c = self.c
        pz = np.maximum(P.sum(1), 1e-12)
        cond = np.maximum(P / pz[:, None], 1e-12) ** (1.0 / c['T_PC'])
        cond /= cond.sum(1, keepdims=True)
        cond = (1 - c['EPS_PC']) * cond + c['EPS_PC'] / 12
        return cond * pz[:, None]

    def plan(self, q, heard):
        c = self.c
        if q < c['Q_MIN'] or not heard or len(heard) != self.fed:
            return None
        on, off, m_last = heard[-1]
        kq = int(q)
        qf = query_feat(q, (on, off, m_last), self.M, self._posk(kq)).astype(np.float64)
        p_on, _, p_z, P = self.F.query(self.s, self.x, qf)
        P = self._calibrate(P)
        rev = (c['REV'] and off is None and q == kq and int(on) < kq and p_on <= c['REV_PON_MAX'])
        att = c['ATT_ON'] and p_on >= c['P_ON_MIN']
        if c['W_R2'] > 0 and (att or self.trace is not None):
            br2 = self._r2_belief(q, q > kq)
            pz = np.maximum(P.sum(1), 1e-12)
            lg = (1 - c['W_R2']) * np.log(pz) + c['W_R2'] * np.log(np.maximum(br2, 1e-300))
            w = np.exp(lg - lg.max())
            P = P * (w / w.sum() / pz)[:, None]
            lz = (1 - c['W_R2']) * np.log(np.maximum(p_z, 1e-12)) + c['W_R2'] * np.log(np.maximum(br2, 1e-300))
            p_z = np.exp(lz - lz.max())
            p_z /= p_z.sum()
        P = P / P.sum()
        p_pc = P.sum(0)
        self.prior = p_pc
        if self.trace is not None:
            self.trace.append((q, p_on, p_pc, P.sum(1), p_z))
        if rev:
            return self._revoice(q, kq, p_on, p_pc, m_last)
        if not att:
            return None
        mp = m_last + ((np.arange(12) + self.t - m_last + 6) % 12) - 6       # her next pitch per relative pc
        iv = np.abs(mp[None, :] - HS[:, None]) % 12                          # [alto note, pc]
        good, harsh = GOOD[iv], HARSH[iv]
        hit = p_on * (((self.ctm_h @ P) * good).sum(1))
        bad_on = harsh @ p_pc
        bad = p_on * bad_on
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
            keep = base[j0] - c['LAM_KEEP'] * bad[j0] if 0 <= j0 < len(HS) else 0.0
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
        if self.atr is not None:
            self.atr.append((q, 'on', self.h, p_on, float(bad_on[j]), float(bad[j])))
        return self.h

    def _revoice(self, q, k, p_on, p_pc, m_last):
        c = self.c
        h0 = self.h
        if h0 < 0:
            return None
        pfit = self._r2_belief(q, True) @ CTM
        ivh = np.abs(m_last - HS) % 12
        ok = (GOOD[ivh] > 0) & (HS < m_last)
        if not ok.any():
            return None
        val = pfit[(HS - self.t) % 12] - c['W_STEP'] * np.abs(HS - h0)
        iv0 = abs(m_last - h0) % 12
        v0 = pfit[(h0 - self.t) % 12] if (GOOD[iv0] and h0 < m_last) else 0.0
        # risk: a new note at q (forecast pc) or her TRUE held note (posterior), each with octave next to m_last
        mp = m_last + ((np.arange(12) + self.t - m_last + 6) % 12) - 6
        harsh = HARSH[np.abs(mp[None, :] - HS[:, None]) % 12]
        bad_on = harsh @ p_pc
        bad_hold = harsh @ self.post
        bad = p_on * bad_on + (1 - p_on) * bad_hold
        val[~ok] = -np.inf
        val[HS == h0] = -np.inf
        val[bad > c['REV_HARSH_MAX']] = -np.inf
        j = int(np.argmax(val))
        if not np.isfinite(val[j]) or val[j] < v0 + c['REV_GAIN']:
            return None
        self.h = int(HS[j])
        self.f.prev = self.h
        self.planned = False
        if self.atr is not None:
            self.atr.append((q, 'rev', self.h, p_on, float(bad_on[j]), float(bad[j])))
        return self.h


def build_with(train, cfg):
    torch.set_num_threads(1)
    mix = r2_mix_v2.build(train)
    sd = torch.load(os.path.join(MODELS, cfg['CKPT']))
    return JointRT2(mix, NpNet(sd), cfg)
