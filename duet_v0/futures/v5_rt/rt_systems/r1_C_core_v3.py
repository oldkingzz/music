"""r1_C_core_v3 (run v5rt_1101, round 1, idea C): HELPER, not a scored system (no build()). r1_C_core_v2 (calibrated
joint forecaster, optional onset attacks, held-note re-voicing at beat boundaries) plus three switches, each off
when its constant is 0 (then the decisions are r1_C_core_v2's):

REV_HALF  re-voicing may also happen at a half-beat plan point q = k + 0.5 (her held note started before beat k and
          Duet did not move at k, e.g. because a new note at k was too likely); same rules as at k.
W_REV_FC  the chord belief used for re-voicing = r2_mix_v2's belief for beat k (held note as evidence) mixed
          log-linearly with the forecaster's own chord head at q, weight W_REV_FC on the forecaster.
SUS_POST  wary ear at hear(): the forecaster's prior for this note (its P(pc) at the last plan point) and the 10 %
          mishearing model give a posterior of her TRUE pitch class; when P(true = heard) < SUS_POST, Duet is still
          sounding a note and the reactive choice h_r differs from it, Duet keeps the sounding note (no new attack)
          unless h_r has the lower posterior P(harsh). m1's voice leading then follows the kept note.
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
PCS = np.arange(12)


def _loglin(a, b, w):
    lg = (1 - w) * np.log(np.maximum(a, 1e-300)) + w * np.log(np.maximum(b, 1e-300))
    e = np.exp(lg - lg.max())
    return e / e.sum()


class JointRT3:
    def __init__(self, mix, net, cfg):
        self.f = mix                                     # r2_mix_v2.Mix (an m1_filter_v5lo.Filter)
        self.F = net                                     # r1_C_fnet_v1.NpNet
        self.c = dict(cfg)
        self.trace = None                                # diagnostics only: plan-time forecasts
        self.atr = None                                  # diagnostics only: every attack made by plan()
        self.n_sus = 0                                   # diagnostics only: wary-ear holds

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
        self.rev_k = -1                                  # beat of the last re-voice

    def _posk(self, k):
        return self.pos[k] if k < len(self.pos) else 0

    def _near(self, m):
        """Her pitch for each relative pc, the one nearest m (ties below)."""
        return m + ((PCS + self.t - m + 6) % 12) - 6

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
        if self.c['SUS_POST'] > 0 and self.h >= 0 and h_r != self.h and self.post[hp] < self.c['SUS_POST']:
            harsh = HARSH[np.abs(self._near(midi)[None, :] - np.array([self.h, max(h_r, 0)])[:, None]) % 12] @ self.post
            if h_r < 0 or harsh[0] <= harsh[1]:
                self.f.prev = self.h
                self.n_sus += 1
                return None
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
        return _loglin(bh, bg, f.a)

    def _calibrate(self, P):
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
        half_ok = c['REV_HALF'] and q > kq and self.rev_k != kq
        rev = (c['REV'] and off is None and (q == kq or half_ok) and int(on) < kq and p_on <= c['REV_PON_MAX'])
        att = c['ATT_ON'] and p_on >= c['P_ON_MIN']
        p_z_fc = p_z
        if c['W_R2'] > 0 and (att or self.trace is not None):
            br2 = self._r2_belief(q, q > kq)
            pz = np.maximum(P.sum(1), 1e-12)
            P = P * (_loglin(pz, br2, c['W_R2']) / pz)[:, None]
            p_z = _loglin(p_z, br2, c['W_R2'])
        P = P / P.sum()
        p_pc = P.sum(0)
        self.prior = p_pc
        if self.trace is not None:
            self.trace.append((q, p_on, p_pc, P.sum(1), p_z))
        if rev:
            return self._revoice(q, kq, p_on, p_pc, m_last, p_z_fc)
        if not att:
            return None
        mp = self._near(m_last)
        iv = np.abs(mp[None, :] - HS[:, None]) % 12
        good, harsh = GOOD[iv], HARSH[iv]
        hit = p_on * (((self.ctm_h @ P) * good).sum(1))
        bad_on = harsh @ p_pc
        bad = p_on * bad_on
        above = p_on * ((HS[:, None] >= mp[None, :]) @ p_pc)
        if off is None and c['HOLD']:
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

    def _revoice(self, q, k, p_on, p_pc, m_last, p_z_fc):
        c = self.c
        h0 = self.h
        if h0 < 0:
            return None
        b = self._r2_belief(q, True)
        if c['W_REV_FC'] > 0:
            b = _loglin(b, p_z_fc, c['W_REV_FC'])
        pfit = b @ CTM
        ivh = np.abs(m_last - HS) % 12
        ok = (GOOD[ivh] > 0) & (HS < m_last)
        if not ok.any():
            return None
        val = pfit[(HS - self.t) % 12] - c['W_STEP'] * np.abs(HS - h0)
        iv0 = abs(m_last - h0) % 12
        v0 = pfit[(h0 - self.t) % 12] if (GOOD[iv0] and h0 < m_last) else 0.0
        harsh = HARSH[np.abs(self._near(m_last)[None, :] - HS[:, None]) % 12]
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
        self.rev_k = k
        if self.atr is not None:
            self.atr.append((q, 'rev', self.h, p_on, float(bad_on[j]), float(bad[j])))
        return self.h


def build_with(train, cfg):
    torch.set_num_threads(1)
    mix = r2_mix_v2.build(train)
    sd = torch.load(os.path.join(MODELS, cfg['CKPT']))
    return JointRT3(mix, NpNet(sd), cfg)
