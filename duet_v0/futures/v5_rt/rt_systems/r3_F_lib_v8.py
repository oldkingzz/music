"""r3_F_lib_v8 (run v5rt_1101, round 3, idea F 'stumble guard'): LIBRARY (no build(): not a system itself). Val only,
tuned on val (no test split). rt_systems.r3_F_lib_v5.PlanF5 (read r3_F_lib_v1 .. v5; the copy-backed exception
post_copy is kept) plus a risk bound on the posterior hear's guesses, off by default (then the decisions are
r3_F_lib_v5's exactly):
  post_doubt_risk  when the posterior hear doubts the heard pitch (not backed by the copy model) and would ATTACK a new
                   note, it does so only if that note's probability of a 2nd / tritone / 7th against her true pitch
                   (under the posterior) is <= post_doubt_risk; otherwise Duet falls silent (or stays silent) until her
                   next note. r3_F_lib_v7's post_doubt_sil (always silent when the sounding note is harsh against the
                   heard pitch) cost too much hit time; this waits only on risky guesses (on the Twinkle phone file:
                   A3 against a posterior that gave her true G4 7 %).
"""
import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID

from rt_systems import r2_lib_v3 as LIB
from rt_systems import r3_F_lib_v5 as F5
from rt_systems.r2_lib_v1 import E_ALL, GOODV, HARSHV, HV, Prior, load_copy_tab


class PlanF8(F5.PlanF5):
    def _hear_post(self, i, onset, midi, prev_dur, post):
        c, f = self.c, self.f
        thr = c.get('post_copy', 2.0)
        pred = self.pred_cur
        backed = False
        if thr <= 1.0 and pred is not None:
            jj = int(np.argmax(post))
            backed = E_ALL[jj] != 0 and midi + int(E_ALL[jj]) == pred['top'] and post[jj] >= thr
        if backed:
            self.stats['hear_post_copy'] += 1
        flag = (lambda k: 0) if backed else (lambda k: c.get(k, 0))
        hs = self.h
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        hs_harsh = keep_ok and HARSHV[(midi - hs) % 12] > 0
        if flag('post_veto') and keep_ok and not hs_harsh:
            f.step(i, onset, midi, prev_dur)
            self.her.append((onset, midi))
            self.planned = False
            f.prev = hs
            self.stats['hear_post_veto'] += 1
            return None
        prev = f.prev
        f.step(i, onset, midi, prev_dur)                  # r2_mix_v2's step: GRU state and filter, exactly once
        self.her.append((onset, midi))
        self.planned = False
        k = int(onset)
        pfit = self._belief(k) @ M1.CTM
        mt = midi + E_ALL
        ivm = (mt[:, None] - HV[None, :]) % 12
        G, Hh = GOODV[ivm], HARSHV[ivm]
        X = (mt[:, None] <= HV[None, :]).astype(np.float64)
        ph = post @ Hh
        P = M1.P
        d = np.abs(HV - prev) if prev is not None else np.zeros(len(HV))
        s = (pfit[(HV - self.t) % 12] * (post @ G) - c['lam_h'] * ph - c.get('w_x_post', c['w_x']) * (post @ X)
             - P['w_step'] * d - P['w_big'] * (d > 7) - P['w_mid'] * np.abs(HV - ALTO_MID))
        if c.get('hedge_post', 0):
            w = self._w_next(onset)
            if w > 0:
                s = s + w * self._u_next(onset, midi, k)
        att = s - c['kappa_h'] * ph
        if flag('post_nh'):
            att = np.where(HARSHV[(midi - HV) % 12] > 0, -np.inf, att)
        if flag('post_below'):
            att = np.where(HV >= midi, -np.inf, att)
        if keep_ok and c.get('st_post', 0) > 0 and self._recent(onset):
            att = att - c['st_post']
        if keep_ok and not (flag('post_nh_keep') and hs_harsh):
            att[hs - ALTO[0]] = s[hs - ALTO[0]]
        elif keep_ok:
            att[hs - ALTO[0]] = -np.inf
        j = int(np.argmax(att))
        if c['hear_sil'] and att[j] < 0:
            self.stats['hear_post_sil'] += 1
            self.h = -1
            return -1 if hs >= 0 else None
        if not np.isfinite(att[j]):
            self.stats['hear_post_none'] += 1
            self.h = -1
            return -1 if hs >= 0 else None
        h = int(HV[j])
        if keep_ok and h == hs:
            f.prev = h
            self.stats['hear_post_keep'] += 1
            return None
        if not backed and c.get('post_doubt_risk', 2.0) <= 1.0 and ph[j] > c['post_doubt_risk']:
            f.prev = hs if hs >= 0 else prev               # Duet's last sung note, as m1 keeps it on a silence
            self.stats['hear_post_wait'] += 1
            self.h = -1
            return -1 if hs >= 0 else None
        f.prev = h
        self.stats['hear_post_attack'] += 1
        self.h = h
        return h


def make(train, cfg):
    tab = load_copy_tab() if cfg.get('copy') else None
    hm = LIB.load_hm_tab() if cfg.get('hm_mode') == 'lr' else None
    return PlanF8(r2_mix_v2.build(train), Prior(), cfg, tab, hm)
