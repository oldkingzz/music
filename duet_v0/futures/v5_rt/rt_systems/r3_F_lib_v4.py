"""r3_F_lib_v4 (run v5rt_1101, round 3, idea F 'stumble guard'): LIBRARY (no build(): not a system itself). Val only,
tuned on val (no test split). rt_systems.r3_F_lib_v3.PlanF3 (read r3_F_lib_v1, v2, v3) with a cautious posterior hear,
each off by default (then the decisions are r3_F_lib_v3's exactly):
  post_veto     when the posterior hear doubts the heard pitch (P(misheard) >= hear_tau) and the sounding note is not
                harsh against the heard pitch, Duet keeps it: no new attack on a note it is unsure of (bench3's random
                mishearings then cause no fresh clash; on the live tracker, where the doubt is usually wrong, Duet does
                not jump to a guess either).
  post_nh_keep  the posterior hear may not keep a note that is harsh against the heard pitch (it must move or, if no
                note is allowed, fall silent).
"""
import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID

from rt_systems import r2_lib_v3 as LIB
from rt_systems import r3_F_lib_v3 as F3
from rt_systems.r2_lib_v1 import E_ALL, GOODV, HARSHV, HV, Prior, load_copy_tab


class PlanF4(F3.PlanF3):
    def _hear_post(self, i, onset, midi, prev_dur, post):
        """r3_F_lib_v2.PlanF2._hear_post plus post_veto and post_nh_keep."""
        c, f = self.c, self.f
        hs = self.h
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        hs_harsh = keep_ok and HARSHV[(midi - hs) % 12] > 0
        if c.get('post_veto', 0) and keep_ok and not hs_harsh:
            f.step(i, onset, midi, prev_dur)              # r2_mix_v2's step: GRU state and filter, exactly once
            self.her.append((onset, midi))
            self.planned = False
            f.prev = hs
            self.stats['hear_post_veto'] += 1
            return None
        prev = f.prev
        f.step(i, onset, midi, prev_dur)
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
        if c.get('post_nh', 0):
            att = np.where(HARSHV[(midi - HV) % 12] > 0, -np.inf, att)
        if c.get('post_below', 0):
            att = np.where(HV >= midi, -np.inf, att)
        if keep_ok and c.get('st_post', 0) > 0 and self._recent(onset):
            att = att - c['st_post']
        if keep_ok and not (c.get('post_nh_keep', 0) and hs_harsh):
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
        f.prev = h
        if keep_ok and h == hs:
            self.stats['hear_post_keep'] += 1
            return None
        self.stats['hear_post_attack'] += 1
        self.h = h
        return h


def make(train, cfg):
    tab = load_copy_tab() if cfg.get('copy') else None
    hm = LIB.load_hm_tab() if cfg.get('hm_mode') == 'lr' else None
    return PlanF4(r2_mix_v2.build(train), Prior(), cfg, tab, hm)
