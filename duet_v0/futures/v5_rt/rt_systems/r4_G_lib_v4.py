"""r4_G_lib_v4 (run v5rt_1101, round 4, idea G 'late-hear policy'): LIBRARY (no build()). rt_systems.r4_G_lib_v3.PlanG3
(read r4_G_lib_v1 .. v3) plus one option for the LARGE lag regime only (off by default; then the decisions are
r4_G_lib_v3's exactly):

  g_ehedge  idea E's guarded choice (the hear path when the tracker posterior doubts the heard pitch, P(misheard) >=
            tau_t; about 38 % of W's hears at bt_cal) scores keep and every attack candidate against her CURRENT note
            only (its hedged reactive candidate aside). In the large regime each note candidate's score gets
            + g_ehedge * w * U_next(h) (w = the lag-aware hedge weight, U_next = the next-note utility with the large
            regime's harsh weight; silence's next-note value is 0, as its current-note value is sil_v), because the
            answer sounds mostly on the rest of her note and the start of her next one. The key-aware doubt rule of
            r3_X_lib_v3 is applied unchanged around it (doubt_mode 'noatt': no attack candidate passes).
            Everything else in E's choice is r3_E_lib_v2.PlanE._hear_trk's code, copied.
Val only, tuned on val (no test split). make(train, cfg, reg=None) as r4_G_lib_v1.make, with this class.
"""
import numpy as np

from systems import m1_filter_v5lo as M1
from systems import r2_mix_v2
from theory import ALTO, ALTO_MID, MAJOR

from rt_systems import r2_lib_v3 as R3
from rt_systems import r3_D_lib_v3 as DL
from rt_systems import r3_E_lib_v2 as EL
from rt_systems import r4_G_lib_v3 as G3
from rt_systems.r2_lib_v1 import GOODV, HARSHV, HV, Prior, load_copy_tab

XL = G3.G2.G1.XL
EV, E0, A0 = EL.EV, EL.E0, EL.A0


class PlanG4(G3.PlanG3):
    def _hear_trk(self, i, onset, midi, prev_dur, post):
        c = self.c
        if c.get('g_ehedge', 0) <= 0 or self._g_lag() is None:
            return XL.PlanEF3._hear_trk(self, i, onset, midi, prev_dur, post)
        pm = 1.0 - float(post[E0])
        fire = pm >= c.get('doubt_hold', 2.0) or ((midi - self.t) % 12 not in MAJOR and pm >= c.get('doubt_key', 2.0))
        if not fire:
            return self._trk_g(i, onset, midi, prev_dur, post)
        if c.get('doubt_mode', 'hold_sil') != 'noatt':
            return XL.PlanEF3._hear_trk(self, i, onset, midi, prev_dur, post)
        hm0 = c['harsh_max']
        c['harsh_max'] = -1.0                              # no attack candidate passes E's harsh bound
        try:
            r = self._trk_g(i, onset, midi, prev_dur, post)
        finally:
            c['harsh_max'] = hm0
        self.stats['doubt_noatt'] += 1
        return r

    def _trk_g(self, i, onset, midi, prev_dur, post):
        """r3_E_lib_v2.PlanE._hear_trk with + g_ehedge * w * U_next on every note candidate's score."""
        c, f = self.c, self.f
        prev, prev_m = f.prev, f.prev_m
        h_r = int(f.step(i, onset, midi, prev_dur))       # r2_mix_v2's step: GRU state and filter, exactly once
        self.her.append((onset, midi))
        self.planned = False
        hs = self.h
        k = int(onset)
        pfit = self._belief(k) @ M1.CTM
        mt = midi + EV
        ivm = (mt[:, None] - HV[None, :]) % 12
        pg = post @ GOODV[ivm]
        ph = post @ HARSHV[ivm]
        px = post @ (mt[:, None] <= HV[None, :]).astype(np.float64)
        P = M1.P
        d = np.abs(HV - prev) if prev is not None else np.zeros(len(HV))
        s = (pfit[(HV - self.t) % 12] * pg - c['lam_t'] * ph - c['w_x'] * px
             - P['w_step'] * d - P['w_big'] * (d > 7) - P['w_mid'] * np.abs(HV - ALTO_MID))
        lo = int(min(mt[(post >= c['p_min']) | (post >= post.max())].min(), midi))
        h_c = h_r
        w = self._w_next(onset)
        un = None
        if (w > 0 or (self._hm_on() and self.shift is not None)) and h_r >= 0:
            v = self._m1vec(midi, pfit, prev, prev_m)
            if w > 0:
                un = self._u_next(onset, midi, k)
                v = v + w * un
            j = int(np.argmax(v))
            if np.isfinite(v[j]):
                h_c = int(HV[j])
        if w > 0:
            if un is None:
                un = self._u_next(onset, midi, k)
            s = s + c['g_ehedge'] * w * un
            self.stats['g_ehedge'] += 1
        cand = {}
        keep_ok = hs >= 0 and ALTO[0] <= hs <= ALTO[1]
        if keep_ok:
            cand[hs] = float(s[hs - A0])

        def att_ok(x):
            return ph[x - A0] <= c['harsh_max'] and (c['above_att'] or x < lo)

        for x in (h_r, h_c):
            if x >= 0 and x != hs and att_ok(x):
                cand[x] = float(s[x - A0] - c['kappa_t'] * ph[x - A0])
        if c['free']:
            for j in range(len(HV)):
                x = int(HV[j])
                if x != hs and x not in cand and att_ok(x):
                    cand[x] = float(s[j] - c['kappa_t'] * ph[j])
        best = max(cand, key=cand.get) if cand else None
        if best is None or cand[best] < c['sil_v']:
            self.stats['trk_sil'] += 1
            self.h = -1
            return -1
        f.prev = best
        if best == hs:
            self.stats['trk_keep'] += 1
            return None
        self.stats['trk_react' if best in (h_r, h_c) else 'trk_attack'] += 1
        self.h = best
        return best


def make_base(mix, prior, cfg, copy_tab, hm_tab, trk_tab):
    return PlanG4(mix, prior, cfg, copy_tab if cfg.get('copy') else None,
                  hm_tab if cfg.get('hm_mode') == 'lr' else None, trk_tab if cfg.get('trk') else None)


def make(train, cfg, reg=None):
    base = make_base(r2_mix_v2.build(train), Prior(), cfg, load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab())
    return DL.RegAdapter(base, reg) if reg else base
