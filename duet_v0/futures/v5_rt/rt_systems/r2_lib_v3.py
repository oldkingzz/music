"""r2_lib_v3 (run v5rt_1101, round 2): LIBRARY (no build()). r2_lib_v2.PlanR2H with a CALIBRATED harmonic memory
(hm_mode = 'lr'; any other hm_mode keeps r2_lib_v2's log-linear rule):

On train, repeated passages keep their chord only part of the time (bach re-harmonises its repeats: about 57-70 %;
pop 71-82 %; folk 84-91 %; pop909 85-97 %). runs_mini/v5rt_1101/r2/train_hm_v1.py counted (TRAIN only, dataset
balanced) r = P(chord of the current beat == chord of the aligned earlier beat) by match length and whether the shift
is a whole number of bars -> models/R2_hm_v1.npz. The earlier beat's smoothed HMM posterior gamma then enters as the
likelihood ratio of 'same chord with probability r, else a chord from m1's stationary mix pi':
  b_k  ~  b_k * ((r * gamma_k' + (1 - r) * pi) / pi) ^ hm_rho.
Val only, tuned on val (no test split).
"""
import numpy as np

from systems import r2_mix_v2

from rt_systems import r2_lib_v2 as R2V2
from rt_systems.r2_lib_v1 import Prior, load_copy_tab

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HM_TAB = ABS + '/runs_mini/v5rt_1101/models/R2_hm_v1.npz'


class PlanR2C(R2V2.PlanR2H):
    def __init__(self, mix, prior, cfg, copy_tab=None, hm_tab=None):
        super().__init__(mix, prior, cfg, copy_tab)
        self.hm_r = hm_tab
        self.pi = np.asarray(mix.m.stat, dtype=np.float64)

    def start(self, meta):
        super().start(meta)
        self.hm_cell = None

    def _align(self):
        cp = self.copy
        self.shift = self.hm_cell = None
        r = cp.r1
        if len(r) == 0:
            return
        best = int(r.max())
        if best < self.c.get('hm_min', 4):
            return
        a = cp.n - 1
        b = int(np.nonzero(r == best)[0][-1])
        d = cp.on[a] - cp.on[b]
        s = int(round(d))
        if s < 1 or abs(d - s) > 0.02:
            return
        bar = int(s % self.M == 0)
        if self.c.get('hm_bar', 0) and not bar:
            return
        self.shift = s
        self.hm_cell = (0 if best < 8 else 1 if best < 12 else 2, bar)

    def _aug(self, b, k):
        if self.c.get('hm_mode') != 'lr':
            return super()._aug(b, k)
        if not self._hm_on() or self.shift is None or self.hm_r is None:
            return b
        g = self._gamma(k - self.shift)
        if g is None:
            return b
        r = float(self.hm_r[self.hm_cell])
        lr = (r * g + (1 - r) * self.pi) / self.pi
        x = np.log(np.maximum(b, 1e-300)) + self.c['hm_rho'] * np.log(lr)
        x = np.exp(x - x.max())
        return x / x.sum()


def load_hm_tab(path=HM_TAB):
    return np.load(path)['r'].astype(np.float64)


def make(train, cfg):
    tab = load_copy_tab() if cfg.get('copy') else None
    hm = load_hm_tab() if cfg.get('hm_mode') == 'lr' else None
    return PlanR2C(r2_mix_v2.build(train), Prior(), cfg, tab, hm)
