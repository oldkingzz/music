"""r2_style_hmm_v1 (v4 round 2, idea E3): m1_filter_v5lo's chooser fed by a style-aware chord HMM, three copies of
m1's HMM (pooled, pop-only, pop909-only) mixed by a running style probability from the meter and each beat's rhythm."""
import numpy as np

from systems import m1_filter_v5lo as m1

# one row per style: (train corpora of its HMM, None = m1's own pooled HMM), (train corpora of its meter and
# rhythm model), lam = weight of her pitch evidence in that HMM
STYLES = ((None, ('bach', 'folk'), 1.0),
          (('pop',), ('pop',), 1.0),
          (('pop909',), ('pop909',), 1.0))
KAPPA, NOISE = 0.5, 0.1      # m1's transition smoothing and 10 % mishearing, for the style-only HMMs
NW = 16                      # rhythm words: notes starting in the beat (0/1/2/3+) x one exactly on it x rest > 1/4
S = m1.S


def rword(notes, k, end=None):
    """Rhythm word of beat k from notes [onset, offset, x] sorted by onset (the first `end` of them)."""
    n_on, at_k, cover = 0, 0, 0.0
    for j in range((len(notes) if end is None else end) - 1, -1, -1):
        on, off = notes[j][0], notes[j][1]
        if off is not None and off <= k:
            break
        if off is None or on >= k + 1:
            continue
        cover += min(off, k + 1) - max(on, k)
        if on >= k - 1e-9:
            n_on += 1
            at_k |= int(abs(on - k) < 1e-6)
    return min(n_on, 3) * 4 + at_k * 2 + int(1 - cover > 0.25)


class SubHMM:
    """m1.Model's counting on some train corpora only, each song weight 1: A per bar-position key, logB, pi."""

    def __init__(self, train, corpora):
        A = np.zeros((6, S, S))
        B = np.zeros((S, 12))
        pi = np.zeros(S)
        for name in corpora:
            for s in train[name]:
                t, M = s['tonic'], s['M']
                z = [b['z'] for b in s['beats']]
                obs = m1.beat_obs([[on, off, (m - t) % 12] for on, off, m in s['sop']], len(z))
                first = True
                for k in range(len(z)):
                    if z[k] < 0:
                        continue
                    B[z[k]] += obs[k]
                    if first:
                        pi[z[k]] += 1.0
                        first = False
                    if k >= 1 and z[k - 1] >= 0:
                        A[m1.tkey(s['beats'][k]['pos'], M), z[k - 1], z[k]] += 1.0
        self.A = (A + KAPPA) / (A + KAPPA).sum(-1, keepdims=True)
        Bc = (B + 1.0) / (B + 1.0).sum(-1, keepdims=True)
        self.logB = np.log((1 - NOISE) * Bc + NOISE / 12)
        self.pi = (pi + 1) / (pi + 1).sum()


def style_priors(train):
    """Per style, counted on its train songs (add-one): log P(meter) and log P(rhythm word of one beat)."""
    metas = sorted({s['M'] for v in train.values() for s in v})
    lw, lm = [], []
    for _, group, _ in STYLES:
        c = np.ones(NW)
        mc = {M: 1.0 for M in metas}
        for ds in group:
            for s in train[ds]:
                notes = sorted([list(n) for n in s['sop']], key=lambda n: n[0])
                mc[s['M']] += 1
                end = 0
                for k in range(len(s['beats'])):
                    while end < len(notes) and notes[end][0] < k + 1:
                        end += 1
                    c[rword(notes, k, end)] += 1
        lw.append(np.log(c / c.sum()))
        tot = sum(mc.values())
        lm.append({M: np.log(v / tot) for M, v in mc.items()})
    return np.array(lw), {M: np.array([x[M] for x in lm]) for M in metas}


class StyleFilter(m1.Filter):
    """m1's chooser (step) unchanged; belief(k) = style-probability-weighted mix of the styles' HMM forward filters."""

    def __init__(self, models, lams, rh, mprior):
        self.m, self.H = models[0], -1           # m1.Filter's two fields: pooled model, no history window
        self.models, self.lams, self.rh, self.mprior = models, lams, rh, mprior

    def start(self, meta):
        super().start(meta)
        n = len(self.models)
        self.alphas = [None] * n
        self.logw = np.full(n, -np.log(n))
        if meta['M'] in self.mprior:
            self.logw = self.logw + self.mprior[meta['M']]

    def _beat_v(self, k):
        """How long each relative pc sounded in beat k; her current note is taken to last to the beat's end (as m1)."""
        v = np.zeros(12)
        for on, off, pc in reversed(self.notes):
            e = off if off is not None else k + 1
            if e <= k:
                break
            ov = min(e, k + 1) - max(on, k)
            if ov > 0:
                v[pc] += ov
        return v

    def _step(self, j, k, v):
        mo = self.models[j]
        ll = self.lams[j] * (mo.logB @ v) if v.sum() > 0 else np.zeros(S)
        a = self.alphas[j]
        pr = mo.pi.copy() if a is None else a @ mo.A[m1.tkey(self._pos(k), self.M)]
        w = pr * np.exp(ll - ll.max())
        return w / w.sum()

    def _finish_beats(self, upto):
        while self.done < upto:
            k = self.done
            self.logw = self.logw + self.rh[:, rword(self.notes, k)]
            v = self._beat_v(k)
            self.alphas = [self._step(j, k, v) for j in range(len(self.models))]
            self.done += 1

    def belief(self, k):
        v = self._beat_v(k)
        bs = np.array([self._step(j, k, v) for j in range(len(self.models))])
        wt = np.exp(self.logw - self.logw.max())
        return (wt / wt.sum()) @ bs


def build(train):
    models = [m1.Model(train) if hmm is None else SubHMM(train, hmm) for hmm, _, _ in STYLES]
    rh, mprior = style_priors(train)
    return StyleFilter(models, [lam for _, _, lam in STYLES], rh, mprior)
