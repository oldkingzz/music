"""The chord hidden Markov model (spec section 6, E7 to E11, E16, E19).

Hidden state z_k: the diatonic triad behind beat k (7 values). Observation y_k: what she sang in
that beat, either the pitch class she held longest (mode 'long', Q3 first choice) or the 12-vector
of how long she held each pitch class (mode 'vec', E6). Transitions depend on where the NEXT beat
falls in the bar (E7): four matrices for 4/4, and a downbeat / other-beat pair for any other meter.
All counts come from the training chorales with Laplace smoothing (E16).
"""
import numpy as np

from theory import N_CHORDS

S = N_CHORDS


def soprano_obs(sop, k, end_frac=1.0):
    """Her melody in beat k as (vec, longest_pc_rel_midi, longest_midi) from notes in beat units.

    sop: [[on, off, midi]] in beats; the window is [k, k + end_frac). end_frac < 1 cuts the end of
    the beat that has not been heard yet at decision time (causality, E5). Pitch classes are NOT
    made relative here; callers subtract the tonic.
    """
    a, b = k, k + end_frac
    vec = np.zeros(12)
    best, best_len = None, 0.0
    for on, off, m in sop:
        if on >= b:
            break
        ov = min(off, b) - max(on, a)
        if ov > 1e-9:
            vec[m % 12] += ov
            if ov > best_len + 1e-9:
                best, best_len = m, ov
    return vec, best


class ChordHMM:
    def __init__(self, kappa=1.0):
        self.kappa = kappa

    # ------------------------------------------------------------------ training (E16)
    def fit(self, chorales):
        k = self.kappa
        init = np.zeros(S)
        A4 = np.zeros((4, S, S))           # 4/4: into beat position 0..3
        A2 = np.zeros((2, S, S))           # any meter: into a downbeat (0) or any other beat (1)
        em = np.zeros((S, 12))
        for r in chorales:
            t, beats, M = r['tonic'], r['beats'], r['M']
            zs = [b['z'] for b in beats]
            first = next((z for z in zs if z >= 0), None)
            if first is not None:
                init[first] += 1
            for i in range(1, len(beats)):
                a, b = zs[i - 1], zs[i]
                if a < 0 or b < 0:
                    continue
                pos = beats[i]['pos']
                if M == 4:
                    A4[pos, a, b] += 1
                A2[0 if pos == 0 else 1, a, b] += 1
            for i, bt in enumerate(beats):
                if bt['z'] < 0:
                    continue
                vec, _ = soprano_obs(r['sop'], i)
                em[bt['z']] += np.roll(vec, -t)          # make pitch classes relative to the tonic
        self.counts = {'init': init, 'A4': A4, 'A2': A2, 'em': em}
        self.init = (init + k) / (init.sum() + S * k)
        self.A4 = (A4 + k) / (A4.sum(-1, keepdims=True) + S * k)
        self.A2 = (A2 + k) / (A2.sum(-1, keepdims=True) + S * k)
        self.B = (em + k) / (em.sum(-1, keepdims=True) + 12 * k)
        self.logB = np.log(self.B)
        return self

    @staticmethod
    def trans_key(pos, M):
        """Which transition matrix leads into a beat at bar position `pos` (E7)."""
        return ('A4', pos % 4) if M == 4 else ('A2', 0 if pos % M == 0 else 1)

    def mat(self, key):
        return self.A4[key[1]] if key[0] == 'A4' else self.A2[key[1]]

    def trans(self, pos, M):
        return self.mat(self.trans_key(pos, M))

    # ------------------------------------------------------------------ inference (E8, E10, E11, E19)
    def loglik(self, obs, mode, lam=1.0):
        """log P(y | z) for all z. obs is a relative pitch class (mode 'long') or a 12-vector of
        fractions of the beat (mode 'vec'); None or an all-zero vector (a rest) gives no evidence."""
        if obs is None:
            return np.zeros(S)
        if mode == 'long':
            return self.logB[:, int(obs) % 12]
        v = np.asarray(obs, float)
        if v.sum() <= 0:
            return np.zeros(S)
        return lam * (self.logB @ v)

    @staticmethod
    def update(prior, ll):
        """Forward-filter update (E10): posterior over chords after one beat's observation."""
        w = prior * np.exp(ll - ll.max())
        return w / w.sum()

    def predict(self, post, pos, M):
        """One-step prediction (E11): chord distribution for the beat at position `pos`."""
        return post @ self.trans(pos, M)
