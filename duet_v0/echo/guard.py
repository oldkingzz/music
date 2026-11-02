"""Echo guard: keep the machine's own voice (speaker -> mic bleed) out of the follower.

Pre-processing stage between the microphone and follow.FeatureTracker. Every 10 ms it gets
the mic frame AND the exact frame the host sent to the speaker (the reference).

    g = EchoGuard(aec=True, mask=True)
    g.calibrate(mic_rec, ref_rec)      # optional 1-2 s sound check (soft noise from the speaker)
    ...
    clean = g.cancel(mic240, ref240)   # (a) adaptive echo canceller
    ft.step(clean)
    g.mask(ft, expected_midis)         # (b) reference-aware pitch masking on ft's newest frame

(a) EchoCanceller: partitioned-block frequency-domain NLMS ("multidelay filter", MDF) with
    10 ms blocks, overlap-save, gradient constraint. Its per-bin step size follows Valin's
    double-talk-robust rule mu(k) = min(mu_max, eta * |Y(k)|^2 / |E(k)|^2), where eta is the
    online estimate of how much echo leaks through the filter. In a duet both voices sound at
    once almost all the time (permanent double talk), so a classic freeze-on-double-talk
    detector would never adapt; the Valin rule keeps adapting in proportion to the estimated
    echo-to-error ratio of each bin instead.
(b) PitchMask: the machine's pitch is measured with YIN on the reference itself (exact, no
    interference), aligned with the echo-path delay. If the pitch the FeatureTracker just
    detected is the machine's (same note, its octave, or a common sub-harmonic of the two
    voices) and none of the singer's expected notes explains it, the frame is re-estimated
    with YIN lags at multiples of the machine period excluded; if no clean singer period is
    left the frame is marked unvoiced.
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dsp import FMAX, FMIN, HOP, NFFT, SR, TAU_MAX, TAU_MIN, W_INT, WIN, hz_to_midi, yin  # noqa: E402


# ---------------------------------------------------------------- (a) adaptive echo canceller
class EchoCanceller:
    """Two-path MDF (partitioned-block frequency-domain NLMS), 10 ms blocks, overlap-save.

    background filter Wb: adapts every block with a constrained, power-normalised NLMS step
    foreground filter Wf: produces the output; Wb is copied into it only when Wb has had
                          clearly less error energy over the last `win` blocks, and Wb is
                          reset to Wf if it starts to diverge (Speex/Valin two-path idea).
    So adaptation can only ever make the output better than the last good filter, which is
    what makes it safe in permanent double talk and with tonal (narrow-band) references.
    """

    def __init__(self, partitions=16, block=HOP, mu=0.3, reg=0.1, two_path=True, win=25,
                 copy_ratio=0.8, reset_ratio=4.0, step='fixed', mu_max=0.5, dtd=0.25):
        self.B, self.N, self.P = block, 2 * block, partitions
        self.K = self.N // 2 + 1
        self.Wb = np.zeros((self.P, self.K), complex)
        self.Wf = np.zeros((self.P, self.K), complex)
        self.X = np.zeros((self.P, self.K), complex)          # input spectra, newest first
        self.x_prev = np.zeros(block)
        self.Px = np.zeros(self.K)
        self.lt = 0.0                                         # long-term mean reference power
        self.mu, self.reg, self.two_path, self.step, self.mu_max = mu, reg, two_path, step, mu_max
        self.win, self.copy_ratio, self.reset_ratio = win, copy_ratio, reset_ratio
        self.dtd = dtd                                        # adapt only if |e|^2 < dtd |y_hat|^2
        self.calibrated = False
        self.adapt_blocks = 0
        self.Sf = self.Sb = 0.0
        self.nwin = 0
        self.blocks = self.copies = self.resets = 0
        self.eta, self.Rey, self.Ryy = 0.3, 0.0, 0.0

    @property
    def W(self):
        return self.Wf if self.two_path else self.Wb

    def _out(self, W):
        return np.fft.irfft(np.sum(W * self.X, axis=0), self.N)[self.B:]

    def process(self, d, x, adapt=True):
        """d: mic block (B), x: reference block (B). Returns the echo-cancelled block."""
        B = self.B
        d = np.asarray(d, float)
        x = np.asarray(x, float)
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = np.fft.rfft(np.concatenate([self.x_prev, x]))
        self.x_prev = x.copy()
        self.blocks += 1
        yb = self._out(self.Wb)
        eb = d - yb
        if self.two_path:
            yf = self._out(self.Wf)
            ef = d - yf
        else:
            yf, ef = yb, eb
        xpow = np.sum(np.abs(self.X) ** 2, axis=0)
        active = float(np.sum(xpow)) > 1e-8
        if not adapt or not active:
            return ef
        if self.calibrated and self.dtd and float(np.sum(ef * ef)) > self.dtd * float(np.sum(yf * yf)):
            return ef                     # the singer (near end) is present: do not adapt
        self.adapt_blocks += 1
        self.lt = 0.995 * self.lt + 0.005 * float(np.mean(xpow)) if self.lt else float(np.mean(xpow))
        self.Px = np.maximum(0.7 * self.Px + 0.3 * xpow, xpow)
        E = np.fft.rfft(np.concatenate([np.zeros(B), eb]))
        if self.step == 'valin':
            Yb = np.fft.rfft(np.concatenate([np.zeros(B), yb]))
            Pe, Py = np.abs(E) ** 2, np.abs(Yb) ** 2
            pe, py = Pe - Pe.mean(), Py - Py.mean()
            beta = float(np.clip(0.02 * np.sum(Py) / (np.sum(Pe) + 1e-12), 0.0, 1.0))
            self.Rey = (1 - beta) * self.Rey + beta * float(np.sum(pe * py))
            self.Ryy = (1 - beta) * self.Ryy + beta * float(np.sum(py * py))
            if self.Ryy > 1e-12 and self.blocks > 100:
                self.eta = float(np.clip(self.Rey / self.Ryy, 1e-3, 1.0))
            mu = np.minimum(self.mu_max, self.eta * Py / (Pe + 1e-12)) if self.blocks > 100 else self.mu
        else:
            mu = self.mu
        G = (mu * E / (self.Px + self.reg * self.lt + 1e-12))[None, :] * np.conj(self.X)
        g = np.fft.irfft(G, self.N, axis=1)
        g[:, B:] = 0.0                                         # gradient constraint (linear conv.)
        self.Wb += np.fft.rfft(g, axis=1)
        if self.two_path:
            self.Sf += float(np.sum(ef * ef))
            self.Sb += float(np.sum(eb * eb))
            self.nwin += 1
            if self.nwin >= self.win:
                if self.Sb < self.copy_ratio * self.Sf:
                    self.Wf = self.Wb.copy()
                    self.copies += 1
                elif self.Sb > self.reset_ratio * self.Sf:
                    self.Wb = self.Wf.copy()
                    self.resets += 1
                self.Sf = self.Sb = 0.0
                self.nwin = 0
        return ef

    def finish_calibration(self):
        """After a sound check: use what the background learned; from now on adapt only when the
        mic is explained by the echo estimate (double-talk gate)."""
        self.Wf = self.Wb.copy()
        self.calibrated = True
        self.Sf = self.Sb = 0.0
        self.nwin = 0

    def impulse_response(self):
        """Time-domain estimate of the echo path (P*B taps) from the output filter."""
        return np.concatenate(np.fft.irfft(self.W, self.N, axis=1)[:, :self.B])

    def delay_samples(self):
        h = self.impulse_response()
        return int(np.argmax(np.abs(h))) if np.any(h) else None


# ---------------------------------------------------------------- (b) reference-aware pitch masking
def yin_curve(frame):
    """Cumulative-mean-normalised YIN difference dn[tau] (same maths as dsp.yin)."""
    x = np.asarray(frame, dtype=np.float64)
    a = x[:W_INT]
    r = np.fft.irfft(np.conj(np.fft.rfft(a, NFFT)) * np.fft.rfft(x, NFFT), NFFT)[:TAU_MAX + 1]
    cs = np.concatenate(([0.0], np.cumsum(x * x)))
    taus = np.arange(TAU_MAX + 1)
    d = cs[W_INT] + (cs[taus + W_INT] - cs[taus]) - 2.0 * r
    d[0] = 0.0
    dn = np.ones_like(d)
    dn[1:] = d[1:] * taus[1:] / np.maximum(np.cumsum(d[1:]), 1e-12)
    return dn


def pick_period(dn, allowed, thresh=0.15):
    """YIN's choice (first dip under thresh, else global min) among allowed lags. -> (tau, dn)"""
    taus = np.arange(TAU_MIN, TAU_MAX + 1)
    ok = allowed[TAU_MIN:]
    seg = np.where(ok, dn[TAU_MIN:], np.inf)
    if not np.isfinite(seg).any():
        return None, 1.0
    below = np.nonzero(seg < thresh)[0]
    if below.size:
        k = int(below[0])
        while k + 1 < seg.size and seg[k + 1] < seg[k]:
            k += 1
    else:
        k = int(np.argmin(seg))
    tau = int(taus[k])
    tauf = float(tau)
    if 1 <= tau < TAU_MAX and ok[k] and np.isfinite(seg[k]):
        s0, s1, s2 = dn[tau - 1], dn[tau], dn[tau + 1]
        den = s0 - 2.0 * s1 + s2
        if abs(den) > 1e-12:
            tauf = tau + float(np.clip(0.5 * (s0 - s2) / den, -1.0, 1.0))
    return tauf, float(dn[tau])


def comb_remove(x, tau):
    """z[n] = x[n] - x[n - tau] with a fractional delay (linear interpolation): zeros at
    every multiple of SR / tau, i.e. on all harmonics of a voice with period tau."""
    x = np.asarray(x, float)
    k = int(np.floor(tau))
    f = tau - k
    d = np.zeros_like(x)
    d[k + 1:] = (1 - f) * x[1:len(x) - k] + f * x[:len(x) - k - 1]
    return x - d


def _pc_dist(a, b):
    """Distance in semitones allowing octave errors (0 for a == b +- 12k)."""
    d = abs(a - b) % 12.0
    return min(d, 12.0 - d)


class PitchMask:
    def __init__(self, delay_samples=528, tol_machine=0.7, tol_singer=0.6, rel_tol=0.03, ap_max=0.25,
                 reest='comb', hold_frames=20):
        self.reest = reest
        self.mic = np.zeros(2 * WIN)
        self.hold_frames = hold_frames
        self.last_midi, self.last_seen = np.nan, -10 ** 9
        self.delay = delay_samples
        self.tol_m, self.tol_s, self.rel_tol, self.ap_max = tol_machine, tol_singer, rel_tol, ap_max
        self.ref = np.zeros(WIN + 4800)
        self.n_masked = self.n_reest = self.n_unvoiced = self.n_frames = self.n_active = self.n_recovered = 0
        self.machine_midi = np.nan

    def push_ref(self, x):
        self.ref = np.concatenate([self.ref[len(x):], x])

    def push_mic(self, x):
        """The (echo-cancelled) audio that the FeatureTracker sees."""
        self.mic = np.concatenate([self.mic[len(x):], x])

    def machine_pitch(self):
        """Pitch of the machine voice that is reaching the mic in the current analysis window."""
        d = int(np.clip(self.delay, 0, len(self.ref) - WIN))
        seg = self.ref[len(self.ref) - WIN - d:len(self.ref) - d]
        if float(np.sqrt(np.mean(seg * seg))) < 1e-4:
            return np.nan
        f0, ap, _ = yin(seg)
        return float(hz_to_midi(f0)) if ap < 0.2 and FMIN <= f0 <= FMAX else np.nan

    def apply(self, ft, expected):
        """Edit ft's newest frame in place if its pitch is the machine's. expected: singer MIDI notes
        that could plausibly be sounding now (previous, current, next note of the score)."""
        self.n_frames += 1
        mm = self.machine_pitch()
        if np.isnan(mm):
            # the room keeps ringing after the machine stops (RT60): hold its last pitch a while
            if self.n_frames - self.last_seen <= self.hold_frames:
                mm = self.last_midi
        else:
            self.last_midi, self.last_seen = mm, self.n_frames
        self.machine_midi = mm
        if np.isnan(mm):
            return
        self.n_active += 1
        tau_m = SR / float(440.0 * 2 ** ((mm - 69) / 12))
        if not ft.v[-1]:
            # loud but aperiodic: often her voice plus the machine (e.g. mid-glide). If removing
            # the machine leaves a clean period, she is singing: recover the frame.
            if self.reest == 'comb' and ft.db[-1] > max(ft.peak_db - 30, -60):
                new = self._comb_pitch(tau_m, mm)
                if not np.isnan(new):
                    self.n_recovered += 1
                    self._set(ft, new)
            return
        p = ft.raw[-1]
        # The harmony line is always below her, so a pitch at (or below) the machine's note is
        # either the machine itself or a sub-harmonic of the two-voice mixture (YIN finds the
        # common period of e.g. a major third, 5:4, two octaves down).
        machine_like = _pc_dist(p, mm) <= self.tol_m or p < mm + self.tol_m
        if not machine_like:
            return
        if any(abs(p - e) <= self.tol_s for e in expected):
            return                                   # she may really be singing this note
        self.n_masked += 1
        if self.reest == 'comb':
            # remove the machine from the analysis window with a comb at its exact period
            # (nulls at every machine harmonic), then run YIN again. Comb filtering does not
            # change the singer's periodicity, so her pitch survives without bias.
            new = self._comb_pitch(tau_m, mm)
        else:
            # re-estimate with lags at/below the machine period and its harmonics excluded
            dn = yin_curve(ft.buf)
            taus = np.arange(TAU_MAX + 1, dtype=float)
            allowed = taus <= tau_m * (1 - self.rel_tol)     # the singer is above the harmony line
            for j in (2, 3):                                  # machine harmonics (octave, 12th above)
                allowed &= np.abs(taus - tau_m / j) > self.rel_tol * tau_m / j + 1.0
            tau, ap = pick_period(dn, allowed)
            if tau is not None and ap < self.ap_max:
                f0 = SR / tau
                new = float(hz_to_midi(f0)) if FMIN <= f0 <= FMAX else np.nan
            else:
                new = np.nan
        if np.isnan(new):
            self.n_unvoiced += 1
            ft.raw[-1] = np.nan
            ft.m[-1] = np.nan
            ft.v[-1] = False
        else:
            self.n_reest += 1
            self._set(ft, new)
        ft.nov[-1] = ft._novelty()

    def _comb_pitch(self, tau_m, mm):
        seg = self.mic[len(self.mic) - WIN - int(np.ceil(tau_m)) - 2:]
        z = comb_remove(seg, tau_m)[-WIN:]
        f0, ap, _ = yin(z)
        if ap < self.ap_max and FMIN <= f0 <= FMAX:
            m = float(hz_to_midi(f0))
            if _pc_dist(m, mm) > self.tol_m:
                return m
        return np.nan

    @staticmethod
    def _set(ft, new):
        """Write a voiced pitch into ft's newest frame, with FeatureTracker's own smoothing."""
        ft.raw[-1] = new
        ft.v[-1] = True
        m = new
        if len(ft.v) >= 3 and ft.v[-2] and ft.v[-3]:
            m = float(np.median(ft.raw[-3:]))
        ft.m[-1] = m
        ft.nov[-1] = ft._novelty()


# ---------------------------------------------------------------- the stage the live app calls
class EchoGuard:
    """The stage the live app calls. aec / mask switch the two mitigations on or off."""

    def __init__(self, aec=True, mask=True, partitions=16, delay_samples=528,
                 sc_mu=0.8, sc_reg=0.01, mu=0.3, reg=1.0, dtd=0.25, reest='comb'):
        self.partitions, self.mu, self.reg, self.dtd = partitions, mu, reg, dtd
        self.sc_mu, self.sc_reg = sc_mu, sc_reg
        self.aec = EchoCanceller(partitions=partitions, mu=mu, reg=reg, dtd=dtd) if aec else None
        self.pm = PitchMask(delay_samples=delay_samples, reest=reest) if mask else None
        self.n = 0
        self.soundcheck_erle_db = None

    def calibrate(self, mic, ref):
        """Sound check: the speaker plays ~2 s of soft broadband noise while nobody sings.
        A tonal machine voice only excites a few frequencies, so an echo canceller that starts
        cold during the song learns the path one note at a time (and must do it in double talk);
        the noise teaches it every frequency at once. mic, ref: the recorded pair."""
        n = min(len(mic), len(ref)) // HOP * HOP
        if self.aec is not None:
            c = EchoCanceller(partitions=self.partitions, mu=self.sc_mu, reg=self.sc_reg, two_path=False,
                              dtd=self.dtd)
            out = np.zeros(n)
            for k in range(0, n, HOP):
                out[k:k + HOP] = c.process(mic[k:k + HOP], ref[k:k + HOP])
            c.mu, c.reg, c.two_path = self.mu, self.reg, True
            c.finish_calibration()
            self.aec = c
            q = n // 4                                            # ERLE over the last quarter
            self.soundcheck_erle_db = float(10 * np.log10(np.sum(mic[n - q:n] ** 2) /
                                                          (np.sum(out[n - q:] ** 2) + 1e-20)))
        if self.pm is not None:
            lag = estimate_delay(mic[:n], ref[:n])
            if lag is not None:
                self.pm.delay = lag
        return self.soundcheck_erle_db

    def cancel(self, mic, ref):
        self.n += 1
        if self.pm is not None:
            self.pm.push_ref(np.asarray(ref, float))
        out = np.asarray(mic, float) if self.aec is None else self.aec.process(np.asarray(mic, float),
                                                                                np.asarray(ref, float))
        if self.pm is not None:
            self.pm.push_mic(out)
        return out

    def mask(self, ft, expected):
        if self.pm is not None:
            self.pm.apply(ft, expected)

    def stats(self):
        s = {}
        if self.aec is not None:
            s.update({'aec_delay_samples': self.aec.delay_samples(), 'aec_copies': self.aec.copies,
                      'aec_resets': self.aec.resets, 'aec_adapt_blocks': self.aec.adapt_blocks,
                      'soundcheck_erle_db': self.soundcheck_erle_db})
        if self.pm is not None:
            s.update({'mask_active_frames': self.pm.n_active, 'masked': self.pm.n_masked,
                      'reestimated': self.pm.n_reest, 'unvoiced': self.pm.n_unvoiced,
                      'recovered': self.pm.n_recovered,
                      'mask_delay_samples': self.pm.delay})
        return s


def estimate_delay(mic, ref, max_lag=4800):
    """GCC-PHAT delay of mic relative to ref (samples), for the sound check."""
    n = 1 << int(np.ceil(np.log2(len(mic) + max_lag)))
    R = np.fft.rfft(mic, n) * np.conj(np.fft.rfft(ref, n))
    cc = np.fft.irfft(R / (np.abs(R) + 1e-12), n)[:max_lag]
    return int(np.argmax(cc)) if np.any(cc) else None
