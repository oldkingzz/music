"""D5 live harmonizer: block-based real-time engine (numpy only).

    fx = Harmonizer(tonic=0)            # key chosen by the user before singing (major-scale tonic pc)
    fx.calibrate(mic_rec, ref_rec)      # optional sound check: speaker plays soft noise, nobody sings
    for mic_block in stream:            # one fixed block (default 64 samples at 24 kHz = 2.67 ms)
        out_block = fx.process(mic_block)   # harmony only; the engine remembers its own output (AEC ref)

Pieces, in signal order:
  1. EchoCanceller (echo/guard.py MDF NLMS, block size = engine block) removes the speaker echo,
     reference = the engine's own output (known exactly).
  2. Streaming YIN on the last ~2 periods of the lowest f0 (female mode 140-1000 Hz: 342 samples).
  3. Note decision with hysteresis -> key-aware diatonic third below -> shift interval s in {3, 4}
     semitones. The shifter applies ratio 2^(-s/12) to HER instantaneous f0, so the harmony keeps
     her vibrato and intonation (a stage harmonizer).
  4. Streaming TD-PSOLA: synthesis marks every T1 = T0 / ratio; each grain is a Hann window of
     half-width T1 cut from her signal at an analysis position ta that is an integer number of her
     periods after the previous one and never later than ts - delta. Grains are rendered
     incrementally (a grain only needs input up to its current output sample), so the audio path
     delay is ts - ta in [delta, delta + T0): about half a period. Grains are not resampled, so the
     spectral envelope (formants) stays hers.
  5. Voice-activity gate (attack 1 block, hold 1 block, release -40 dB in 60 ms) with an echo
     guard: it opens only if the cleaned mic is voiced, above the noise floor, clearly louder than
     the canceller's own echo estimate, and (when quiet) not at the pitch Duet just played.
  6. Output limiter (instant attack, 50 ms release, ceiling -3 dBFS) + hard clip safety.
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echo.guard import EchoCanceller  # noqa: E402  (imported unmodified)

MAJOR = (0, 2, 4, 5, 7, 9, 11)


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.asarray(f, float) / 440.0)


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def third_below(note, tonic):
    """Diatonic third below `note` (int MIDI) in the major scale on `tonic` (pc).
    Chromatic notes: the scale tone 3 or 4 semitones below (prefer 4, then 3)."""
    scale = {(tonic + s) % 12 for s in MAJOR}
    note = int(note)
    if note % 12 in scale:
        below = [m for m in range(note - 1, note - 6, -1) if m % 12 in scale]
        return below[1]
    for d in (4, 3):
        if (note - d) % 12 in scale:
            return note - d
    return note - 3


# ------------------------------------------------------------------------------------- YIN
class StreamYin:
    """YIN on the newest W + tau_max samples (integration window W = tau_max: ~2 lowest periods)."""

    def __init__(self, sr, fmin, fmax, thresh=0.15):
        self.sr = sr
        self.tau_max = int(np.ceil(sr / fmin))
        self.tau_min = max(2, int(sr / fmax))
        self.W = self.tau_max
        self.N = self.W + self.tau_max
        self.nfft = 1 << int(np.ceil(np.log2(self.N + self.W)))
        self.thresh = thresh
        self.taus = np.arange(self.tau_max + 1)

    def __call__(self, x):
        """x: the newest N samples. -> (f0 Hz or nan, aperiodicity 0..1)."""
        W, tm = self.W, self.tau_max
        a = x[:W]
        r = np.fft.irfft(np.conj(np.fft.rfft(a, self.nfft)) * np.fft.rfft(x, self.nfft), self.nfft)[:tm + 1]
        cs = np.concatenate(([0.0], np.cumsum(x * x)))
        d = cs[W] + (cs[self.taus + W] - cs[self.taus]) - 2.0 * r
        d[0] = 0.0
        dn = np.ones_like(d)
        dn[1:] = d[1:] * self.taus[1:] / np.maximum(np.cumsum(d[1:]), 1e-12)
        seg = dn[self.tau_min:]
        below = np.nonzero(seg < self.thresh)[0]
        if below.size:
            k = int(below[0])
            while k + 1 < seg.size and seg[k + 1] < seg[k]:
                k += 1
        else:
            k = int(np.argmin(seg))
        tau = k + self.tau_min
        tauf = float(tau)
        if 1 <= tau < tm:
            s0, s1, s2 = dn[tau - 1], dn[tau], dn[tau + 1]
            den = s0 - 2.0 * s1 + s2
            if abs(den) > 1e-12:
                tauf = tau + float(np.clip(0.5 * (s0 - s2) / den, -1.0, 1.0))
        ap = float(np.clip(dn[tau], 0.0, 1.0))
        return self.sr / tauf, ap


# ------------------------------------------------------------------------------------- TD-PSOLA
class StreamPsola:
    """Streaming TD-PSOLA with incremental grain rendering (see module docstring, step 4)."""

    def __init__(self, sr, block, tmax, delta=2.0):
        self.B = block
        self.L = int(8 * tmax + 4 * block)               # input history kept
        self.hist = np.zeros(self.L)
        self.n = 0                                        # absolute index of the next input sample
        self.grains = []                                  # [ts, ta, Tw]
        self.next_ts = None
        self.last_ta = None
        self.delta = delta
        self.win_idx = np.arange(block, dtype=float)

    def process(self, x, T0, ratio):
        B = self.B
        n0 = self.n
        self.hist = np.concatenate([self.hist[B:], x])
        self.n += B
        T1 = T0 / ratio
        if self.next_ts is None or self.next_ts < n0 - T1:
            self.next_ts = n0 + T1
            self.last_ta = None
        while self.next_ts - T1 < n0 + B:
            ts = self.next_ts
            target = ts - self.delta
            if self.last_ta is None or target - self.last_ta > 3 * T0:
                ta = target
            else:
                m = np.floor((target - self.last_ta) / T0)
                ta = self.last_ta + max(m, 0.0) * T0
            self.grains.append((ts, ta, T1))
            self.last_ta = ta
            self.next_ts = ts + T1
        y = np.zeros(B)
        o = n0 + self.win_idx
        keep = []
        base = self.n - self.L                           # absolute index of hist[0]
        for ts, ta, Tw in self.grains:
            a = max(int(np.ceil(ts - Tw)), n0)
            b = min(int(np.floor(ts + Tw)) + 1, n0 + B)
            if b > a:
                oo = o[a - n0:b - n0]
                src = oo + (ta - ts) - base
                i = np.floor(src).astype(int)
                f = src - i
                np.clip(i, 0, self.L - 2, out=i)
                seg = self.hist[i] * (1 - f) + self.hist[i + 1] * f
                w = 0.5 + 0.5 * np.cos(np.pi * (oo - ts) / Tw)
                y[a - n0:b - n0] += seg * w
            if ts + Tw >= n0 + B:
                keep.append((ts, ta, Tw))
        self.grains = keep
        return y


# ------------------------------------------------------------------------------------- lean AEC
class FrozenMDF:
    """Echo canceller used during the song: the MDF filter learned in the sound check (echo/guard.py
    EchoCanceller), applied with overlap-save; no adaptation during the song (cheap and immune to the
    permanent double talk of a harmonizer, whose reference is her own voice shifted)."""

    def __init__(self, W, block):
        self.W = W.copy()
        self.P, self.K = W.shape
        self.B = block
        self.X = np.zeros_like(W)
        self.prev = np.zeros(block)

    def process(self, d, x):
        self.X[1:] = self.X[:-1]
        self.X[0] = np.fft.rfft(np.concatenate([self.prev, x]))
        self.prev = x
        y = np.fft.irfft(np.einsum('pk,pk->k', self.W, self.X), 2 * self.B)[self.B:]
        return d - y


# ------------------------------------------------------------------------------------- engine
class Harmonizer:
    def __init__(self, tonic=0, sr=24000, block=64, fmin=140.0, fmax=1000.0, aec=True, guard=True,
                 partitions=128, out_gain=1.0, ceiling_db=-3.0, release_s=0.06, hyst=0.6, hyst_blocks=2,
                 open_snr_db=15.0, echo_margin_db=6.0, mode='correct', unv_close=8, ap_open=0.25, ap_stay=0.45,
                 vib_clip=0.25, close_db=22.0, log=False):
        self.sr, self.B, self.tonic = sr, block, tonic
        self.yin = StreamYin(sr, fmin, fmax)
        self.inbuf = np.zeros(self.yin.N)
        self.psola = StreamPsola(sr, block, self.yin.tau_max)
        self.aec = FrozenMDF(np.zeros((partitions, block + 1), complex), block) if aec else None
        self.partitions = partitions
        self.guard = guard
        self.ref_prev = np.zeros(block)                  # the engine's last output = speaker reference
        self.T0 = sr / 220.0
        self.f0 = np.nan
        self.note = None                                  # her current note (int MIDI)
        self.cand, self.cand_n = None, 0
        self.shift = 3
        self.hyst, self.hyst_blocks = hyst, hyst_blocks
        self.gate = 0.0
        self.gate_open = False
        self.unv = 0
        self.L_ref = -30.0
        self.N_floor = -70.0
        self.open_snr, self.echo_margin = open_snr_db, echo_margin_db
        bd = block / sr
        self.rel_coef = 10 ** (-40.0 / 20 * bd / release_s)    # -40 dB in release_s
        self.ref_decay = 6.0 * bd                               # singing-level tracker decay, dB/block
        self.lim = 1.0
        self.ceiling = 10 ** (ceiling_db / 20)
        self.lim_rel = 2.0 ** (bd / 0.05)                       # limiter release: gain doubles per 50 ms
        self.out_gain = out_gain
        self.out_pitch_hist = deque()                     # (block idx, midi) of what Duet played
        self.hist_blocks = int(0.6 * sr / block)
        self.ramp = np.arange(1, block + 1) / block
        self.close_db = close_db
        self.mode, self.unv_close, self.ap_open, self.ap_stay, self.vib_clip = mode, unv_close, ap_open, ap_stay, vib_clip
        self.lv = deque(maxlen=int(round(0.05 * sr / block)))   # block levels over the last 50 ms
        self.tz = 0j                                      # tuning tracker (circular mean of her cents)
        self.tz_a = 1.0 - np.exp(-bd / 2.0)
        self.n_t = 0
        self.tun = 0.0
        self.ratio = 1.0
        self.k = 0
        self.log = [] if log else None
        self.dt_blocks = []

    # ---------------------------------------------------------------- sound check
    def calibrate(self, mic, ref):
        """Speaker played `ref` (soft broadband noise) while nobody sang; mic recorded `mic`, both on one
        time base (mic[t] contains the echo of ref up to t). process() uses the block it emitted on the
        PREVIOUS call as the newest reference, so the sound check is run with the reference one block late
        to learn the same alignment."""
        if self.aec is None:
            return None
        B = self.B
        ref = np.concatenate([np.zeros(B), np.asarray(ref, float)])[:len(ref)]
        n = min(len(mic), len(ref)) // B * B
        c = EchoCanceller(partitions=self.partitions, block=B, mu=0.8, reg=0.01, two_path=False, dtd=0.25)
        out = np.zeros(n)
        for k in range(0, n, B):
            out[k:k + B] = c.process(mic[k:k + B], ref[k:k + B])
        self.aec = FrozenMDF(c.Wb, B)
        q = n // 4
        return float(10 * np.log10(np.sum(mic[n - q:n] ** 2) / (np.sum(out[n - q:] ** 2) + 1e-20)))

    # ---------------------------------------------------------------- one block
    def process(self, mic):
        B = self.B
        mic = np.asarray(mic, float)
        if self.aec is not None:
            e = self.aec.process(mic, self.ref_prev)
            yhat = mic - e
        else:
            e, yhat = mic, np.zeros(B)
        self.inbuf = np.concatenate([self.inbuf[B:], e])
        f0, ap = self.yin(self.inbuf)
        Lb = 10 * np.log10(np.mean(self.inbuf[-2 * B:] ** 2) + 1e-12)      # ~5 ms level
        Ly = 10 * np.log10(np.mean(yhat ** 2) + 1e-12)
        periodic = np.isfinite(f0) and ap < self.ap_stay
        voiced = periodic and ap < self.ap_open
        # noise floor: fast fall, slow rise only while the gate is closed
        if Lb < self.N_floor:
            self.N_floor = 0.7 * self.N_floor + 0.3 * Lb
        elif not self.gate_open:
            self.N_floor += 0.5 * B / self.sr
        midi = float(hz_to_midi(f0)) if periodic else np.nan
        # ---- gate decision
        self.lv.append(Lb)
        lv = self.lv
        rise = Lb > min(list(lv)[-12:-1] or [Lb]) + 6.0            # an onset: level rose >= 6 dB in 30 ms
        drop = Lb < max(lv) - 10.0                                  # an offset: fell >= 10 dB within 50 ms
        want_open = (voiced and Lb > max(self.N_floor + self.open_snr, -60.0)
                     and (Lb > self.L_ref - 14.0 or rise))
        if want_open and self.guard:
            if self.aec is not None and Lb < Ly + self.echo_margin:
                want_open = False                          # mostly explained by the echo estimate
            if want_open and not self.gate_open and Lb < self.L_ref - 6.0 and self._echo_pitch(midi):
                want_open = False                          # quiet and at the pitch Duet just played
        if self.gate_open:
            self.unv = self.unv + 1 if not periodic else 0
            close = (Lb < max(self.L_ref - self.close_db, self.N_floor + 9.0, -65.0)) or self.unv >= self.unv_close or drop
            if close:
                self.gate_open = False
        elif want_open:
            self.gate_open = True
            self.unv = 0
        if self.gate_open:
            self.L_ref = max(Lb, self.L_ref - self.ref_decay)
        else:
            self.L_ref = max(self.L_ref - self.ref_decay, -40.0)
        # ---- note + shift decision
        if periodic and self.gate_open:
            self.T0 = self.sr / f0
            self.f0 = f0
            self.n_t += 1
            self.tz += max(1.0 / self.n_t, self.tz_a) * (np.exp(2j * np.pi * midi) - self.tz)
            self.tun = float(np.angle(self.tz) / (2 * np.pi)) if abs(self.tz) > 0.05 else 0.0
            mt = midi - self.tun
            q = int(np.round(mt))
            if self.note is None:
                self.note, self.cand_n = q, 0
            elif q != self.note and abs(mt - self.note) > self.hyst:
                if q == self.cand:
                    self.cand_n += 1
                else:
                    self.cand, self.cand_n = q, 1
                if self.cand_n >= self.hyst_blocks:
                    self.note, self.cand_n = q, 0
            else:
                self.cand_n = 0
            target = third_below(self.note, self.tonic)
            self.shift = self.note - target
            dev = mt - self.note
            if self.mode == 'correct' and self.cand_n == 0 and abs(dev) <= 0.5:
                # on a note: the harmony sits on the diatonic note (her tuning), keeping her vibrato /
                # intonation within +-vib_clip semitones, so it stays on the note grid
                dev = float(np.clip(dev, -self.vib_clip, self.vib_clip))
                self.ratio = 2.0 ** ((target + self.tun + dev - midi) / 12.0)
            else:
                self.ratio = 2.0 ** (-self.shift / 12.0)
        ratio = self.ratio
        y = self.psola.process(e, self.T0, ratio)
        # ---- gate gain ramp (per sample, linear inside the block)
        g0 = self.gate
        g1 = 1.0 if self.gate_open else g0 * self.rel_coef
        if g1 < 1e-3:
            g1 = 0.0
        self.gate = g1
        y = y * (g0 + (g1 - g0) * self.ramp) * self.out_gain
        # ---- limiter
        pk = float(np.max(np.abs(y))) + 1e-12
        target = min(1.0, self.ceiling / pk)
        l0 = self.lim
        l1 = target if target < l0 else min(target, l0 * self.lim_rel)
        self.lim = l1
        l0 = min(l0, target)
        y = y * (l0 + (l1 - l0) * self.ramp)
        y = np.clip(y, -0.99, 0.99)
        if self.gate_open and periodic:
            self.out_pitch_hist.append((self.k, midi + 12 * np.log2(ratio)))
        while self.out_pitch_hist and self.k - self.out_pitch_hist[0][0] >= self.hist_blocks:
            self.out_pitch_hist.popleft()
        if self.log is not None:
            self.log.append((self.k, f0 if voiced else np.nan, ap, Lb, Ly, int(self.gate_open), self.gate,
                             self.note if self.note is not None else -1, self.shift, self.L_ref, self.N_floor))
        self.ref_prev = y
        self.k += 1
        return y

    def _echo_pitch(self, midi):
        if not self.out_pitch_hist or np.isnan(midi):
            return False
        for _, m in self.out_pitch_hist:
            d = abs(midi - m) % 12.0
            if min(d, 12.0 - d) < 0.6:
                return True
        return False


def run_open(x, tonic, **kw):
    """Run the engine over a whole signal with no echo (open loop). -> (output, engine)."""
    fx = Harmonizer(tonic=tonic, **kw)
    B = fx.B
    n = len(x) // B * B
    y = np.zeros(n)
    for k in range(0, n, B):
        y[k:k + B] = fx.process(x[k:k + B])
    return y, fx
