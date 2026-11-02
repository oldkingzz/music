"""The audio path (spec E1-E6): a synthetic singer, the click, the repo's YIN front end, Duet's voice.

AudioSource offers the same two calls as planner.ScoreSource, but computed from what the
microphone front end actually reports, frame by frame:
  full(k)            every frame whose audio lies inside beat k;
  partial(k, frac)   only frames that have ARRIVED by the decision time tau_k + frac * T_b
                     (a frame arrives 10 ms after its block and describes the audio 20 ms
                     before that, follow.FeatureTracker.t), so the decision cannot use the future.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import singer  # noqa: E402  (repo: synthetic 'ah' singer with vibrato and glides)
from dsp import HOP, SR, midi_to_hz, render_voice, write_wav  # noqa: E402
from follow import FeatureTracker  # noqa: E402


def her_notes(song, t0, T_b, gap=0.0):
    return [(t0 + on * T_b, t0 + off * T_b - gap, m) for on, off, m in song['sop']]


def render_clean(notes, tail=0.6, level=0.7):
    """A clean 'ah' voice: no vibrato, no glides, no drift (assumption A4)."""
    n = int((notes[-1][1] + tail) * SR)
    y = np.zeros(n)
    for on, off, m in notes:
        i0, i1 = int(on * SR), int(off * SR)
        t = np.arange(i1 - i0) / SR
        f0 = midi_to_hz(m)
        ph = 2 * np.pi * f0 * t
        s = np.zeros_like(t)
        for h in range(1, 30):
            fh = h * f0
            if fh > 5000:
                break
            s += singer._formant_gain(np.array([fh]))[0] / h * np.sin(h * ph)
        dur = t[-1] if len(t) else 0
        env = np.clip(t / 0.03, 0, 1) * np.clip((dur - t) / 0.04, 0, 1)
        y[i0:i1] += s * env
    return (y * level / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def render_her(song, t0, T_b, style='clean'):
    if style == 'clean':
        return render_clean(her_notes(song, t0, T_b, gap=0.03))
    y, _ = singer.render(her_notes(song, t0, T_b))       # vibrato, glides, drift, +-0.1 st intonation
    return y


def click(n, t0, T_b, K, M, count_in):
    y = np.zeros(n, np.float32)
    for k in range(-count_in, K):
        i0 = int((t0 + k * T_b) * SR)
        if i0 < 0 or i0 >= n:
            continue
        L = int(0.025 * SR)
        t = np.arange(min(L, n - i0)) / SR
        f = 1760.0 if k % M == 0 else 1320.0
        y[i0:i0 + len(t)] += (0.35 if k % M == 0 else 0.22) * np.sin(2 * np.pi * f * t) * np.exp(-t / 0.006)
    return y


class AudioSource:
    def __init__(self, audio, t0, T_b, K):
        ft = FeatureTracker()
        for i in range(0, len(audio) - HOP + 1, HOP):
            ft.step(audio[i:i + HOP].astype(float))
        a = ft.arrays()
        self.t = a['t']                                   # when the frame's audio happened
        self.arrive = (np.arange(len(self.t)) + 1) * HOP / SR
        self.midi, self.v = a['midi'], a['voiced']
        self.t0, self.T_b, self.K = t0, T_b, K
        self.fr = HOP / SR / T_b                          # one frame, in beats
        nov = a['nov']
        rise = (nov >= 0.5) & np.concatenate([[True], nov[:-1] < 0.5]) & self.v
        self.on_t, self.on_arrive = self.t[rise], self.arrive[rise]   # detected note onsets (E17)
        self.sync_grid = 0.5                              # E17 grid: nearest half beat (1/3 for triplets)

    def drift_ms(self, dec_time, m=4):
        """E17: median distance (ms) of her last m heard onsets from the nearest grid point, or None."""
        sel = self.on_arrive <= dec_time
        ts = self.on_t[sel][-m:]
        if len(ts) < m:
            return None
        g = self.sync_grid * self.T_b
        e = (ts - self.t0) - np.round((ts - self.t0) / g) * g
        return float(np.median(e) * 1000.0)

    def _obs(self, sel):
        idx = np.where(sel & self.v)[0]
        vec = np.zeros(12)
        if len(idx) == 0:
            return vec, None
        r = np.round(self.midi[idx]).astype(int)
        for m in r:
            vec[m % 12] += self.fr
        pc = int(np.argmax(vec))
        return vec, int(np.median(r[(r % 12) == pc]))

    def full(self, k):
        a = self.t0 + k * self.T_b
        return self._obs((self.t >= a) & (self.t < a + self.T_b))

    def partial(self, k, frac):
        a = self.t0 + k * self.T_b
        dec = a + frac * self.T_b
        return self._obs((self.t >= a) & (self.t < dec) & (self.arrive <= dec))


def duet_notes(beats, t0, T_b, joined_only=True):
    """Merge Duet's per-beat notes into sung notes (a held pitch is one long note)."""
    out, cur = [], None
    for b in beats:
        if b.h < 0:
            if cur:
                out.append(cur)
                cur = None
            continue
        start = t0 + b.k * T_b
        if cur and cur[2] == b.h and abs(cur[0] + cur[1] - start) < 1e-6:
            cur = (cur[0], cur[1] + T_b, cur[2])
        else:
            if cur:
                out.append(cur)
            cur = (start, T_b, b.h)
    if cur:
        out.append(cur)
    return out


def mix(her, duet_n, clk, path_wav, path_mp3=None, duet_level=0.30):
    n = max(len(her), len(clk))
    her = np.pad(her, (0, n - len(her)))
    clk = np.pad(clk, (0, n - len(clk)))
    du = render_voice([(s, max(d - 0.03, 0.05), m) for s, d, m in duet_n], n, level=duet_level)
    left = 0.75 * her + 0.35 * du + 0.6 * clk
    right = 0.35 * her + 0.75 * du + 0.6 * clk
    st = np.stack([left, right], 1)
    st *= 0.9 / (np.max(np.abs(st)) + 1e-9)
    write_wav(path_wav, st.astype(np.float32))
    if path_mp3:                                   # the page serves .mp3, not .m4a
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path_wav, '-c:a', 'libmp3lame', '-b:a', '128k', path_mp3], check=True)
    return st
