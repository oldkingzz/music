"""Closed-loop simulation of speaker->mic bleed ("acoustic echo") in a Duet performance.

The existing engine (FeatureTracker / Follower / Accompanist from follow.py) is imported
unmodified. What this file adds around it:

* StreamVoice  - renders the machine's voice causally, 10 ms at a time, from the notes the
                 Accompanist has committed so far (same timbre/envelope as dsp.render_voice).
* EchoPath     - loudspeaker -> microphone path: system delay + acoustic delay, small-speaker
                 high-pass, optional speaker distortion, exponential-decay room tail (RT60).
* run()        - one performance: at every 10 ms step the mic frame is
                     singer(frame) + gain * echo(machine voice so far)
                 optionally cleaned by an echo guard, then fed to the engine.  Because the
                 machine voice depends on what the engine decided from earlier mic frames,
                 the loop is closed exactly as it would be on the device.
* score()      - the same metrics as run_v0.run_scenario (predictive asynchrony vs the
                 true onsets, tracking error, played/held/missed/wrong, false entries).

Nothing here writes files. sweep.py drives it.
"""
import functools
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import perform  # noqa: E402
import singer  # noqa: E402
from dsp import HOP, SR, colored_noise, midi_to_hz, render_voice  # noqa: E402
from follow import (Accompanist, FeatureTracker, Follower, analyse, offline_notes,  # noqa: E402
                    plan_harmony)
from run_v0 import SCENARIOS, calibrate, find_gaps, match  # noqa: E402

C_SOUND = 343.0          # m/s at 20 C
VOICE_LEVEL = 0.22       # dsp.render_voice default level
SOUNDCHECK_RMS = 0.05    # sound-check noise, ~10 dB below the machine voice


# ---------------------------------------------------------------- reference song
@functools.lru_cache(maxsize=None)
def synthetic_context():
    """Everything run_v0.main() builds before the scenarios, for the synthetic singer."""
    s_notes = singer.score_notes()
    xs, true_s = singer.render(s_notes)
    F = analyse(xs)
    det = offline_notes(F)
    gaps = find_gaps(F)
    ref_bias, live_bias, _ = calibrate(xs, true_s, F, det, [(a + b) / 2 for a, b in gaps])
    true_on = [n['on'] for n in true_s]
    truth_on = [true_on[k] if k is not None else None for k in match([n['on'] for n in det], true_on)]
    for nt in det:
        nt['on'] -= ref_bias.get(nt['kind'], 0.0)
    plan, key = plan_harmony(det)
    dur = len(xs) / SR
    join_r = next((nt['on'] for nt in det if nt['on'] >= 0.25 * dur and nt['kind'] == 'attack'),
                  next(nt['on'] for nt in det if nt['on'] >= 0.25 * dur))
    fr = np.sqrt(np.mean(xs[:len(xs) // HOP * HOP].reshape(-1, HOP) ** 2, axis=1))
    singer_rms = float(np.sqrt(np.mean(fr[fr > 0.02] ** 2)))   # RMS over her sung 10 ms frames
    return dict(x=xs, F=F, notes=det, plan=plan, key=key, gaps=gaps, join_r=join_r,
                truth_on=truth_on, bias_live=live_bias, singer_rms=singer_rms)


# ---------------------------------------------------------------- machine voice, streamed
class StreamVoice:
    """Causal version of dsp.render_voice driven by Accompanist.pred.

    A note is known once it is committed (always >= compute + output latency before it
    sounds). Its length is the planned length x tempo, cut 20 ms before the next committed
    note (as Accompanist.machine_notes does). A note the Accompanist takes back (a false
    phrase entry) stops at that moment with a 60 ms release, as it would on the device.
    """

    def __init__(self, level=VOICE_LEVEL):
        self.level = level
        self.notes = {}            # (plan index, at) -> state

    def sync(self, acc, now):
        played = {(j, round(v['at'], 6)): v for j, v in acc.pred.items() if v.get('status') == 'played'}
        for key, v in played.items():
            if key not in self.notes:
                j = key[0]
                self.notes[key] = {'at': v['at'], 'i0': int(round(v['at'] * SR)), 'midi': acc.plan[j]['h'],
                                   'dur': acc.plan[j]['dur'] * v['b'], 'ph': 0.0, 'done': False}
        for key, st in self.notes.items():
            if key not in played and not st.get('retracted'):
                st['retracted'] = True
                st['dur'] = min(st['dur'], max(0.0, now - st['at']))
        live = sorted((st['at'], k) for k, st in self.notes.items() if not st.get('retracted'))
        for (a0, k0), (a1, _) in zip(live[:-1], live[1:]):
            st = self.notes[k0]
            if st.get('next_at') == a1:
                continue                              # end already fixed for this successor
            st['next_at'] = a1
            cut = max(min(st['dur'], a1 - a0 - 0.02), 0.08)
            if cut < st['dur']:
                # never jump the envelope: if the cut point is already past, release from now
                st['dur'] = min(st['dur'], max(cut, now - a0))
        for k in [k for k, st in self.notes.items() if st.get('retracted') and st['i0'] >= now * SR]:
            del self.notes[k]                         # taken back before it ever sounded

    def render(self, i_start, n=HOP):
        out = np.zeros(n)
        midis = []
        for st in self.notes.values():
            if st['done']:
                continue
            length = int((st['dur'] + 0.08) * SR)
            a = max(i_start, st['i0'])
            b = min(i_start + n, st['i0'] + length)
            if st['i0'] + length <= i_start:
                st['done'] = True
                continue
            if b <= a:
                continue
            t = (np.arange(a, b) - st['i0']) / SR
            vib = 0.15 * np.sin(2 * np.pi * 5.0 * t) * np.clip((t - 0.15) / 0.2, 0.0, 1.0)
            ph = st['ph'] + 2 * np.pi * np.cumsum(midi_to_hz(st['midi'] + vib)) / SR
            st['ph'] = float(ph[-1])
            y = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
            env = np.clip(t / 0.025, 0.0, 1.0) * np.clip((st['dur'] + 0.06 - t) / 0.06, 0.0, 1.0)
            out[a - i_start:b - i_start] += y * env * self.level
            if env[-1] > 0.3:
                midis.append(st['midi'])
        return out, midis


# ---------------------------------------------------------------- loudspeaker -> mic path
def room_ir(acoustic_ms=2.0, rt60=0.3, room_m3=40.0, q=2.0, sys_ms=20.0, hp_hz=250.0, seed=0,
            length_s=None):
    """Impulse response from the reference the host sends to the mic samples it receives.

    sys_ms      : USB jitter buffer (2 frames per PLAN) + converter latency; not acoustic.
    acoustic_ms : speaker->mic distance / c  (1-3 ms = 0.34-1.03 m).
    Direct-to-reverberant ratio from the critical distance r_c = 0.057 sqrt(Q V / RT60)
    (diffuse-field room acoustics), so a farther mic gets relatively more room tail.
    Tail: Gaussian noise x exp(-6.91 t / RT60) (amplitude falls 60 dB in RT60), starting
    at the first reflection ~1.5 ms after the direct sound.
    """
    rng = np.random.default_rng(seed)
    length_s = length_s or rt60
    n = int((sys_ms + acoustic_ms) / 1000 * SR) + int(length_s * SR)
    h = np.zeros(n)
    d0 = (sys_ms + acoustic_ms) / 1000 * SR
    k0 = int(np.floor(d0))
    frac = d0 - k0                                       # fractional-delay direct path (linear interp)
    r = acoustic_ms / 1000 * C_SOUND
    h[k0] += (1 - frac) / max(r, 0.05)
    h[k0 + 1] += frac / max(r, 0.05)
    direct_e = float(np.sum(h ** 2))
    r_c = 0.057 * np.sqrt(q * room_m3 / rt60)
    drr_db = 20 * np.log10(r_c / max(r, 0.05))
    t0 = k0 + int(0.0015 * SR)
    tt = np.arange(n - t0) / SR
    tail = rng.standard_normal(n - t0) * np.exp(-6.9078 * tt / rt60)
    tail *= np.sqrt(direct_e / 10 ** (drr_db / 10) / np.sum(tail ** 2))
    h[t0:] += tail
    if hp_hz:                                            # small full-range driver: 2nd-order roll-off
        b, a = butter(2, hp_hz / (SR / 2), 'high')
        h = lfilter(b, a, h)
    return h, {'drr_db': float(drr_db), 'distance_m': float(r), 'critical_distance_m': float(r_c),
               'delay_ms': sys_ms + acoustic_ms}


class EchoPath:
    """Streaming FIR echo path with an optional memoryless speaker non-linearity."""

    def __init__(self, h, thd=0.0):
        self.h = h
        self.zi = np.zeros(len(h) - 1)
        self.thd = thd

    def distort(self, s):
        if not self.thd:
            return s
        # 2nd + 3rd order polynomial; coefficients give ~thd (amplitude ratio) at full voice level
        a = self.thd / VOICE_LEVEL
        return s + 0.6 * a * s * s + 0.8 * a * s ** 3 / VOICE_LEVEL

    def step(self, spk):
        y, self.zi = lfilter(self.h, [1.0], self.distort(spk), zi=self.zi)
        return y


def nominal_echo_rms(h):
    """RMS at the mic of two sustained machine notes (A3, D4) through path h, gain 1."""
    v = render_voice([(0.0, 2.5, 57), (2.5, 2.5, 62)], int(5.2 * SR))
    e = fftconvolve(v, h)[:len(v)]
    seg = e[int(0.5 * SR):int(4.5 * SR)]
    return float(np.sqrt(np.mean(seg ** 2)))


def echo_gain(h, bleed_db, singer_rms):
    """Scale so the echo of a sustained machine note sits bleed_db re the singer's voiced RMS."""
    return 10 ** (bleed_db / 20) * singer_rms / nominal_echo_rms(h)


# ---------------------------------------------------------------- one closed-loop performance
def run(ctx, scen='natural', bleed_db=None, guard=None, ir=None, thd=0.0, compute=0.002,
        out_latency=0.005, soundcheck=None, keep_audio=False, ir_after=None, change_frac=0.4):
    """Simulate one performance.

    guard: None or an object with
        calibrate(mic, ref)            sound check (only called if soundcheck = seconds)
        cancel(mic240, ref240) -> mic  before the FeatureTracker
        mask(ft, expected)             edits the newest feature frame (optional)
    ir_after: if given, the echo path cross-fades (0.5 s) to this IR at change_frac of the song
              (somebody moved / the mic was bumped), to test how the canceller copes.
    compute: fixed engine compute time passed to Accompanist (run_v0 measures it; fixed here
             so that runs are deterministic).
    Returns (metrics, extras)."""
    spec = dict(SCENARIOS[scen])
    spec.pop('rehearsal', None)
    x, F, notes, plan = ctx['x'], ctx['F'], ctx['notes'], ctx['plan']
    gaps = ctx['gaps']
    perf = perform.make(x, gaps, **spec)
    y = perf.audio.astype(np.float64)
    ft, fol = FeatureTracker(), Follower(F)
    breaths = [(a + b) / 2 for a, b in gaps]
    acc = Accompanist(notes, plan, ctx['join_r'], out_latency=out_latency, bias_live=ctx['bias_live'],
                      breaths=breaths)
    voice = StreamVoice()
    if ir is None:
        ir, _ = room_ir()
    path = EchoPath(ir, thd=thd)
    path2 = EchoPath(ir_after, thd=thd) if ir_after is not None else None
    k_change = int(change_frac * len(y) / HOP)
    g = 0.0 if bleed_db is None else echo_gain(ir, bleed_db, ctx['singer_rms'])
    if guard is not None and soundcheck:
        # 2 s of soft pink noise from the speaker before the song; the room noise is at the
        # same level perform.make uses (-60 dB re her voice)
        rng = np.random.default_rng(11)
        ns = int(soundcheck * SR) // HOP * HOP
        ref_sc = SOUNDCHECK_RMS * colored_noise(ns, rng, -0.5)
        mic_sc = g * EchoPath(ir, thd=thd).step(ref_sc) + ctx['singer_rms'] * 1e-3 * colored_noise(ns, rng, -0.5)
        guard.calibrate(mic_sc, ref_sc)
    track_err, guard_ms = [], []
    n_frames = len(y) // HOP
    keep = {'mic': [], 'clean': [], 'spk': [], 'echo': []} if keep_audio else None
    echo_e = sing_e = 0.0
    echo_n = sing_n = 0
    for k in range(n_frames):
        now = (k + 1) * HOP / SR
        voice.sync(acc, k * HOP / SR)
        spk, midis = voice.render(k * HOP)
        e = g * path.step(spk) if g else np.zeros(HOP)
        if path2 is not None:
            e2 = g * path2.step(spk) if g else np.zeros(HOP)
            w = np.clip((k - k_change + np.arange(HOP) / HOP) / 50.0, 0.0, 1.0)
            e = (1 - w) * e + w * e2
        s = y[k * HOP:(k + 1) * HOP]
        mic = s + e
        if np.any(spk):
            echo_e += float(np.sum(e * e))
            echo_n += HOP
        if np.sqrt(np.mean(s * s)) > 0.02:
            sing_e += float(np.sum(s * s))
            sing_n += HOP
        t0 = time.perf_counter()
        clean = guard.cancel(mic, spk) if guard is not None else mic
        guard_ms.append((time.perf_counter() - t0) * 1000)
        ft.step(clean)
        if guard is not None and hasattr(guard, 'mask'):
            nk = acc.k
            exp = [notes[i]['midi'] for i in (nk - 1, nk, nk + 1) if 0 <= i < len(notes)]
            t0 = time.perf_counter()
            guard.mask(ft, exp)
            guard_ms[-1] += (time.perf_counter() - t0) * 1000
        fol.step(ft.m[-1], ft.v[-1], ft.nov[-1])
        acc.step(now, ft, fol, compute)
        r_true = perf.live_to_ref(ft.t(ft.i))[0]
        if ft.v[-1] and not np.isnan(r_true):
            track_err.append((ft.t(ft.i), fol.ref_time() - r_true))
        if keep is not None:
            keep['mic'].append(mic)
            keep['clean'].append(np.asarray(clean))
            keep['spk'].append(spk)
            keep['echo'].append(e)
    metrics = score(acc, fol, perf, ctx, track_err)
    metrics['scenario'] = scen
    metrics['bleed_db_set'] = bleed_db
    metrics['bleed_db_measured'] = (float(10 * np.log10((echo_e / max(echo_n, 1)) / (sing_e / max(sing_n, 1))))
                                    if echo_e > 0 else None)
    metrics['guard_ms_median'] = float(np.median(guard_ms))
    metrics['guard_ms_p99'] = float(np.percentile(guard_ms, 99))
    extras = {'acc': acc, 'ft': ft, 'fol': fol, 'perf': perf}
    if keep is not None:
        extras.update({k: np.concatenate(v) for k, v in keep.items()})
    if guard is not None and hasattr(guard, 'stats'):
        metrics['guard'] = guard.stats()
    return metrics, extras


def score(acc, fol, perf, ctx, track_err):
    """Same definitions as run_v0.run_scenario (predictive path only + tracking)."""
    truth_on = ctx['truth_on']
    errs = []
    status = {'played': 0, 'held': 0, 'missed': 0, 'not_decided': 0, 'wrong': 0}
    for j, p in enumerate(acc.plan):
        t_true = perf.ref_to_live(truth_on[p['note']]) if truth_on[p['note']] is not None else float('nan')
        v = acc.pred.get(j)
        if v is None:
            status['not_decided'] += 1
            continue
        status[v['status']] += 1
        if v['status'] == 'played':
            if np.isnan(t_true):
                status['wrong'] += 1
            else:
                errs.append(v['at'] - t_true)
    a = np.abs(errs) if errs else np.array([np.nan])
    te = np.array([e for _, e in track_err]) if track_err else np.array([np.nan])
    return {
        'n_plan': len(acc.plan), 'n_err': len(errs),
        'median_abs_ms': float(np.median(a) * 1000), 'p90_abs_ms': float(np.percentile(a, 90) * 1000),
        'mean_signed_ms': float(np.mean(errs) * 1000) if errs else None,
        'within30': float(np.mean(a <= 0.030)), 'within60': float(np.mean(a <= 0.060)),
        'errors_ms': [round(e * 1000, 2) for e in errs], **status,
        'false_entries': len(acc.false_entries),
        'tracking_within100ms': float(np.mean(np.abs(te) < 0.1)),
        'tracking_lost_frac': float(np.mean(np.abs(te) > 0.25)),
        'resets': fol.resets, 'anchors': len(acc.anchors),
    }


if __name__ == '__main__':
    ctx = synthetic_context()
    for scen in ('steady', 'natural'):
        m, _ = run(ctx, scen)
        print(scen, {k: v for k, v in m.items() if k != 'errors_ms'})
