"""Glue between the live app and echo/ (speaker -> mic bleed): the guard, the sound check and,
for file mode, a simulated room so the guard can be exercised offline.

echo/guard.py and echo/sim.py are imported unmodified (under private module names, because
echo/ has no package __init__ and live/ must not shadow other files called guard.py / sim.py).

How the guard sits in the loop (Engine.process):

    clean = guard.cancel(mic_block_k, ref)     ref = the speaker block the host SENT after block k-1
    FeatureTracker.step(clean)
    guard.mask(ft, expected)                   expected = MIDI of notes[acc.k-1 .. acc.k+1]

ref alignment is the same in every backend: the block returned by the previous Engine.process
call (file: written to the speaker timeline; mac: written to the PortAudio output buffer;
device: sent as SPK_AUDIO right after that mic frame). The echo of ref reaches the mic about
L + acoustic delay later; the canceller learns that delay (and the room) during the sound check,
and the pitch mask gets it from GCC-PHAT on the same recording.

Sound check (echo/README.md): 2 s of soft pink noise (RMS 0.05 FS, ~10 dB under the machine
voice) is played while the mic records, before the song. A canceller started cold during the
song is worse than none (it learns to cancel the singer), so if the sound check is skipped the
canceller is switched off and only the pitch mask runs.
"""
import importlib.util
import sys
import threading
import time
from pathlib import Path

import numpy as np

from dsp import HOP, SR, colored_noise

ROOT = Path(__file__).resolve().parent.parent
SOUNDCHECK_RMS = 0.05            # echo/sim.py SOUNDCHECK_RMS
DEFAULT_ACOUSTIC_MS = 2.0        # speaker -> mic distance 0.7 m (echo/sim.room_ir default)


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def guard_module():
    return _load('duet_echo_guard', ROOT / 'echo' / 'guard.py')


def sim_module():
    return _load('duet_echo_sim', ROOT / 'echo' / 'sim.py')


def make_guard(aec=True, mask=True, freeze_after_soundcheck=True):
    """echo/guard.EchoGuard. freeze_after_soundcheck=True (live default): after the sound check the
    canceller's output (foreground) filter is kept fixed for the whole song; the background
    filter still adapts but is never copied over (its copy_ratio is set to 0 - a constructor
    parameter of echo/guard.EchoCanceller, so echo/ is unchanged). Reason, measured in
    live/tests/run_tests.py section 10 and live/README.md: in the 'pause' scenario at >= 0 dB bleed
    the background filter, adapting while the machine's tonal false entry is the only sound in
    the room, got copied to the foreground 3 times and the duet lost track (9/26 notes); frozen,
    25-26/26 in every scenario up to +12 dB. Cost: a mid-song change of the echo path (someone
    moves the mic) is not tracked; repeat the sound check instead (echo/README.md says the same)."""
    g = guard_module().EchoGuard(aec=aec, mask=mask)
    g.freeze_after_soundcheck = freeze_after_soundcheck
    return g


def soundcheck_noise(n, seed=11):
    """The sound-check signal: pink noise, RMS 0.05 full scale (same as echo/sim.py)."""
    return (SOUNDCHECK_RMS * colored_noise(n, np.random.default_rng(seed), -0.5)).astype(np.float32)


def flush_history(guard):
    """Forget the reference history after the sound check (the gap before the song is silent):
    otherwise the canceller would subtract the echo of noise that is no longer playing."""
    if guard.aec is not None:
        z = np.zeros(HOP)
        for _ in range(guard.aec.P):
            guard.aec.process(z, z, adapt=False)
    if guard.pm is not None:
        guard.pm.ref[:] = 0.0
        guard.pm.mic[:] = 0.0


def expected_midis(ref_notes, k):
    return [ref_notes[i]['midi'] for i in (k - 1, k, k + 1) if 0 <= i < len(ref_notes)]


# ---------------------------------------------------------------- the sound check, block by block
class SoundCheck:
    """Runs inside a backend's block loop BEFORE the engine starts: feed(mic_block) returns the
    speaker block to send. Phases: 'noise' (pink noise out, mic and ref recorded) -> 'calibrate'
    (silence out while guard.calibrate runs, in a worker thread for the real-time backends) ->
    'done' after at least gap_s of silence (so the noise and its room tail have died away).

    ref pairing is the engine's: mic block i is paired with the block sent after mic block i-1."""

    def __init__(self, guard, seconds=2.0, gap_s=0.3, seed=11, threaded=True, log=print):
        self.guard = guard
        self.nb = max(1, int(round(seconds * SR / HOP)))
        self.noise = soundcheck_noise(self.nb * HOP, seed)
        self.gap_blocks = int(round(gap_s * SR / HOP))
        self.threaded = threaded
        self.log = log
        self.i = 0
        self.mic, self.ref = [], []
        self.prev = np.zeros(HOP, np.float32)
        self.phase = 'noise'
        self.worker = None
        self.cal_done = threading.Event()
        self.gap = 0
        self.result = {}
        self.t_start = None

    @property
    def done(self):
        return self.phase == 'done'

    def _calibrate(self):
        t0 = time.perf_counter()
        mic = np.concatenate(self.mic).astype(np.float64)
        ref = np.concatenate(self.ref).astype(np.float64)
        try:
            erle = self.guard.calibrate(mic, ref)
            if self.guard.aec is not None and getattr(self.guard, 'freeze_after_soundcheck', False):
                self.guard.aec.copy_ratio = 0.0
            lag = None
            if self.guard.pm is not None:
                lag = int(self.guard.pm.delay)
            else:
                lag = guard_module().estimate_delay(mic, ref)
            mic_rms = float(np.sqrt(np.mean(mic ** 2)))
            self.result = {'erle_db': None if erle is None else float(erle), 'delay_samples': lag,
                           'delay_ms': None if lag is None else lag / SR * 1000, 'mic_rms': mic_rms,
                           'echo_to_noise_db': float(20 * np.log10(mic_rms / SOUNDCHECK_RMS + 1e-12)),
                           'calibrate_s': round(time.perf_counter() - t0, 3), 'blocks': self.nb,
                           'canceller_frozen': bool(self.guard.aec is not None and self.guard.aec.copy_ratio == 0.0)}
        except Exception as ex:                       # never kill the performance over the sound check
            self.result = {'error': repr(ex)}
        self.cal_done.set()

    def feed(self, x):
        if self.phase == 'noise':
            if self.t_start is None:
                self.t_start = time.perf_counter()
            self.mic.append(np.asarray(x, np.float32).copy())
            self.ref.append(self.prev)
            y = self.noise[self.i * HOP:(self.i + 1) * HOP]
            self.prev = y
            self.i += 1
            if self.i >= self.nb:
                self.phase = 'calibrate'
                if self.threaded:
                    self.worker = threading.Thread(target=self._calibrate, daemon=True, name='soundcheck')
                    self.worker.start()
                else:
                    self._calibrate()
            return y
        if self.phase == 'calibrate':
            self.gap += 1
            if self.cal_done.is_set() and self.gap >= self.gap_blocks:
                flush_history(self.guard)
                self.phase = 'done'
                self.result['gap_blocks'] = self.gap
                r = self.result
                if 'error' in r:
                    self.log(f'sound check FAILED: {r["error"]}', flush=True)
                else:
                    self.log(f"sound check: {self.nb * HOP / SR:.1f} s pink noise, echo-path delay "
                             f"{r['delay_ms'] if r['delay_ms'] is not None else float('nan'):.1f} ms, canceller ERLE "
                             f"{r['erle_db'] if r['erle_db'] is not None else float('nan'):.1f} dB "
                             f"(mic/noise {r['echo_to_noise_db']:.1f} dB)", flush=True)
            self.prev = np.zeros(HOP, np.float32)
            return self.prev
        return np.zeros(HOP, np.float32)


# ---------------------------------------------------------------- simulated room (file mode)
def voiced_rms(x, rel=0.1):
    """RMS over the singer's sung 10 ms frames: frames above rel x the 95th-percentile frame RMS.
    (echo/sim.synthetic_context uses a fixed 0.02 FS threshold, which only suits the synthetic
    singer's level; a relative one also works for real recordings at other levels.)"""
    x = np.asarray(x, np.float64)
    fr = np.sqrt(np.mean(x[:len(x) // HOP * HOP].reshape(-1, HOP) ** 2, axis=1))
    sel = fr[fr > rel * np.percentile(fr, 95)]
    return float(np.sqrt(np.mean(sel ** 2))) if sel.size else float(np.sqrt(np.mean(x ** 2)))


class BleedSim:
    """The echo agent's room model (echo/sim.py room_ir + EchoPath) applied to the machine voice
    on the acoustic timeline. The IR has NO system delay here (sys_ms = 0): in the live engine the
    output latency L already places each output block at its acoustic time, so the reference ->
    mic delay seen by the guard is L + acoustic delay, as on the device.

    bleed_db: level of a sustained machine note at the mic relative to the singer's voiced RMS
    (echo/sim.echo_gain). 0 dB = the machine is as loud as she is at the mic."""

    def __init__(self, x_singer, bleed_db, acoustic_ms=DEFAULT_ACOUSTIC_MS, rt60=0.3, thd=0.0, seed=0):
        sim = sim_module()
        self.h, self.info = sim.room_ir(acoustic_ms=acoustic_ms, rt60=rt60, sys_ms=0.0, seed=seed)
        self.singer_rms = voiced_rms(x_singer)
        self.gain = float(sim.echo_gain(self.h, bleed_db, self.singer_rms))
        self.bleed_db = bleed_db
        self.thd = thd
        self.path = sim.EchoPath(self.h, thd=thd)
        self.floor_rng = np.random.default_rng(seed + 7)

    def fresh_path(self):
        return sim_module().EchoPath(self.h, thd=self.thd)

    def step(self, spk_block, path=None):
        return self.gain * (path or self.path).step(np.asarray(spk_block, np.float64))

    def room_floor(self, n):
        """Room noise during the sound check: -60 dB re her voice (as echo/sim.run)."""
        return self.singer_rms * 1e-3 * colored_noise(n, self.floor_rng, -0.5)

    def describe(self):
        return {'bleed_db': self.bleed_db, 'gain': self.gain, 'singer_voiced_rms': self.singer_rms,
                'room': self.info, 'thd': self.thd, 'model': 'echo/sim.py room_ir(sys_ms=0) + EchoPath'}
