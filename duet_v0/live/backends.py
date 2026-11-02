"""Audio/IO backends. Each one only moves 10 ms blocks in and out; all processing is
Engine.process(), the same call in every backend.

file    a wav is fed block by block, at real-time pace (sleeping until each block would have
        arrived) or as fast as possible (--fast); writes a stereo wav: left = what the engine
        heard (the singer, plus the simulated echo with --simulate-bleed), right = machine voice,
        both on the same acoustic timeline.
mac     sounddevice duplex stream on the default (or chosen) mic/speaker; Engine.process runs
        inside the PortAudio callback (no extra buffering). NOT run by the author (needs mic).
device  the ESP32-S3 over USB CDC with the protocol in docs/PLAN.md + firmware/README.md
        (codec: firmware/tools/proto.py via live/proto.py): MIC_AUDIO in; SPK_AUDIO / SERVO /
        LED / PING out; STATUS / PONG (/ optional DIAG) in.

When the engine has an echo guard, every backend first runs the sound check
(echo_link.SoundCheck: ~2 s of soft pink noise out while the mic records, then calibration
while silence is sent) and only then starts the engine. The singer starts after the noise.
"""
import queue
import statistics
import threading
import time

import numpy as np

import proto
from dsp import HOP, SR, write_wav
from echo_link import SoundCheck


def _sleep_until(t_target):
    while True:
        d = t_target - time.perf_counter()
        if d <= 0:
            return
        time.sleep(d - 0.0015 if d > 0.002 else 0)        # last 1.5 ms: spin (macOS sleep overshoots ~1 ms)


def set_interactive_qos():
    """Ask macOS to schedule this thread as QOS_CLASS_USER_INTERACTIVE (P-cores, high priority).
    No root needed. The PortAudio callback thread is already a real-time thread; this is for the
    file and device backends, whose loop runs in an ordinary Python thread. Returns True on success."""
    try:
        import ctypes
        import sys as _sys
        if _sys.platform != 'darwin':
            return False
        lib = ctypes.CDLL('/usr/lib/libSystem.dylib')
        return lib.pthread_set_qos_class_self_np(0x21, 0) == 0     # 0x21 = QOS_CLASS_USER_INTERACTIVE
    except Exception:
        return False


def _stats(a):
    a = np.asarray(a, float)
    if not a.size:
        return None
    return {'p50': float(np.median(a)), 'p99': float(np.percentile(a, 99)), 'p999': float(np.percentile(a, 99.9)),
            'max': float(a.max()), 'over_10ms': int((a > 10).sum()), 'over_20ms': int((a > 20).sum()),
            'over_30ms': int((a > 30).sum()), 'over_40ms': int((a > 40).sum()), 'n': int(a.size)}


# ---------------------------------------------------------------- file
class FileBackend:
    def __init__(self, engine, x, out_path=None, realtime=True, tail=0.5, monitor=False, bleed=None,
                 soundcheck_s=2.0, log=print):
        self.e = engine
        self.x = np.concatenate([np.asarray(x, np.float32), np.zeros(int(tail * SR), np.float32)])
        self.out_path = out_path
        self.realtime = realtime
        self.monitor = monitor
        self.bleed = bleed                     # echo_link.BleedSim or None
        self.soundcheck_s = soundcheck_s
        self.log = log
        self.stop = False
        self.result = None
        self.soundcheck = None

    def _run_soundcheck(self):
        """The sound check through the simulated room (echo_link.BleedSim), block by block with the
        same send -> acoustic -> mic alignment as the song (block i sent after mic block i-1 sounds
        from (i+1)*HOP + L). With no --simulate-bleed the mic only hears a quiet room."""
        e = self.e
        sc = SoundCheck(e.echo, self.soundcheck_s, threaded=False, log=self.log)
        path = self.bleed.fresh_path() if self.bleed else None
        rng = np.random.default_rng(5)
        n_max = sc.nb + sc.gap_blocks + 2
        y = np.zeros((n_max + 3) * HOP + e.L, np.float32)
        k = 0
        while not sc.done:
            if self.bleed:
                mic = self.bleed.step(y[k * HOP:(k + 1) * HOP], path) + self.bleed.room_floor(HOP)
            else:
                mic = 1e-5 * rng.standard_normal(HOP)
            out = sc.feed(mic)
            s = (k + 1) * HOP + e.L
            y[s:s + HOP] = out
            k += 1
        self.soundcheck = sc.result

    def run(self):
        e, x = self.e, self.x
        if e.echo is not None and self.soundcheck_s > 0:
            self._run_soundcheck()
        nb = len(x) // HOP
        y = np.zeros(nb * HOP + e.L + 2 * HOP, np.float32)     # machine voice, acoustic timeline
        mic_all = x.copy() if self.bleed else x
        stream = None
        if self.monitor:                       # listen while it runs (user's choice; never run in tests)
            import sounddevice as sd
            stream = sd.OutputStream(samplerate=SR, channels=2, blocksize=HOP, dtype='float32', latency='low')
            stream.start()
        self.qos = set_interactive_qos() if self.realtime else None
        t0 = time.perf_counter()
        wake_late, late = [], []
        for k in range(nb):
            if self.stop:
                break
            if self.realtime and stream is None:
                target = t0 + (k + 1) * HOP / SR               # block k is complete at this instant
                _sleep_until(target)
                arrive = time.perf_counter()
                wake_late.append((arrive - target) * 1000)
            else:
                arrive = time.perf_counter()
            blk = x[k * HOP:(k + 1) * HOP]
            if self.bleed:                     # her voice + the machine's echo in the simulated room
                blk = blk + self.bleed.step(y[k * HOP:(k + 1) * HOP]).astype(np.float32)
                mic_all[k * HOP:(k + 1) * HOP] = blk
            out = e.process(blk, arrive=arrive if self.realtime else None)
            done = time.perf_counter()
            s = (k + 1) * HOP + e.L
            y[s:s + HOP] = out
            e.drain_ctrl()                                       # servo/LED go to the log only
            budget = HOP / SR * 1000
            if self.realtime and stream is None:
                late.append((done - target) * 1000)
                e.annotate(deadline_ms=budget - (done - target) * 1000, wake_late_ms=(arrive - target) * 1000)
            else:
                e.annotate(deadline_ms=budget - (done - arrive) * 1000)
            if stream is not None:             # play the timeline one block behind: both channels final
                blk2 = np.stack([mic_all[k * HOP:(k + 1) * HOP], y[k * HOP:(k + 1) * HOP]], axis=1)
                stream.write(np.ascontiguousarray(blk2, dtype=np.float32))
        wall = time.perf_counter() - t0
        if stream is not None:
            stream.stop()
            stream.close()
        e.close()
        n = len(x)
        stereo = np.stack([mic_all[:n], y[:n]], axis=1)
        if self.out_path:
            write_wav(self.out_path, stereo)
        self.result = {'wall_s': wall, 'audio_s': nb * HOP / SR, 'rtf': wall / (nb * HOP / SR),
                       'realtime': self.realtime, 'qos_user_interactive': self.qos,
                       # block finished this long after the block was complete (arrival + compute):
                       # a speaker path with an N-frame buffer glitches only when this exceeds N*10 ms
                       'done_after_arrival_ms': _stats(late) if late else None,
                       'wake_late_ms': None if not wake_late else {
                           'p50': float(np.median(wake_late)), 'p99': float(np.percentile(wake_late, 99)),
                           'max': float(np.max(wake_late))},
                       'soundcheck': self.soundcheck,
                       'bleed': None if self.bleed is None else self.bleed.describe()}
        return stereo, self.result


# ---------------------------------------------------------------- mac (sounddevice)
class MacBackend:
    """Duplex stream at 24 kHz, 240-sample blocks; the engine runs in the callback.

    Output latency L is measured from PortAudio's timestamps in the first 50 callbacks
    (outputBufferDacTime - inputBufferAdcTime - block) and given to the engine before the
    machine has any note scheduled. With the echo guard (default for this backend) the stream
    first plays the sound check (pink noise, ~2 s) and calibrates the guard in a worker thread;
    the engine starts after that. Without the guard, use headphones (see echo/).
    """
    def __init__(self, engine, device=None, latency='low', duration=None, record_path=None, soundcheck_s=2.0,
                 log=print):
        self.e = engine
        self.device = device
        self.latency = latency
        self.duration = duration
        self.record_path = record_path
        self.fin, self.fout = np.zeros(0, np.float32), np.zeros(0, np.float32)
        self.meas = []
        self.calls = 0
        self.flags = {'input_overflow': 0, 'output_underflow': 0, 'other': 0, 'fifo_underrun': 0}
        self.rec_in, self.rec_out = [], []
        self.done = threading.Event()
        self.sc = SoundCheck(engine.echo, soundcheck_s, threaded=True, log=log) \
            if engine.echo is not None and soundcheck_s > 0 else None
        self.engine_blocks = 0
        self.in_latency = 0.0

    def _block(self, x, arrive):
        if self.sc is not None and not self.sc.done:
            return self.sc.feed(x)
        self.engine_blocks += 1
        return self.e.process(x, arrive=arrive)

    def callback(self, indata, outdata, frames, time_info, status):
        t_enter = time.perf_counter()
        e = self.e
        if status:
            if status.input_overflow:
                self.flags['input_overflow'] += 1
            if status.output_underflow:
                self.flags['output_underflow'] += 1
            if not (status.input_overflow or status.output_underflow):
                self.flags['other'] += 1
        self.fin = np.concatenate([self.fin, indata[:, 0].astype(np.float32)])
        while len(self.fin) >= HOP:
            y = self._block(self.fin[:HOP], t_enter)
            self.fin = self.fin[HOP:]
            self.fout = np.concatenate([self.fout, y])
        if len(self.fout) >= frames:
            outdata[:, 0] = self.fout[:frames]
            self.fout = self.fout[frames:]
        else:
            outdata.fill(0)
            self.flags['fifo_underrun'] += 1
        if outdata.shape[1] > 1:
            outdata[:, 1:] = outdata[:, :1]
        if self.record_path:                   # left = mic as captured, right = what was played
            self.rec_in.append(indata[:, 0].copy())
            self.rec_out.append(outdata[:, 0].copy())
        self.calls += 1
        adc, dac, cur = time_info.inputBufferAdcTime, time_info.outputBufferDacTime, time_info.currentTime
        if adc > 0 and dac > 0:
            L = dac - adc - frames / SR
            if self.calls <= 50:
                self.meas.append(L)
                if self.calls == 50:
                    e.set_out_latency(statistics.median(self.meas))
            elapsed = time.perf_counter() - t_enter
            if self.sc is None or self.sc.done:
                e.annotate(deadline_ms=(dac - (cur + elapsed)) * 1000, cb_ms=elapsed * 1000)
        if self.sc is None or self.sc.done:
            e.drain_ctrl()
        if self.duration and self.engine_blocks * HOP / SR >= self.duration:
            self.done.set()

    def run(self):
        import sounddevice as sd
        stream = sd.Stream(samplerate=SR, blocksize=HOP, channels=1, dtype='float32', latency=self.latency,
                           device=self.device, callback=self.callback)
        lat_in, lat_out = stream.latency
        self.e.set_out_latency(lat_in + lat_out)                 # first guess; refined from timestamps
        if self.e.midi is not None:
            self.e.midi.clock.in_latency = lat_in
        with stream:
            try:
                while not self.done.wait(0.2):
                    pass
            except KeyboardInterrupt:
                pass
        self.e.close()
        res = {'stream_latency_s': [lat_in, lat_out], 'measured_L_s': statistics.median(self.meas) if self.meas else None,
               'flags': self.flags, 'callbacks': self.calls, 'engine_blocks': self.engine_blocks,
               'soundcheck': None if self.sc is None else self.sc.result}
        if self.record_path and self.rec_in:
            write_wav(self.record_path, np.stack([np.concatenate(self.rec_in), np.concatenate(self.rec_out)], axis=1))
        return res


# ---------------------------------------------------------------- device (ESP32-S3 over USB CDC)
def open_serial_no_toggle(port, timeout=0.02, write_timeout=0.1):
    """Open the CDC port WITHOUT touching DTR/RTS.

    On the ESP32-S3's USB-Serial/JTAG port the DTR/RTS line states can reset the chip (that is
    how flashing tools enter download mode; firmware/README.md). pyserial's open() normally
    drives both lines (DTR first, then RTS) and close() drops them through HUPCL. Here the two
    line updates are disabled for this instance and HUPCL is cleared after opening, so the host
    neither raises nor lowers either line: they stay as the OS left them. (Whatever the macOS
    driver itself does on open() is outside our control; confirm on the board that opening the
    port does not reset it.)"""
    import termios

    import serial

    s = serial.Serial()
    s.port = port
    s.baudrate = 115200                  # ignored by USB CDC
    s.timeout = timeout
    s.write_timeout = write_timeout
    s.dsrdtr = False
    s.rtscts = False
    s._update_dtr_state = lambda: None   # instance attributes shadow the class methods open() calls
    s._update_rts_state = lambda: None
    s.open()                             # open() also discards stale input (tcflush TCIFLUSH)
    try:
        attrs = termios.tcgetattr(s.fd)
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(s.fd, termios.TCSANOW, attrs)
    except termios.error:
        pass
    return s


class LevelKeeper:
    """Keeps the device's speaker jitter-buffer level (hence the speaker latency L and the echo
    path delay the canceller learned) where the pre-fill put it.

    1:1 pacing keeps the level constant, except after an underrun: the device plays silence for
    U ticks, re-primes, and when the host catches up the level is U frames higher than before
    (+U x 10 ms latency). The firmware trims one frame per 0.5 s while the level is above 6
    (SPK_TRIM_ABOVE = SPK_PRIME_FRAMES 4 + 2, firmware/duet_io/config.h);
    the firmware team's fake device does not trim at all. The keeper reads STATUS
    (spk_buffer_frames, every 100 ms) and, once the last 3 reports agree on a level that differs
    from the baseline (minimum of the first 5 reports after the pre-fill), drops (or inserts
    silent) SPK_AUDIO frames to restore it, then waits 5 reports before judging again. Transients
    (the dip during a stall, the burst while the host catches up) never act: they do not last 3
    reports at one level."""

    def __init__(self, enabled=True, settle_s=0.25):
        self.enabled = enabled
        self.settle_s = settle_s
        self.t_arm = None
        self.levels = []
        self.base = None
        self.pending = 0
        self.cooldown = 0
        self.lock = threading.Lock()
        self.actions = []

    def arm(self, t):
        self.t_arm = t

    def see(self, level, t):
        if not self.enabled or self.t_arm is None or t < self.t_arm + self.settle_s:
            return
        with self.lock:
            self.levels.append(level)
            if self.base is None:
                if len(self.levels) >= 5:
                    self.base = min(self.levels[-5:])
                return
            if self.cooldown > 0:
                self.cooldown -= 1
                return
            last = self.levels[-3:]
            if len(set(last)) != 1:              # still moving (a stall in progress, the host catching up)
                return
            d = last[0] - self.base
            if d and self.pending == 0:
                self.pending = d
                self.cooldown = 5
                self.actions.append({'t': round(t - self.t_arm, 3), 'level': self.base + d, 'base': self.base,
                                     'frames_added': -d})

    def take(self):
        """+1: drop this frame; -n: insert n silent frames first; 0: send normally."""
        if not self.pending:
            return 0
        with self.lock:
            if self.pending > 0:
                self.pending -= 1
                return 1
            n, self.pending = self.pending, 0
            return n

    def summary(self):
        return {'enabled': self.enabled, 'baseline_level': self.base, 'actions': self.actions}


class DeviceBackend:
    """The ESP32-S3 over USB CDC.

    Start-up: open the port without touching DTR/RTS; discard whatever the device queued before
    we opened it (sync_s of frames, so the first frame we answer is a fresh one); pre-fill the
    device's speaker jitter buffer with `prefill` silent frames (default 4 = 40 ms: the firmware
    starts playing at 4 (SPK_PRIME_FRAMES) and trims only when the level stays ABOVE 6 for 0.5 s); then answer every
    MIC_AUDIO frame with exactly one SPK_AUDIO frame. The device's mic and speaker clocks are
    locked, so 1:1 pacing keeps the buffer level constant and there is no drift to correct.

    Sequence numbers are per message type in both directions (firmware/README.md 1.5): the host
    numbers SPK_AUDIO, SERVO, LED and PING frames with separate counters (the device counts gaps
    in the SPK_AUDIO seq); incoming MIC_AUDIO gaps = mic frames lost; PONG is matched to its PING
    by seq. STATUS counters are wrapping u16, so underruns/overruns are reported as deltas.

    Shutdown: servo to rest, LED off, wait 50 ms, then DISCARD (tcflush) rather than drain
    anything still unsent: tcdrain()/flush() blocks forever if the device has stopped reading.
    Writes have a short timeout for the same reason; a timed-out write is counted, not retried.
    """

    def __init__(self, engine, port, ping_every=1.0, idle_timeout=2.0, record_path=None, servo_rest=90,
                 prefill=4, sync_s=0.2, soundcheck_s=2.0, in_latency=0.003, log=print, ser=None, keep_level=True,
                 test_stall=None):
        self.e = engine
        self.ser = ser or open_serial_no_toggle(port)
        self.parser = proto.Parser()
        self.q = queue.Queue()
        self.seq_out = proto.SeqCounters()
        self.seq_in = proto.SeqTracker()
        self.ping_every = ping_every
        self.idle_timeout = idle_timeout
        self.record_path = record_path
        self.servo_rest = servo_rest
        self.prefill = int(prefill)
        self.sync_s = sync_s
        self.log = log
        self.status = {}
        self.status_first = None
        self.status_run0 = None
        self.spk_levels = []                   # STATUS spk_buffer_frames seen during the song
        self.diag = None
        self.rtt_ms = []
        self.ping_sent = {}
        self.counts = {'mic': 0, 'mic_discarded': 0, 'spk': 0, 'servo': 0, 'led': 0, 'ping': 0, 'pong': 0,
                       'status': 0, 'diag': 0, 'other': 0, 'backlog_max': 0, 'write_timeouts': 0,
                       'mic_lost_after_sync': 0, 'spk_dropped_by_keeper': 0, 'spk_inserted_by_keeper': 0}
        self._stop = threading.Event()
        self.rec_in, self.rec_out = [], []
        self.sc = SoundCheck(engine.echo, soundcheck_s, threaded=True, log=log) \
            if engine.echo is not None and soundcheck_s > 0 else None
        if engine.midi is not None:
            engine.midi.clock.in_latency = in_latency
        self.keeper = LevelKeeper(enabled=keep_level)
        self.test_stall = test_stall            # tests only: (song frame, seconds) host stall to force an underrun
        self.reader = threading.Thread(target=self._read, daemon=True, name='serial-reader')

    def _send_spk(self, y):
        """One SPK_AUDIO frame, unless the level keeper asks to drop it; plus any silent frames it
        asks to insert."""
        act = self.keeper.take()
        if act > 0:                       # buffer too deep (e.g. after an underrun): skip this frame
            self.counts['spk_dropped_by_keeper'] += 1
            return
        for _ in range(-act):             # buffer too shallow: top it up with silence
            self._send(proto.SPK_AUDIO, proto.audio_payload(np.zeros(HOP, np.float32)))
            self.counts['spk_inserted_by_keeper'] += 1
        self._send(proto.SPK_AUDIO, proto.audio_payload(y))
        self.counts['spk'] += 1

    def _send(self, ftype, payload):
        fr = proto.encode(ftype, self.seq_out.next(ftype), payload)
        try:
            self.ser.write(fr)
        except Exception as ex:                    # SerialTimeoutException: device not reading
            if type(ex).__name__ == 'SerialTimeoutException':
                self.counts['write_timeouts'] += 1
            else:
                raise

    def _read(self):
        while not self._stop.is_set():
            try:
                data = self.ser.read(max(1, self.ser.in_waiting))
            except Exception:                          # port closed / device unplugged
                break
            if not data:
                continue
            now = time.perf_counter()
            for ftype, seq, p in self.parser.feed(data):
                self.seq_in.see(ftype, seq)
                if ftype == proto.MIC_AUDIO:
                    self.counts['mic'] += 1
                    self.q.put((now, seq, proto.audio_from_payload(p)))
                elif ftype == proto.STATUS:
                    self.counts['status'] += 1
                    self.status = proto.parse_status(p)
                    self.keeper.see(self.status['spk_buffer_frames'], now)
                    if self.status_first is None:
                        self.status_first = dict(self.status)
                    if self.status_run0 is not None:
                        self.spk_levels.append(self.status['spk_buffer_frames'])
                elif ftype == proto.PONG:
                    self.counts['pong'] += 1
                    t_sent = self.ping_sent.pop(seq, None)
                    if t_sent is not None:
                        self.rtt_ms.append((now - t_sent) * 1000)
                elif ftype == proto.DIAG:
                    self.counts['diag'] += 1
                    self.diag = proto.parse_diag(p)
                else:
                    self.counts['other'] += 1

    def _ping(self):
        tnow = time.perf_counter()
        seq = self.seq_out.c.get(proto.PING, 0)
        self.ping_sent[seq] = tnow
        self._send(proto.PING, proto.ping_payload(int(tnow * 1e6)))
        self.counts['ping'] += 1
        return tnow

    def run(self, max_frames=None):
        e = self.e
        self.qos = set_interactive_qos()
        self.reader.start()
        self._send(proto.LED, proto.led_payload(proto.LED_OFF))
        self._send(proto.SERVO, proto.servo_payload(self.servo_rest, 0))
        # 1. discard stale frames queued on the device / in the OS before we opened the port
        t_sync = time.perf_counter() + self.sync_s
        while time.perf_counter() < t_sync:
            try:
                self.q.get(timeout=0.05)
                self.counts['mic_discarded'] += 1
            except queue.Empty:
                pass
        while True:
            try:
                self.q.get_nowait()
                self.counts['mic_discarded'] += 1
            except queue.Empty:
                break
        # 2. pre-fill the speaker jitter buffer
        silence = np.zeros(HOP, np.float32)
        for _ in range(self.prefill):
            self._send(proto.SPK_AUDIO, proto.audio_payload(silence))
            self.counts['spk'] += 1
        self.keeper.arm(time.perf_counter())
        t_ping = 0.0
        n = 0
        sc_blocks = 0
        last_seq = None
        try:
            while max_frames is None or n < max_frames:
                try:
                    arrive, seq, x = self.q.get(timeout=self.idle_timeout)
                except queue.Empty:
                    break                                  # device stopped streaming
                backlog = self.q.qsize()
                self.counts['backlog_max'] = max(self.counts['backlog_max'], backlog)
                g = proto.seq_gap(last_seq, seq)        # mic frames lost after start-up (seq set at capture)
                if g:
                    self.counts['mic_lost_after_sync'] += g
                last_seq = seq
                if self.sc is not None and not self.sc.done:          # 3. sound check (guard only)
                    y = self.sc.feed(x)
                    self._send_spk(y)
                    sc_blocks += 1
                    continue
                if self.status_run0 is None:
                    self.status_run0 = dict(self.status) if self.status else {}
                e.queue_depth = backlog
                y = e.process(x, arrive=arrive)                    # 4. the song
                self._send_spk(y)
                for kind, kw in e.drain_ctrl():
                    if kind == 'servo':
                        self._send(proto.SERVO, proto.servo_payload(kw['angle'], kw['speed']))
                        self.counts['servo'] += 1
                    elif kind == 'led':
                        self._send(proto.LED, proto.led_payload(kw['state']))
                        self.counts['led'] += 1
                if time.perf_counter() - t_ping >= self.ping_every:
                    t_ping = self._ping()
                if self.test_stall and n == self.test_stall[0]:
                    time.sleep(self.test_stall[1])
                done = time.perf_counter()
                e.annotate(deadline_ms=HOP / SR * 1000 - (done - arrive) * 1000, host_ms=(done - arrive) * 1000,
                           spk_buf=self.status.get('spk_buffer_frames'), mic_seq=seq)
                if self.record_path:
                    self.rec_in.append(x)
                    self.rec_out.append(y)
                n += 1
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self._send(proto.SERVO, proto.servo_payload(self.servo_rest, 2))
                self._send(proto.LED, proto.led_payload(proto.LED_OFF))
                time.sleep(0.05)
                self.ser.reset_output_buffer()     # discard, never tcdrain (blocks if the device stopped reading)
            except Exception:
                pass
            self._stop.set()
            self.reader.join(timeout=1)
            try:
                self.ser.close()
            except Exception:
                pass
            e.close()
        if self.record_path and self.rec_in:
            # left = mic as received, right = machine voice as sent (sent order, not acoustic time)
            write_wav(self.record_path, np.stack([np.concatenate(self.rec_in), np.concatenate(self.rec_out)], axis=1))
        rtt = self.rtt_ms
        st0, st = self.status_run0 or {}, self.status
        lv = np.array(self.spk_levels) if self.spk_levels else None
        return {'frames': n, 'counts': self.counts, 'link': self.parser.stats(), 'seq': self.seq_in.summary(),
                'device_status': st, 'diag': self.diag, 'prefill': self.prefill, 'soundcheck_blocks': sc_blocks,
                'soundcheck': None if self.sc is None else self.sc.result,
                'spk_underruns_during_song': (proto.u16_delta(st0['spk_underruns'], st['spk_underruns'])
                                              if st0 and st else None),
                'mic_overruns_during_song': (proto.u16_delta(st0['mic_overruns'], st['mic_overruns'])
                                             if st0 and st else None),
                'spk_buffer_frames_during_song': None if lv is None else {
                    'min': int(lv.min()), 'median': float(np.median(lv)), 'max': int(lv.max()),
                    'hist': {int(v): int((lv == v).sum()) for v in np.unique(lv)}},
                'level_keeper': self.keeper.summary(),
                'ping_rtt_ms': None if not rtt else {'median': float(np.median(rtt)), 'max': float(np.max(rtt)), 'n': len(rtt)}}
