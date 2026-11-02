"""The real-time Duet engine: one call per 10 ms block, identical for every audio backend.

Timeline. Block k of microphone audio covers stream samples [k*HOP, (k+1)*HOP); when it has
arrived, "now" = (k+1)*HOP/SR (the same clock follow.py and run_v0.py use). The engine then
returns one block of machine voice that will SOUND at stream samples
[(k+1)*HOP + L, (k+2)*HOP + L), where L = out_latency is the delay from the moment a mic block
is complete to the moment the output block produced from it reaches the speaker (buffering +
converters + transport). The Accompanist is told the same L (its `out_latency`), so a note it
commits at time `at >= now + L` can be rendered sample-exactly; scheduling slack per note =
at - (now + L) (>= 0 means on time).

The processing time of the block is hidden inside L as long as the block is finished before
its deadline (that is what the backends measure as deadline slack), so Accompanist's
`compute` argument is a constant margin (default 0), which also keeps runs deterministic.
"""
import collections
import json
import queue
import threading
import time

import numpy as np

from dsp import HOP, SR
from echo_link import expected_midis
from follow import Accompanist, FeatureTracker, Follower
from voice import StreamingVoice

PARTS = ('features', 'follower', 'accompanist', 'synth', 'echo_guard')
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def note_name(m):
    m = int(round(m))
    return f'{NAMES[m % 12]}{m // 12 - 1}'


# ---------------------------------------------------------------- JSONL log (off the audio thread)
class JsonlLogger:
    def __init__(self, path):
        self.path = path
        self.q = queue.SimpleQueue()
        self.f = open(path, 'w')
        self.n = 0
        self.th = threading.Thread(target=self._run, daemon=True, name='jsonl-log')
        self.th.start()

    def log(self, rec):
        self.q.put(rec)

    def _run(self):
        while True:
            rec = self.q.get()
            if rec is None:
                break
            self.f.write(json.dumps(rec, separators=(',', ':')) + '\n')
            self.n += 1
        self.f.close()

    def close(self):
        self.q.put(None)
        self.th.join(timeout=5)


# ---------------------------------------------------------------- the nod
class NodPlanner:
    """A visible preparatory gesture one beat before the machine comes in (cf. Shimon).

    Lift the head one beat before the entry, drop it so the bottom of the nod lands on the
    entry, return to rest. The entry time is re-predicted every frame until the drop is sent,
    so the gesture follows her tempo. No nod if the engine is not confident it will sing.
    Servo speeds are in degrees per 10 ms (protocol unit); an SG90 manages about 6 deg/10 ms.
    """
    REST, LIFT, DIP = 90, 102, 72
    SPEED_LIFT, SPEED_DOWN, SPEED_BACK = 2, 5, 2
    DOWN_LEAD = 0.10          # 30 deg at 5 deg/10 ms = 60 ms travel + ~40 ms command/servo lag
    HOLD = 0.25

    def __init__(self, engine, mode='first'):
        self.e = engine
        self.mode = mode
        acc = engine.acc
        entries = []
        prev_ph = None
        for j, p in enumerate(acc.plan):
            ph = acc._phrase(p['r'])
            if j == 0 or (mode == 'phrases' and ph != prev_ph):
                entries.append(j)
            prev_ph = ph
        self.entries = entries if mode != 'off' else []
        self.phase = 'idle'
        self.t_down = None
        self.T = None
        self.done = set()
        self.log = []             # (now, angle, speed, entry j, predicted T)

    def _predict(self, j):
        acc = self.e.acc
        v = acc.pred.get(j)
        if v is not None and v['status'] == 'played':
            return v['at']
        return acc.a + acc.b * acc.u(acc.plan[j]['r'])

    def step(self, now):
        if not self.entries:
            return
        acc, fol = self.e.acc, self.e.fol
        pending = [j for j in self.entries if j not in self.done]
        if not pending:
            return
        j = pending[0]
        if self.phase == 'idle':
            v = acc.pred.get(j)
            if j < acc.next_plan and (v is None or v['status'] != 'played'):
                self.done.add(j)              # held / missed: no promise to make
                return
            T = self._predict(j)
            beat = self.e.ref.beat_s * acc.b
            if now > T - self.DOWN_LEAD:      # too late to prepare (e.g. reactive entry)
                if j < acc.next_plan:
                    self.done.add(j)
                return
            confident = fol.conf >= acc.thr and fol.global_left == 0 and acc.unconfirmed < 2
            if now >= T - beat and confident and acc.waiting_phrase is None:
                self._send(now, self.LIFT, self.SPEED_LIFT, j, T)
                self.phase = 'lift'
        elif self.phase == 'lift':
            T = self._predict(j)
            if now >= T - self.DOWN_LEAD:
                self._send(now, self.DIP, self.SPEED_DOWN, j, T)
                self.phase, self.t_down, self.T = 'down', now, T
        elif self.phase == 'down':
            if now >= self.t_down + self.DOWN_LEAD + self.HOLD:
                self._send(now, self.REST, self.SPEED_BACK, j, self.T)
                self.phase = 'idle'
                self.done.add(j)

    def _send(self, now, angle, speed, j, T):
        self.log.append({'t': round(now, 4), 'angle': angle, 'speed': speed, 'entry': j, 'T_pred': round(T, 4)})
        self.e.emit('servo', angle=angle, speed=speed, entry=j, T_pred=round(T, 4))


# ---------------------------------------------------------------- engine
class Engine:
    MIN_IOI = 0.06            # two machine notes never start closer than this (see _collect)

    def __init__(self, ref, out_latency=0.005, compute_margin=0.0, conf_thr=0.55, nod='first',
                 use_profile=True, logger=None, log_every=1, view_hist=500, guard=True, echo=None, midi=None,
                 synth=True, gate_thr=None):
        """echo: an echo/guard.EchoGuard (or None); midi: a midi.MidiOut (or None);
        synth=False: the built-in voice is still rendered (as the echo guard's reference and for the
        log) but process() returns silence, e.g. when Logic Pro is the machine's voice.
        gate_thr: absolute threshold (dBFS) of the evaluation's input level gate
        (eval/duet_eval/features.StreamGate, the frame-by-frame form of apply_gate), applied to the mic
        block after the echo canceller and before the pitch tracker; None (default) = no gate."""
        self.ref = ref
        self.echo = echo
        self.gate_thr = gate_thr
        self.gate = None
        if gate_thr is not None:
            from rehearsal import eval_features
            self.gate = eval_features().StreamGate(float(gate_thr))
        self.midi = midi
        self.synth = synth
        self._ref_prev = np.zeros(HOP)      # the block returned by the previous process() call
        self.ft = FeatureTracker()
        self.fol = Follower(ref.F)
        self.acc = Accompanist(ref.notes, ref.plan, ref.join_r, out_latency=out_latency, conf_thr=conf_thr,
                               bias_live=ref.bias_live, profile=ref.profile() if use_profile else None,
                               breaths=ref.breaths)
        self.compute_margin = compute_margin
        self.voice = StreamingVoice()
        self.k = -1
        self.set_out_latency(out_latency, force=True)
        self.nod = NodPlanner(self, nod)
        self.logger = logger
        self.log_every = max(1, int(log_every))
        self.ctrl = []            # control messages for the backend: (kind, kwargs)
        self.handled = {}         # plan index -> (status, at) already acted on
        self.base_dur = {}
        self.out_notes = {}       # plan index -> dict of what was actually scheduled
        self.events = []
        self.taken_back = []
        self.suppressed = []
        self.guard = guard
        self.frame_ms = []
        self.part_ms = []
        self.deadline_ms = []
        self.queue_depth = 0
        self._rec = None
        self.led = None
        self._led_t = -1.0
        self.hist = collections.deque(maxlen=view_hist)
        self.t_wall0 = None

    # -------- configuration
    def set_out_latency(self, L, force=False):
        """Change the output offset. Only allowed before any machine note exists."""
        if not force and (self.voice.active() or self.out_notes):
            return False
        self.out_latency = float(L)
        self.L = int(round(L * SR))
        self.acc.O = float(L)
        self.voice.pos = (self.k + 2) * HOP + self.L        # first sample of the next output block
        return True

    def emit(self, kind, **kw):
        self.ctrl.append((kind, kw))
        self._event({'ev': kind, **kw})
        if kind == 'servo' and self.midi is not None:
            self.midi.nod(kw['angle'])

    def _event(self, rec):
        rec = {'t': round((self.k + 1) * HOP / SR, 4), **rec}
        self.events.append(rec)
        if self.logger:
            self.logger.log(rec)

    # -------- the per-block path
    def process(self, x, arrive=None):
        """x: HOP mic samples (float32, -1..1). Returns HOP samples of machine voice.
        arrive: time.perf_counter() when this block reached the host (MIDI clock; optional)."""
        if self._rec is not None:
            self._flush()
        self.k += 1
        k = self.k
        now = (k + 1) * HOP / SR
        out_start = (k + 1) * HOP + self.L
        if self.midi is not None and arrive is not None:
            self.midi.clock.observe(k, now, arrive)
        t0 = time.perf_counter()
        ft, fol, acc = self.ft, self.fol, self.acc
        x = np.asarray(x, dtype=np.float64)
        if self.echo is not None:
            # (a) cancel the echo of what we sent after the previous block, then (b) after the
            # pitch tracker, mask frames whose pitch is the machine's own (echo/guard.py)
            x = self.echo.cancel(x, self._ref_prev)
            if self.gate is not None:
                x = self.gate(x)
            te = time.perf_counter()
            ft.step(x)
            tf = time.perf_counter()
            self.echo.mask(ft, expected_midis(self.ref.notes, acc.k))
            t_echo = (te - t0) + (time.perf_counter() - tf)
        else:
            if self.gate is not None:
                x = self.gate(x)
            ft.step(x)
            t_echo = 0.0
        t1 = time.perf_counter()
        fol.step(ft.m[-1], ft.v[-1], ft.nov[-1])
        t2 = time.perf_counter()
        acc.step(now, ft, fol, self.compute_margin)
        t3 = time.perf_counter()
        self._collect(now, out_start)
        self.nod.step(now)
        self._led(now)
        assert self.voice.pos == out_start, (self.voice.pos, out_start)
        y = self.voice.render(HOP)
        self._ref_prev = y if self.synth or self.echo is None or self.echo.aec is None else np.zeros(HOP)
        if not self.synth:
            y = np.zeros(HOP, np.float32)
        t4 = time.perf_counter()
        tot = (t4 - t0) * 1000
        self.frame_ms.append(tot)
        self.part_ms.append(((t1 - t0 - t_echo) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000,
                             t_echo * 1000))
        if k % 2 == 0:
            self.hist.append((round(now, 3), round(fol.ref_time(), 3),
                              round(ft.m[-1], 2) if ft.v[-1] else None, round(fol.conf, 3)))
        if k % self.log_every == 0:
            self._rec = {'k': k, 't': round(now, 4), 'ft': round((t1 - t0 - t_echo) * 1000, 4),
                         'echo': round(t_echo * 1000, 4), 'fol': round((t2 - t1) * 1000, 4),
                         'acc': round((t3 - t2) * 1000, 4), 'syn': round((t4 - t3) * 1000, 4), 'tot': round(tot, 4),
                         'pos': round(fol.ref_time(), 3), 'conf': round(fol.conf, 3), 'b': round(acc.b, 4),
                         'v': int(ft.v[-1]), 'midi': round(ft.m[-1], 2) if ft.v[-1] else None,
                         'nv': self.voice.active(), 'q': self.queue_depth}
        return y

    def annotate(self, **kw):
        """Backends add per-frame facts known only after process() (e.g. deadline slack)."""
        if 'deadline_ms' in kw and kw['deadline_ms'] is not None:
            self.deadline_ms.append(kw['deadline_ms'])
        if self._rec is not None:
            self._rec.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items()})

    def _flush(self):
        if self.logger and self._rec is not None:
            self.logger.log(self._rec)
        self._rec = None

    def _collect(self, now, out_start):
        """Turn the Accompanist's new decisions into scheduled notes on the output timeline."""
        acc = self.acc
        for j in [j for j in self.handled if j not in acc.pred]:        # false entry taken back
            if self.handled[j][0] == 'played' and j in self.out_notes:
                applied = self.voice.release_now(j)
                q = self.out_notes[j]
                self.taken_back.append({'j': j, 'at': self.handled[j][1], 't': now, 'dur': applied,
                                        'start_sample': q['start_sample'], 'h': q['h']})
                self._event({'ev': 'take_back', 'j': j, 'dur': applied})
                if self.midi is not None and applied is not None:
                    self.midi.set_end(j, q['start_sample'] / SR + applied)
                self.out_notes.pop(j, None)
            del self.handled[j]
        for j, v in acc.pred.items():
            key = (v['status'], v.get('at'))
            if self.handled.get(j) == key:
                continue
            self.handled[j] = key
            p = acc.plan[j]
            if v['status'] != 'played':
                self._event({'ev': v['status'], 'j': j, 'h': p['h'], 'r': round(p['r'], 3)})
                continue
            at = v['at']
            if self.guard:
                prev = [q for q in self.out_notes.values() if q['j'] < j and abs(at - q['at']) < self.MIN_IOI]
                if prev:
                    # Two notes decided for (almost) the same instant: after a pause the Accompanist
                    # can commit a stale prediction together with the re-entry note. Sing only the
                    # lower-index (re-entry) note instead of a two-note blip. Rendering-only guard;
                    # Accompanist decisions are unchanged (see README, proposed follow.py change).
                    self.suppressed.append({'j': j, 'at': at, 'with': prev[0]['j']})
                    self._event({'ev': 'suppressed', 'j': j, 'at': round(at, 4), 'with': prev[0]['j']})
                    continue
            i0 = int(round(at * SR))
            base = p['dur'] * v['b']
            dur = max(base, 0.08)
            later = [q for q in self.out_notes.values() if q['at'] > at]
            if later:                                   # (only if decisions arrive out of order)
                dur = max(min(base, min(q['at'] for q in later) - at - 0.02), 0.08)
            start = self.voice.schedule(i0, dur, p['h'], tag=j)
            if self.midi is not None:
                self.midi.note(j, start / SR, start / SR + dur, p['h'])
            prev = [q for q in self.out_notes.values() if q['at'] < at]
            if prev:                                    # legato: cut the previous note 20 ms early
                q = max(prev, key=lambda q: q['at'])
                q['dur'] = self.voice.set_dur(q['j'], max(min(q['base'], at - q['at'] - 0.02), 0.08)) or q['dur']
                if self.midi is not None:
                    self.midi.set_end(q['j'], q['start_sample'] / SR + q['dur'])
            slack = (i0 - out_start) / SR
            rec = {'j': j, 'at': at, 'start_sample': start, 'late_ms': (start - i0) / SR * 1000, 'dur': dur,
                   'base': base, 'h': p['h'], 'r': p['r'], 'sent': now, 'slack_ms': slack * 1000}
            self.out_notes[j] = rec
            self._event({'ev': 'note', 'j': j, 'h': p['h'], 'name': note_name(p['h']), 'at': round(at, 4),
                         'slack_ms': round(slack * 1000, 2), 'late_ms': round(rec['late_ms'], 2),
                         'b': round(v['b'], 4)})

    def _led(self, now):
        """Status LED: off before the machine is armed, on while confident, blink while holding."""
        acc, fol = self.acc, self.fol
        if not acc.plan or fol.ref_time() < acc.plan[0]['r'] - 2.0:
            want = 0
        elif fol.conf >= acc.thr and fol.global_left == 0 and acc.unconfirmed < 2:
            want = 1
        else:
            want = 2
        if want != self.led and now - self._led_t >= 0.3:
            self.led, self._led_t = want, now
            self.emit('led', state=want)

    def drain_ctrl(self):
        c, self.ctrl = self.ctrl, []
        return c

    # -------- reporting
    def status(self):
        acc, fol = self.acc, self.fol
        if fol.global_left > 0:
            return 'searching'
        if acc.waiting_phrase is not None or acc.unconfirmed >= 2 or fol.conf < acc.thr:
            return 'holding'
        if self.voice.active():
            return 'singing'
        return 'following'

    def view_state(self):
        """Snapshot for the live view (called from the HTTP thread; reads only)."""
        acc, fol, ft = self.acc, self.fol, self.ft
        now = (self.k + 1) * HOP / SR
        nxt = None
        if acc.next_plan < len(acc.plan):
            j = acc.next_plan
            p = acc.plan[j]
            T = acc.a + acc.b * acc.u(p['r'])
            nxt = {'j': j, 'h': p['h'], 'name': note_name(p['h']), 'r': round(p['r'], 3), 'eta': round(T - now, 3)}
        fm = self.frame_ms[-500:]
        counts = collections.Counter(v['status'] for v in list(acc.pred.values()))
        notes = sorted(self.out_notes.values(), key=lambda q: q['at'])[-12:]
        return {
            't': round(now, 3), 'status': self.status(), 'ref_t': round(fol.ref_time(), 3),
            'ref_dur': round(self.ref.duration, 3), 'conf': round(fol.conf, 3), 'b': round(acc.b, 4),
            'tempo_pct': round(100.0 / acc.b, 1) if acc.b else None,
            'her_midi': round(ft.m[-1], 2) if ft.v and ft.v[-1] else None,
            'her_name': note_name(ft.m[-1]) if ft.v and ft.v[-1] else None,
            'next': nxt, 'played': counts.get('played', 0), 'held': counts.get('held', 0),
            'missed': counts.get('missed', 0), 'resets': fol.resets,
            'servo': self.nod.log[-1] if self.nod.log else None, 'nod_phase': self.nod.phase, 'led': self.led,
            'frame_ms': {'last': round(fm[-1], 3) if fm else None,
                         'p50': round(float(np.median(fm)), 3) if fm else None,
                         'max': round(float(np.max(fm)), 3) if fm else None},
            'queue': self.queue_depth, 'out_latency_ms': round(self.out_latency * 1000, 1),
            'harmony': self.ref.meta.get('harmony'), 'echo_guard': self.echo is not None,
            'machine_midi': (None if self.echo is None or self.echo.pm is None or
                             self.echo.pm.machine_midi != self.echo.pm.machine_midi else round(self.echo.pm.machine_midi, 2)),
            'machine': [{'j': q['j'], 'at': round(q['at'], 3), 'dur': round(q['dur'], 3), 'h': q['h'],
                         'r': round(q['r'], 3)} for q in notes],
            'hist': list(self.hist)[-300:],
        }

    def reference_view(self):
        return {'notes': [{'on': round(n['on'], 3), 'off': round(n['off'], 3), 'midi': n['midi']} for n in self.ref.notes],
                'plan': [{'r': round(p['r'], 3), 'dur': round(p['dur'], 3), 'h': p['h']} for p in self.acc.plan],
                'breaths': [round(b, 3) for b in self.ref.breaths], 'key': self.ref.key,
                'join_r': round(self.ref.join_r, 3), 'duration': round(self.ref.duration, 3)}

    def summary(self):
        fm = np.array(self.frame_ms) if self.frame_ms else np.array([np.nan])
        parts = np.array(self.part_ms) if self.part_ms else np.zeros((1, 5))
        counts = collections.Counter(v['status'] for v in self.acc.pred.values())
        slack = [q['slack_ms'] for q in self.out_notes.values()]
        dl = np.array(self.deadline_ms) if self.deadline_ms else None
        return {
            'frames': self.k + 1, 'audio_s': round((self.k + 1) * HOP / SR, 3),
            'frame_ms': {'p50': float(np.median(fm)), 'p99': float(np.percentile(fm, 99)), 'p999': float(np.percentile(fm, 99.9)),
                         'max': float(fm.max()), 'max_excl_first10': float(fm[10:].max()) if fm.size > 10 else float(fm.max()),
                         'mean': float(fm.mean())},
            'part_ms_mean': dict(zip(PARTS, [float(v) for v in parts.mean(axis=0)])),
            'part_ms_max': dict(zip(PARTS, [float(v) for v in parts.max(axis=0)])),
            'notes': {'played': counts.get('played', 0), 'held': counts.get('held', 0), 'missed': counts.get('missed', 0),
                      'taken_back': len(self.taken_back), 'suppressed': len(self.suppressed), 'plan': len(self.acc.plan)},
            'slack_ms': {'min': float(min(slack)) if slack else None, 'median': float(np.median(slack)) if slack else None},
            'late_notes': sum(1 for q in self.out_notes.values() if q['late_ms'] > 0),
            'deadline_ms': None if dl is None else {'min': float(dl.min()), 'p1': float(np.percentile(dl, 1)),
                                                     'misses': int((dl < 0).sum())},
            'nods': self.nod.log, 'resets': self.fol.resets, 'out_latency_s': self.out_latency,
            'echo_guard': None if self.echo is None else self.echo.stats(),
            'midi': None if self.midi is None else self.midi.stats(), 'synth': self.synth,
            'harmony': self.ref.meta.get('harmony'), 'key': self.ref.key,
            'input_gate_thr_db': self.gate_thr, 'bias_live': self.acc.bias_live,
        }

    def close(self):
        self._flush()
