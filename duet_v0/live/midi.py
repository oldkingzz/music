"""MIDI out: Logic Pro (or any synth) as the machine's voice, plus MIDI-file export of a run.

Live output (python-rtmidi on CoreMIDI)
---------------------------------------
MidiOut opens either a VIRTUAL output port (a CoreMIDI source other apps can listen to; default
name "Duet") or connects to an existing destination (`--midi-connect`, e.g. an IAC bus or a test
sink). Each machine note decided by the Accompanist becomes note_on at its scheduled time and
note_off at its (possibly later shortened) end; the servo nod is mirrored as a control change.

Timing. The engine works in stream time t (seconds of mic audio since the run started). A note
the engine plans for stream time `at` must SOUND at the wall-clock instant

    wall(at) = at + W0,   W0 = min over recent blocks (arrival_wall_k - now_k) - in_latency

(arrival_wall_k: time.perf_counter() when mic block k reached the host; now_k = (k+1)*10 ms;
the sliding minimum over the last 2 s removes OS jitter, which only ever delays an arrival, and
follows slow clock drift between the audio clock and the Mac clock; in_latency = capture ->
arrival: 0 in file mode, PortAudio's input latency on the mac, ~3 ms over USB). The message is
sent `midi_latency` earlier than that (the synth's own MIDI-in -> sound delay; Logic at a
64-128 sample buffer is a few ms, see LOGIC.md), by a scheduler thread that sleeps until 2 ms
before the due time and then spins. The engine is run with an output latency L of at least
in_latency + midi_latency + 8 ms so every note is decided before it has to be sent.

MIDI files
----------
export_midi() writes a type-1 file: tempo track, the singer's melody, the machine's harmony
(and the nod as CC on the harmony track). One constant tempo is used, so tick <-> second is
exact everywhere (960 ticks per beat; at 84 bpm one tick is 0.74 ms): positions in the file are
the real seconds of the performance. render_midi() reads it back and renders every track with
the built-in voice (dsp.render_voice), which doubles as a check of the file.
"""
import collections
import heapq
import threading
import time

import numpy as np

from dsp import SR

NOTE_ON, NOTE_OFF, CC = 0x90, 0x80, 0xB0


class StreamClock:
    """Stream time (s) -> time.perf_counter() wall time of the moment that stream instant is
    acoustic (captured at the mic / heard at the speaker)."""

    def __init__(self, in_latency=0.0, window_blocks=200):
        self.in_latency = float(in_latency)
        self.window = window_blocks
        self.q = collections.deque()          # monotone deque of (k, d) for a sliding minimum
        self.n = 0
        self.history = []                     # (k, offset) every 100 blocks, for the log

    def observe(self, k, now, arrive):
        d = arrive - now
        q = self.q
        while q and q[-1][1] >= d:
            q.pop()
        q.append((k, d))
        while q[0][0] <= k - self.window:
            q.popleft()
        self.n += 1
        if k % 100 == 0:
            self.history.append((k, round(self.offset(), 6)))

    def ready(self):
        return bool(self.q)

    def offset(self):
        return self.q[0][1] - self.in_latency

    def wall(self, t):
        return t + self.offset()


class MidiOut:
    def __init__(self, name='Duet', connect=None, channel=0, midi_latency=0.010, in_latency=0.0, velocity=90,
                 nod_cc=20, log_events=True):
        import rtmidi
        self.rt = rtmidi.MidiOut(name='Duet live')
        if connect:
            ports = self.rt.get_ports()
            idx = next((i for i, p in enumerate(ports) if connect in p), None)
            if idx is None:
                raise RuntimeError(f'MIDI destination containing {connect!r} not found; have {ports}')
            self.rt.open_port(idx, name='Duet live out')
            self.port_desc = f'connected to destination {ports[idx]!r}'
        else:
            self.rt.open_virtual_port(name)
            self.port_desc = f'virtual source {name!r}'
        self.name = name
        self.ch = int(channel) & 0x0F
        self.latency = float(midi_latency)
        self.vel = int(velocity)
        self.nod_cc = nod_cc
        self.clock = StreamClock(in_latency)
        self.lock = threading.Condition()
        self.heap = []
        self.ctr = 0
        self.notes = {}                       # uid -> dict(j, pitch, on, off, state, ver)
        self.by_j = {}                        # plan index -> uid of its latest note
        self.uid = 0
        self.sounding = {}                    # pitch -> uid currently on
        self.log = [] if log_events else None
        self.sent = collections.Counter()
        self.late = []                        # (j, how late the note_on was when it was scheduled, s)
        self.running = True
        self.th = threading.Thread(target=self._run, daemon=True, name='midi-out')
        self.th.start()

    # ---------------- engine side (called from the audio thread; cheap)
    def _push(self, t, kind, j, ver):
        self.ctr += 1
        heapq.heappush(self.heap, (t, self.ctr, kind, j, ver))

    def note(self, j, at, end, pitch):
        """Machine note j: sounds from stream time `at` to `end`."""
        if not self.clock.ready():
            return
        on = self.clock.wall(at) - self.latency
        off = self.clock.wall(end) - self.latency
        nowc = time.perf_counter()
        if on < nowc:
            self.late.append((j, nowc - on))
        with self.lock:
            old = self.by_j.get(j)                     # a plan note decided again: retire the old one
            if old is not None and self.notes[old]['state'] == 'pending':
                self.notes[old]['state'] = 'off'
            elif old is not None and self.notes[old]['state'] == 'on':
                self.notes[old]['ver'] += 1
                self._push(time.perf_counter(), 'off', old, self.notes[old]['ver'])
            self.uid += 1
            u = self.uid
            self.by_j[j] = u
            self.notes[u] = {'j': j, 'pitch': int(round(pitch)), 'on': on, 'off': max(off, on + 0.005),
                             'state': 'pending', 'ver': 0, 'at': at, 'end': end}
            self._push(on, 'on', u, 0)
            self._push(self.notes[u]['off'], 'off', u, 0)
            self.lock.notify()

    def set_end(self, j, end):
        """The note's end changed (legato cut by the next note, or a false entry taken back)."""
        with self.lock:
            u = self.by_j.get(j)
            n = self.notes.get(u)
            if n is None or n['state'] == 'off':
                return
            off = self.clock.wall(end) - self.latency
            n['end'] = end
            n['ver'] += 1
            if n['state'] == 'pending' and off <= n['on'] + 0.005:
                n['state'] = 'off'                    # taken back before it ever sounded
                if self.log is not None:
                    self.log.append({'j': j, 'kind': 'cancel', 'due': off, 'sent': time.perf_counter()})
                return
            n['off'] = max(off, n['on'] + 0.005)
            self._push(n['off'], 'off', u, n['ver'])
            self.lock.notify()

    def cc(self, value, number=None):
        """Control change now (the nod)."""
        number = self.nod_cc if number is None else number
        if number is None:
            return
        self.rt.send_message([CC | self.ch, int(number) & 0x7F, int(np.clip(value, 0, 127))])
        self.sent['cc'] += 1
        if self.log is not None:
            self.log.append({'kind': 'cc', 'cc': int(number), 'value': int(value), 'sent': time.perf_counter()})

    def nod(self, angle):
        if self.nod_cc is not None:
            self.cc(int(round(float(angle) / 180.0 * 127)))

    # ---------------- scheduler thread
    def _run(self):
        while True:
            with self.lock:
                if not self.running and not self.heap:
                    return
                if not self.heap:
                    self.lock.wait(0.05)
                    continue
                t, _, kind, j, ver = self.heap[0]
                dt = t - time.perf_counter()
                if dt > 0.002 and self.running:
                    self.lock.wait(dt - 0.002)       # woken early by a new, earlier event
                    continue
                heapq.heappop(self.heap)
                n = self.notes.get(j)
                if n is None or ver != n['ver'] and kind == 'off':
                    continue
                if kind == 'on' and n['state'] != 'pending':
                    continue
                if kind == 'off' and n['state'] != 'on':
                    if n['state'] == 'pending':        # off came before on (should not happen): drop both
                        n['state'] = 'off'
                    continue
            while time.perf_counter() < t:            # last <= 2 ms: spin (sleep overshoots on macOS)
                pass
            with self.lock:
                n = self.notes[j]
                if kind == 'on':
                    if n['state'] != 'pending':
                        continue
                    p = n['pitch']
                    prev = self.sounding.get(p)
                    if prev is not None and self.notes[prev]['state'] == 'on':     # same pitch re-struck
                        self.rt.send_message([NOTE_OFF | self.ch, p, 0])
                        self.notes[prev]['state'] = 'off'
                        self.sent['off_restrike'] += 1
                    ts = time.perf_counter()
                    self.rt.send_message([NOTE_ON | self.ch, p, self.vel])
                    n['state'] = 'on'
                    self.sounding[p] = j
                    self.sent['on'] += 1
                else:
                    if n['state'] != 'on' or ver != n['ver']:
                        continue
                    ts = time.perf_counter()
                    self.rt.send_message([NOTE_OFF | self.ch, n['pitch'], 0])
                    n['state'] = 'off'
                    if self.sounding.get(n['pitch']) == j:
                        del self.sounding[n['pitch']]
                    self.sent['off'] += 1
                if self.log is not None:
                    self.log.append({'j': n['j'], 'kind': kind, 'pitch': n['pitch'], 'due': t, 'sent': ts,
                                     'at': n['at'], 'end': n['end']})

    def close(self, drain_s=1.0):
        """Let pending events play out (up to drain_s), then all notes off and close the port."""
        t_end = time.perf_counter() + drain_s
        while time.perf_counter() < t_end:
            with self.lock:
                if not self.heap:
                    break
            time.sleep(0.01)
        with self.lock:
            self.running = False
            self.heap.clear()
            self.lock.notify()
        self.th.join(timeout=1.0)
        for p, j in list(self.sounding.items()):
            self.rt.send_message([NOTE_OFF | self.ch, p, 0])
        if self.sent['on']:                                       # only if we ever started a note
            self.rt.send_message([CC | self.ch, 123, 0])         # all notes off
            time.sleep(0.01)
        self.rt.close_port()
        del self.rt

    def stats(self):
        on = [e for e in (self.log or []) if e.get('kind') == 'on']
        err = np.array([(e['sent'] - e['due']) * 1000 for e in on]) if on else np.zeros(0)
        return {'port': self.port_desc, 'channel': self.ch + 1, 'midi_latency_s': self.latency,
                'in_latency_s': self.clock.in_latency, 'sent': dict(self.sent), 'scheduled': len(self.notes),
                'clock_observations': self.clock.n,
                'decided_too_late': len(self.late),
                'send_minus_due_ms': None if not err.size else {'median': float(np.median(err)),
                                                                'max': float(err.max()), 'min': float(err.min())},
                'clock_offset_history': self.clock.history[-20:]}


# ---------------------------------------------------------------- MIDI files
TPB = 960


def _ticks(t, spb):
    return int(round(t / spb * TPB))


def export_midi(path, melody, harmony, bpm=120.0, title='Duet', nods=None, melody_name='Singer (melody)',
                harmony_name='Duet (harmony)', nod_cc=20):
    """melody / harmony: lists of (start_s, dur_s, midi). nods: list of (t_s, angle_deg).
    Times are real seconds of the performance; one tempo, so seconds map exactly to ticks."""
    import mido
    spb = 60.0 / bpm
    mf = mido.MidiFile(type=1, ticks_per_beat=TPB)
    t0 = mido.MidiTrack()
    t0.append(mido.MetaMessage('track_name', name=title, time=0))
    t0.append(mido.MetaMessage('set_tempo', tempo=int(round(spb * 1e6)), time=0))
    t0.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    mf.tracks.append(t0)

    def track(name, notes, ch, program, extra=()):
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage('track_name', name=name, time=0))
        tr.append(mido.Message('program_change', channel=ch, program=program, time=0))
        ev = []
        for s, d, m in notes:
            a, b = _ticks(s, spb), max(_ticks(s + d, spb), _ticks(s, spb) + 1)
            ev.append((b, 0, mido.Message('note_off', channel=ch, note=int(round(m)), velocity=0)))
            ev.append((a, 1, mido.Message('note_on', channel=ch, note=int(round(m)), velocity=90)))
        for t, msg in extra:
            ev.append((_ticks(t, spb), 2, msg))
        ev.sort(key=lambda e: (e[0], e[1]))        # at equal ticks: offs first, then ons
        last = 0
        for tick, _, msg in ev:
            tr.append(msg.copy(time=tick - last))
            last = tick
        tr.append(mido.MetaMessage('end_of_track', time=0))
        return tr

    mf.tracks.append(track(melody_name, melody, 0, 52))          # GM 53 "Choir Aahs"
    extra = [(t, mido.Message('control_change', channel=1, control=nod_cc, value=int(round(a / 180 * 127))))
             for t, a in (nods or [])]
    mf.tracks.append(track(harmony_name, harmony, 1, 53, extra))  # GM 54 "Voice Oohs"
    mf.save(str(path))
    return mf


def read_midi_notes(path):
    """-> {track name: [(start_s, dur_s, midi)]} using the file's own tempo map."""
    import mido
    mf = mido.MidiFile(str(path))
    tempo = 500000
    for msg in mf.tracks[0]:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break
    spt = tempo / 1e6 / mf.ticks_per_beat
    out = {}
    for tr in mf.tracks[1:]:
        name, t, on, notes = None, 0, {}, []
        for msg in tr:
            t += msg.time
            if msg.type == 'track_name':
                name = msg.name
            elif msg.type == 'note_on' and msg.velocity > 0:
                on[msg.note] = t
            elif msg.type in ('note_off', 'note_on') and msg.note in on:
                s = on.pop(msg.note)
                notes.append((s * spt, (t - s) * spt, msg.note))
        out[name] = sorted(notes)
    return out


def render_midi(path, n_samples=None, levels=(0.18, 0.22)):
    """Render each track of a MIDI file with the built-in voice -> (samples, n_tracks) float32."""
    from dsp import render_voice
    tracks = read_midi_notes(path)
    end = max((s + d for notes in tracks.values() for s, d, _ in notes), default=0.0) + 0.5
    n = n_samples or int(end * SR)
    cols = [render_voice(notes, n, level=lv) for notes, lv in zip(tracks.values(), levels)]
    return np.stack(cols, axis=1).astype(np.float32), list(tracks)
